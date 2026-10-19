"""Local policies and decision-time values for the two-stage energy game."""

import math

import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.inference_only_api import InferenceOnlyAPI
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from torch import nn
from torch.distributions import Beta, Normal
from torch.nn import functional as F

BOUNDS_FIELDS = (
    "cut_max_kw",
    "due_kw",
    "due_kwh",
    "backlog_kwh",
    "backlog_limit_kwh",
    "baseline_kw",
    "rigid_kw",
    "flex_kw",
    "dt_hours",
    "allow_new_shift",
    "der_bid_min",
    "der_bid_max",
    "uc_tariff",
)


def signed_log1p(values):
    """Fixed input scaling; collection and PPO reuse exactly the same transform."""
    return values.sign() * torch.log1p(values.abs())


def mlp(input_dim, hidden_dim, output_dim=None):
    layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
    if output_dim is not None:
        layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


def shift_interval(bounds, cut):
    """Equation (14), evaluated in double precision at the sampled cut."""
    _, due, _, backlog, capacity, baseline, rigid, flex, dt, allow, _, _, _ = bounds.unbind(-1)
    low = -torch.minimum(flex - cut, backlog / dt)
    high = torch.minimum(torch.minimum(baseline - rigid - cut, flex - cut), (capacity - backlog) / dt)
    high = torch.where(allow.bool(), high, high.clamp_max(0.0))
    return low, torch.where(due > 0, -due, high)


class StackelbergRLModule(TorchRLModule, InferenceOnlyAPI, ValueFunctionAPI):
    """UC raw Gaussian or one consumer's conditional Beta policy.

    Only ``obs`` enters an actor. UC value reads that same aggregate observation;
    consumer value reads ``critic_obs`` from the trusted post-announcement state.
    ``draw`` returns UC internal controls or consumer physical actions. The
    experiment's explicit output adapter maps UC controls to six physical values.
    """

    def setup(self):
        self.role = self.model_config["role"]
        if self.role not in {"uc", "consumer"}:
            raise ValueError("Policy role must be uc or consumer")
        hidden = self.model_config.get("hidden_dim", 64)
        scale = self.model_config.get("value_scale", 1.0)
        self.value_scale = float(scale[self.role] if isinstance(scale, dict) else scale)
        if not math.isfinite(self.value_scale) or self.value_scale <= 0:
            raise ValueError("value_scale must be finite and positive")
        obs_dim = self.observation_space.shape[0]
        self.actor = nn.ModuleDict({"encoder": mlp(obs_dim, hidden)})
        if self.role == "uc":
            self.actor["head"] = nn.Linear(hidden, 12)
            initial_mean = torch.as_tensor(
                self.model_config.get("uc_initial_mean", [0.5, 0.5, 0.5, 0.5, 0.0, 0.5])
            )
            initial_std = torch.as_tensor(self.model_config.get("uc_initial_std", math.exp(-0.5)))
            if (
                initial_mean.shape != (6,)
                or initial_std.shape not in ((), (6,))
                or not torch.isfinite(initial_mean).all()
                or not torch.isfinite(initial_std).all()
                or (initial_std <= 0).any()
            ):
                raise ValueError(
                    "UC initialization requires six finite means and positive scalar/six-vector std"
                )
            with torch.no_grad():
                self.actor["head"].weight.mul_(self.model_config.get("uc_head_scale", 1.0))
                self.actor["head"].bias[:6].copy_(initial_mean)
                self.actor["head"].bias[6:].copy_(initial_std.log().expand(6))
        else:
            self.actor["cut"] = nn.Linear(hidden, 2)
            self.actor["shift"] = nn.Linear(hidden + 1, 2)
            self.actor["market"] = nn.Linear(hidden + 2, 4)
        if not self.inference_only:
            critic_dim = obs_dim if self.role == "uc" else self.model_config["critic_dim"]
            self.value = mlp(critic_dim, hidden, 1)

    def get_non_inference_attributes(self):
        # Native runner broadcasts also serve GAE sampling and need fresh values.
        # Actor-only exports construct inference_only modules and save actor.*.
        return []

    def _features(self, obs):
        return self.actor["encoder"](signed_log1p(obs.float()))

    @staticmethod
    def _beta(parameters):
        # Concentrations above one keep the bounded density finite at both ends.
        alpha, beta = (F.softplus(parameters) + 1.0).unbind(-1)
        return Beta(alpha, beta)

    def _normal(self, features):
        mean, log_std = self.actor["head"](features).chunk(2, -1)
        return Normal(mean, log_std.clamp(-5.0, 2.0).exp())

    def _consumer_distributions(self, features, controls):
        cut = controls[..., :1].to(features.dtype)
        shift = controls[..., 1:2].to(features.dtype)
        first = self._beta(self.actor["cut"](features))
        second = self._beta(self.actor["shift"](torch.cat((features, cut), -1)))
        last = self.actor["market"](torch.cat((features, cut, shift), -1)).reshape(*features.shape[:-1], 2, 2)
        return first, second, self._beta(last)

    @staticmethod
    def _coordinate(distribution, low, high, deterministic, reparameterize=False):
        unit = (
            distribution.mean
            if deterministic
            else (distribution.rsample() if reparameterize else distribution.sample())
        )
        width = high - low
        unit = torch.where(width > 0.0, unit.double(), 0.0)
        physical = low + width * unit
        return unit, physical

    def _draw_consumer(self, features, bounds, deterministic=False, reparameterize=False):
        bounds = bounds.double()
        zero = torch.zeros_like(bounds[..., 0])
        cut_max = bounds[..., 0]
        cut_dist = self._beta(self.actor["cut"](features))
        cut_u, cut = self._coordinate(cut_dist, zero, cut_max, deterministic, reparameterize)
        low, high = shift_interval(bounds, cut)
        shift_dist = self._beta(self.actor["shift"](torch.cat((features, cut_u[..., None].float()), -1)))
        shift_u, shift = self._coordinate(shift_dist, low, high, deterministic, reparameterize)
        market_inputs = torch.cat((features, cut_u[..., None].float(), shift_u[..., None].float()), -1)
        market_dist = self._beta(self.actor["market"](market_inputs).reshape(*features.shape[:-1], 2, 2))
        market_low = torch.stack((zero, bounds[..., 10]), -1)
        market_high = torch.stack((zero + 1.0, bounds[..., 11]), -1)
        market_u, market = self._coordinate(
            market_dist, market_low, market_high, deterministic, reparameterize
        )
        controls = torch.cat((cut_u[..., None], shift_u[..., None], market_u), -1)
        actions = torch.cat((cut[..., None], shift[..., None], market), -1)
        lows = torch.cat((zero[..., None], low[..., None], market_low), -1)
        highs = torch.cat((cut_max[..., None], high[..., None], market_high), -1)
        intervals = torch.stack((lows, highs), -1)
        return actions, controls, intervals

    def _consumer_logp_entropy(self, features, controls, intervals):
        first, second, market = self._consumer_distributions(features, controls)
        widths = intervals[..., 1] - intervals[..., 0]
        active = widths > 0.0
        # Zero-width coordinates are point masses. Evaluate their unused Beta
        # branch at 1/2 to avoid 0 * infinity in both forward and backward passes.
        unit = torch.where(active, controls, 0.5)
        logp = torch.cat(
            (
                first.log_prob(unit[..., 0])[..., None],
                second.log_prob(unit[..., 1])[..., None],
                market.log_prob(unit[..., 2:]),
            ),
            -1,
        )
        entropy = torch.cat((first.entropy()[..., None], second.entropy()[..., None], market.entropy()), -1)
        log_width = torch.where(active, widths, 1.0).log()
        return (
            torch.where(active, logp - log_width, 0.0).sum(-1),
            torch.where(active, entropy + log_width, 0.0).sum(-1),
        )

    def draw(self, obs, bounds=None, deterministic=False):
        features = self._features(obs)
        if self.role == "uc":
            distribution = self._normal(features)
            controls = distribution.mean if deterministic else distribution.sample()
            return {
                Columns.ACTIONS: controls,
                "raw_latents": controls,
                Columns.ACTION_LOGP: distribution.log_prob(controls).sum(-1),
                "conditional_bounds": controls.new_zeros((*controls.shape, 2)),
                "zero_width_mask": torch.zeros_like(controls, dtype=torch.bool),
            }
        actions, controls, intervals = self._draw_consumer(features, bounds, deterministic)
        logp, _ = self._consumer_logp_entropy(features, controls, intervals)
        return {
            Columns.ACTIONS: actions,
            "raw_latents": controls,
            Columns.ACTION_LOGP: logp,
            "conditional_bounds": intervals,
            "zero_width_mask": intervals[..., 1] == intervals[..., 0],
        }

    def evaluate_actions(self, batch):
        features = self._features(batch[Columns.OBS])
        controls = batch["raw_latents"]
        if self.role == "uc":
            distribution = self._normal(features)
            return {
                Columns.ACTION_LOGP: distribution.log_prob(controls).sum(-1),
                "entropy": distribution.entropy().sum(-1),
            }
        logp, entropy = self._consumer_logp_entropy(features, controls, batch["conditional_bounds"])
        return {Columns.ACTION_LOGP: logp, "entropy": entropy}

    def policy_entropy(self, batch):
        """Differentiable Monte Carlo entropy under the current conditional policy."""
        features = self._features(batch[Columns.OBS])
        if self.role == "uc":
            return self._normal(features).entropy().sum(-1)
        _, controls, intervals = self._draw_consumer(features, batch["bounds"], reparameterize=True)
        return self._consumer_logp_entropy(features, controls, intervals)[1]

    def compute_values(self, batch, embeddings=None):
        """Return original reward units; only the network's output unit is scaled."""
        inputs = batch[Columns.OBS] if self.role == "uc" else batch["critic_obs"]
        return self.value(signed_log1p(inputs.float())).squeeze(-1) * self.value_scale

    def _forward(self, batch, **kwargs):
        return self.draw(batch[Columns.OBS], batch.get("bounds"))

    def _forward_exploration(self, batch, **kwargs):
        return self._forward(batch, **kwargs)

    def _forward_inference(self, batch, **kwargs):
        return self.draw(batch[Columns.OBS], batch.get("bounds"), deterministic=True)

    def _forward_train(self, batch, **kwargs):
        return {**self.evaluate_actions(batch), Columns.VF_PREDS: self.compute_values(batch)}
