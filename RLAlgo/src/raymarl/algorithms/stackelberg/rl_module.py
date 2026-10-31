"""Local policies and decision-time values for the two-stage energy game."""

import math

import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.distribution.torch.torch_distribution import (
    TorchCategorical,
    TorchDiagGaussian,
    TorchMultiDistribution,
)
from ray.rllib.core.rl_module.apis.inference_only_api import InferenceOnlyAPI
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from torch import nn
from torch.distributions import Beta
from torch.nn import functional as F

from .popart import register_popart

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


class ValueNetwork(nn.Module):
    """Normalize each state's hidden features before Tanh."""

    def __init__(self, input_dim, hidden_dim, linear_scales=None):
        super().__init__()
        self.linear_scales = tuple((int(index), float(scale)) for index, scale in (linear_scales or {}).items())
        if any(not 0 <= index < input_dim or not math.isfinite(scale) or scale <= 0
               for index, scale in self.linear_scales):
            raise ValueError("Value input scales require valid indices and finite positive divisors")
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(),
        )
        self.head = nn.Linear(hidden_dim, 1)

    def transform(self, inputs):
        transformed = signed_log1p(inputs)
        for index, scale in self.linear_scales:
            transformed[..., index] = inputs[..., index] / scale
        return transformed

    def forward(self, inputs):
        return self.head(self.encoder(self.transform(inputs))).squeeze(-1)

    @torch.no_grad()
    def saturation(self, inputs):
        hidden, fractions = self.transform(inputs), []
        for layer in self.encoder:
            hidden = layer(hidden)
            if isinstance(layer, nn.Tanh):
                fractions.append(float((hidden.abs() > 0.99).float().mean()))
        return fractions


def shift_interval(bounds, cut):
    """Equation (14), evaluated in double precision at the sampled cut."""
    _, due, _, backlog, capacity, baseline, rigid, flex, dt, allow, _, _, _ = bounds.unbind(-1)
    low = -torch.minimum(flex - cut, backlog / dt)
    high = torch.minimum(torch.minimum(baseline - rigid - cut, flex - cut), (capacity - backlog) / dt)
    high = torch.where(allow.bool(), high, high.clamp_max(0.0))
    return low, torch.where(due > 0, -due, high)


class ActiveDiagGaussian(TorchDiagGaussian):
    """RLlib Gaussian with point masses for inactive/zero-width coordinates.

    Sampling and the active coordinate densities use RLlib's Gaussian. Only
    the sum excludes coordinates whose physical interval has collapsed.
    """

    def __init__(self, loc, scale, active):
        self.active = active
        super().__init__(torch.where(active, loc, 0.0), torch.where(active, scale, 1.0))

    def sample(self):
        return torch.where(self.active, super().sample(), 0.0)

    def logp(self, value):
        return torch.where(self.active, self._dist.log_prob(value), 0.0).sum(-1)

    def entropy(self):
        return torch.where(self.active, self._dist.entropy(), 0.0).sum(-1)


class StackelbergRLModule(TorchRLModule, InferenceOnlyAPI, ValueFunctionAPI):
    """UC masked categorical/conditional Gaussian; Consumer conditional Beta.

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
        self.soc_input = self.model_config.get("soc_input") if self.role == "uc" else None
        if self.role == "uc":
            self.actor["head"] = nn.Linear(hidden, 6)  # grid, curtailment, DR
            self.actor["mode"] = nn.Linear(hidden + 3, 3)
            self.actor["storage"] = nn.Linear(hidden + 6, 6)
            if self.soc_input is not None:
                index, low, high = (self.soc_input[key] for key in ("index", "min", "max"))
                if not (0 <= index < obs_dim and math.isfinite(low) and math.isfinite(high) and high >= low):
                    raise ValueError("soc_input requires an observation index and finite ordered energy bounds")
                # Zero construction consumes no RNG and leaves all existing initial weights unchanged.
                self.actor["mode_soc"] = nn.ParameterDict({"weight": nn.Parameter(torch.zeros(3, 1))})
            initial_mean = torch.as_tensor(
                self.model_config.get("uc_initial_mean", [0.5, 0.5, 0.2, 0.2, 0.5, 0.5])
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
                for name, indices in (("head", [0, 1, 5]), ("storage", [2, 3, 4])):
                    self.actor[name].weight.mul_(self.model_config.get("uc_head_scale", 1.0))
                    self.actor[name].bias[:3].copy_(initial_mean[indices])
                    self.actor[name].bias[3:].copy_(initial_std.log().expand(6)[indices])
                self.actor["mode"].weight.mul_(self.model_config.get("uc_head_scale", 1.0))
                self.actor["mode"].bias.zero_()
        else:
            self.actor["cut"] = nn.Linear(hidden, 2)
            self.actor["shift"] = nn.Linear(hidden + 1, 2)
            self.actor["market"] = nn.Linear(hidden + 2, 4)
        if not self.inference_only:
            self.value = ValueNetwork(
                self.model_config["critic_dim"], self.model_config.get("value_hidden_dim", hidden),
                self.model_config.get("value_input_scales"))
            if self.model_config.get("online_popart", False):
                register_popart(self.value, self._initial_value_scale)

    @property
    def value_scale(self):
        if hasattr(self, "value") and hasattr(self.value, "popart_scale"):
            return float(self.value.popart_scale)
        return self._initial_value_scale

    @value_scale.setter
    def value_scale(self, scale):
        self._initial_value_scale = float(scale)

    @torch.no_grad()
    def set_state(self, state):
        super().set_state(state)
        # RLlib's generic conversion downcasts float64; keep exact EMA recovery.
        if hasattr(self, "value") and hasattr(self.value, "popart_scale"):
            for name in ("mean_ema", "second_ema", "mass"):
                key = "value.popart_" + name
                if key in state:
                    buffer = getattr(self.value, "popart_" + name)
                    buffer.copy_(torch.as_tensor(state[key], device=buffer.device, dtype=buffer.dtype))

    def get_non_inference_attributes(self):
        # Native runner broadcasts also serve GAE sampling and need fresh values.
        # Actor-only exports construct inference_only modules and save actor.*.
        return []

    def _features(self, obs):
        inputs = signed_log1p(obs.float())
        if self.soc_input is not None:
            inputs[..., self.soc_input["index"]] = self.centered_soc(obs)
        return self.actor["encoder"](inputs)

    def centered_soc(self, obs):
        """Actor-only capacity fraction; raw observations and all Critics retain kWh."""
        if self.soc_input is None:
            return None
        low, high = self.soc_input["min"], self.soc_input["max"]
        energy = obs[..., self.soc_input["index"]].float()
        return 2 * ((energy - low) / (high - low)).clamp(0, 1) - 1 if high > low else torch.zeros_like(energy)

    @staticmethod
    def _beta(parameters):
        # Concentrations above one keep the bounded density finite at both ends.
        alpha, beta = (F.softplus(parameters) + 1.0).unbind(-1)
        return Beta(alpha, beta)

    @staticmethod
    def _gaussian(parameters, active):
        mean, log_std = parameters.double().chunk(2, -1)
        return ActiveDiagGaussian(mean, log_std.clamp(-5.0, 2.0).exp(), active)

    def _uc_base(self, features, bounds):
        return self._gaussian(self.actor["head"](features), bounds[..., [0, 1, 5]] > 0)

    def _uc_mode(self, features, base, bounds, soc=None):
        unit = base.clamp(0, 1)
        grid = unit[..., 0] * bounds[..., 0]
        der = bounds[..., 1] * (1 - unit[..., 1])
        mask = torch.stack((torch.ones_like(grid, dtype=torch.bool),
                            (bounds[..., 2] > 0) & (grid + der > 0), bounds[..., 4] > 0), -1)
        logits = self.actor["mode"](torch.cat((features, unit.float()), -1))
        if self.soc_input is not None:
            if soc is None:
                raise ValueError("The UC mode distribution requires the original observation's SOC")
            logits = logits + F.linear(soc[..., None], self.actor["mode_soc"]["weight"])
        return TorchCategorical.from_logits(logits.double().masked_fill(~mask, -torch.inf)), mask

    def _uc_storage_parameters(self, features, base, mode):
        # Like RLlib's autoregressive_actions_rlm: condition on the sampled
        # categorical action. PPO supplies the stored action, never a new draw.
        inputs = torch.cat((features, base.clamp(0, 1).float(), F.one_hot(mode.long(), 3).float()), -1)
        return self.actor["storage"](inputs)

    @staticmethod
    def _uc_intervals(bounds, controls, mode):
        unit = controls.clamp(0, 1)
        charge = mode == 1
        der_cap = torch.where(charge, torch.minimum(bounds[..., 1] * (1 - unit[..., 1]),
                                                    bounds[..., 2]), 0.0)
        grid_cap = torch.where(charge, torch.minimum(bounds[..., 0] * unit[..., 0],
                              (bounds[..., 2] - unit[..., 2] * der_cap).clamp_min(0)), 0.0)
        upper = torch.stack((bounds[..., 0], bounds[..., 1], der_cap, grid_cap,
                             torch.where(mode == 2, bounds[..., 4], 0.0), bounds[..., 5]), -1)
        return torch.stack((torch.zeros_like(upper), upper), -1)

    def _uc_distribution(self, features, controls, bounds, mode, soc=None):
        base = controls[..., [0, 1, 5]]
        base_dist = self._uc_base(features, bounds)
        mode_dist, mask = self._uc_mode(features, base, bounds, soc)
        intervals = self._uc_intervals(bounds, controls, mode)
        storage = self._gaussian(self._uc_storage_parameters(features, base, mode),
                                 intervals[..., [2, 3, 4], 1] > 0)
        return TorchMultiDistribution((base_dist, mode_dist, storage)), intervals, mask

    def _draw_uc(self, features, bounds, deterministic=False, soc=None, *, actions_only=False):
        bounds = bounds.double()
        base_dist = self._uc_base(features, bounds)
        base = base_dist.loc if deterministic else base_dist.sample()
        mode_dist, _ = self._uc_mode(features, base, bounds, soc)
        mode = mode_dist.to_deterministic().sample() if deterministic else mode_dist.sample()
        storage_dist = self._gaussian(self._uc_storage_parameters(features, base, mode),
                                      torch.ones_like(base, dtype=torch.bool))
        storage = storage_dist.loc if deterministic else storage_dist.sample()
        controls = torch.stack((base[..., 0], base[..., 1], storage[..., 0], storage[..., 1],
                                storage[..., 2], base[..., 2]), -1)
        if actions_only:
            intervals = self._uc_intervals(bounds, controls, mode)
            controls = torch.where(intervals[..., 1] > 0, controls, 0.0)
            return torch.cat((mode[..., None].to(controls.dtype), controls), -1)
        distribution, intervals, mask = self._uc_distribution(features, controls, bounds, mode, soc)
        active = intervals[..., 1] > 0
        controls = torch.where(active, controls, 0.0)
        logp = distribution.logp((controls[..., [0, 1, 5]], mode, controls[..., [2, 3, 4]]))
        return {
            Columns.ACTIONS: controls.clamp(0, 1) * intervals[..., 1],
            "raw_latents": controls,
            "storage_mode": mode,
            "mode_mask": mask,
            "bounds": bounds,
            Columns.ACTION_LOGP: logp,
            "logp_parts": self._uc_logp_parts(distribution, controls, mode),
            "conditional_bounds": intervals,
            "zero_width_mask": ~active,
        }

    @staticmethod
    def _uc_logp_parts(distribution, controls, mode):
        values = (controls[..., [0, 1, 5]], mode, controls[..., [2, 3, 4]])
        return torch.stack([dist.logp(value) for dist, value in
                            zip(distribution._flat_child_distributions, values, strict=True)], -1)

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

    def draw_action(self, obs, bounds, deterministic=False):
        """Action-only evaluation with identical conditional sampling order."""
        features = self._features(obs)
        if self.role == "uc":
            return self._draw_uc(features, bounds, deterministic, self.centered_soc(obs), actions_only=True)
        return self._draw_consumer(features, bounds, deterministic)[0]

    def draw_conditional_mode(self, obs, bounds, base, mode):
        """Sample native storage power after fixing a feasible base and mode."""
        features = self._features(obs)
        mode = torch.as_tensor(mode, dtype=torch.long, device=features.device).expand(len(obs))
        _, valid = self._uc_mode(features, base, bounds, self.centered_soc(obs))
        if not valid.gather(-1, mode[:, None]).all():
            raise ValueError("Cannot intervene on an infeasible storage mode")
        storage = self._gaussian(self._uc_storage_parameters(features, base, mode),
                                 torch.ones_like(base, dtype=torch.bool)).sample()
        controls = torch.stack((base[:, 0], base[:, 1], storage[:, 0], storage[:, 1],
                                storage[:, 2], base[:, 2]), -1)
        intervals = self._uc_intervals(bounds, controls, mode)
        controls = torch.where(intervals[..., 1] > 0, controls, 0.)
        return torch.cat((mode[:, None].to(controls.dtype), controls), -1)

    def draw(self, obs, bounds=None, deterministic=False):
        features = self._features(obs)
        if self.role == "uc":
            return self._draw_uc(features, bounds, deterministic, self.centered_soc(obs))
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
            mode = batch["storage_mode"].long()
            soc = self.centered_soc(batch[Columns.OBS])
            distribution, _, _ = self._uc_distribution(features, controls, batch["bounds"].double(), mode, soc)
            return {
                Columns.ACTION_LOGP: distribution.logp((controls[..., [0, 1, 5]], mode,
                                                       controls[..., [2, 3, 4]])),
                "entropy": distribution.entropy(),
                "logp_parts": self._uc_logp_parts(distribution, controls, mode),
                "mode_probs": distribution._flat_child_distributions[1]._dist.probs,
                "mode_inputs": torch.cat((features, controls[..., [0, 1, 5]].clamp(0, 1).float()), -1),
            }
        logp, entropy = self._consumer_logp_entropy(features, controls, batch["conditional_bounds"])
        return {Columns.ACTION_LOGP: logp, "entropy": entropy}

    def policy_entropy(self, batch):
        """Differentiable Monte Carlo entropy under the current conditional policy."""
        features = self._features(batch[Columns.OBS])
        if self.role == "uc":
            # Unbiased score-function entropy gradient for the full conditional
            # mixed distribution, including mode-dependent active dimensions.
            with torch.no_grad():
                sample = self._draw_uc(features, batch["bounds"], soc=self.centered_soc(batch[Columns.OBS]))
            logp = self.evaluate_actions({Columns.OBS: batch[Columns.OBS], **sample})[Columns.ACTION_LOGP]
            surrogate = -(logp.detach() + 1) * logp
            return -logp.detach() + surrogate - surrogate.detach()
        _, controls, intervals = self._draw_consumer(features, batch["bounds"], reparameterize=True)
        return self._consumer_logp_entropy(features, controls, intervals)[1]

    def compute_values(self, batch, embeddings=None):
        """Return original reward units; only the network's output unit is scaled."""
        inputs = batch["critic_obs"]
        if hasattr(self.value, "popart_scale"):
            return self.value(inputs.float()) * self.value.popart_scale + self.value.popart_shift
        return self.value(inputs.float()) * self.value_scale

    def _forward(self, batch, **kwargs):
        return self.draw(batch[Columns.OBS], batch.get("bounds"))

    def _forward_exploration(self, batch, **kwargs):
        return self._forward(batch, **kwargs)

    def _forward_inference(self, batch, **kwargs):
        return self.draw(batch[Columns.OBS], batch.get("bounds"), deterministic=True)

    def _forward_train(self, batch, **kwargs):
        return {**self.evaluate_actions(batch), Columns.VF_PREDS: self.compute_values(batch)}
