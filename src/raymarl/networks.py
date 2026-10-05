"""Private policy networks and shared-attention critics."""

import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from torch import nn

from .distributions import HybridDistribution, bound_distribution
from .spaces import ActionLayout, ObservationLayout


class Encoder(nn.Module):
    def __init__(self, space, hidden, project=True):
        super().__init__()
        self.layout = ObservationLayout(space)
        self.history_lstm = (
            nn.LSTM(self.layout.history_shape[1], hidden, batch_first=True)
            if (self.layout.history_shape)
            else None
        )
        width = self.layout.current_dim + (hidden if self.history_lstm else 0)
        self.feature_dim = hidden if project else width
        self.output = (
            nn.Sequential(nn.Linear(width, hidden), nn.LeakyReLU(), nn.Linear(hidden, hidden), nn.LeakyReLU())
            if project
            else nn.Identity()
        )

    def forward(self, obs):
        x = obs["current"]
        if self.history_lstm:
            h, _ = self.history_lstm(obs["history"])
            mask = obs["history_mask"]
            # Histories are right-padded. Last valid timestep is used.
            index = (mask.sum(-1).long() - 1).clamp_min(0)
            h = h[torch.arange(h.shape[0], device=h.device), index]
            h = h * (mask.sum(-1) > 0).unsqueeze(-1)
            x = torch.cat((x, h), -1)
        return self.output(x)


class ActorModule(TorchRLModule):
    def setup(self):
        hidden = self.model_config.get("hidden_dim", 64)
        self.encoder = Encoder(self.observation_space, hidden)
        self.layout = ActionLayout(self.action_space)
        self.head = nn.Linear(hidden, self.layout.param_dim)

    def distribution(self, obs):
        return HybridDistribution(self.head(self.encoder(obs)), action_space=self.action_space)

    def _forward(self, batch, **kwargs):
        return {
            Columns.ACTION_DIST_INPUTS: self.head(
                self.encoder(self.encoder.layout.torch_batch(batch[Columns.OBS]))
            )
        }

    def get_train_action_dist_cls(self):
        return bound_distribution(self.action_space)

    get_inference_action_dist_cls = get_train_action_dist_cls
    get_exploration_action_dist_cls = get_train_action_dist_cls


class AttentionCritic(nn.Module):
    def __init__(self, observation_spaces, action_spaces, hidden=64, mode="attention"):
        super().__init__()
        self.mode, self.n = mode, len(observation_spaces)
        self.encoders = nn.ModuleList(Encoder(s, hidden, project=False) for s in observation_spaces)
        self.action_encoders = nn.ModuleList(
            nn.Sequential(nn.Linear(enc.feature_dim + ActionLayout(a).feature_dim, hidden), nn.LeakyReLU())
            for enc, a in zip(self.encoders, action_spaces)
        )
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Sequential(nn.Linear(hidden, hidden, bias=False), nn.LeakyReLU())
        context = (
            hidden if mode in ("attention", "uniform") else (hidden * (self.n - 1) if mode == "joint" else 0)
        )
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden + context, hidden), nn.LeakyReLU(), nn.Linear(hidden, 1))
            for _ in observation_spaces
        )

    def encode_observations(self, observations):
        return [enc(o) for enc, o in zip(self.encoders, observations)]

    def forward(self, observations, actions, return_attention=False, encoded=None):
        encoded = self.encode_observations(observations) if encoded is None else encoded
        e = torch.stack(
            [g(torch.cat((o, a), -1)) for g, o, a in zip(self.action_encoders, encoded, actions)], 1
        )
        weights = e.new_zeros((e.shape[0], self.n, self.n))
        if self.mode in ("attention", "uniform"):
            if self.n == 1:
                x = torch.zeros_like(e)
            else:
                logits = self.q(e) @ self.k(e).transpose(-2, -1)
                mask = torch.eye(self.n, device=e.device, dtype=torch.bool)
                if self.mode == "uniform":
                    logits = torch.zeros_like(logits)
                weights = logits.masked_fill(mask, -torch.inf).softmax(-1)
                x = weights @ self.v(e)
            inputs = [torch.cat((e[:, i], x[:, i]), -1) for i in range(self.n)]
        elif self.mode == "joint":
            inputs = [
                torch.cat([e[:, i]] + [e[:, j] for j in range(self.n) if j != i], -1) for i in range(self.n)
            ]
        else:
            inputs = [e[:, i] for i in range(self.n)]
        result = torch.stack([head(x).squeeze(-1) for head, x in zip(self.heads, inputs)], -1)
        return (result, weights) if return_attention else result


class JointModule(MultiRLModule):
    """RLlib module collection; training-only networks have one state owner."""

    def __init__(self, observation_spaces, action_spaces, hidden=64, mode="attention", twin=False, **kwargs):
        from ray.rllib.core.rl_module.rl_module import RLModuleSpec

        specs = {
            f"agent_{i}": RLModuleSpec(
                module_class=ActorModule,
                observation_space=o,
                action_space=a,
                model_config={"hidden_dim": hidden},
            )
            for i, (o, a) in enumerate(zip(observation_spaces, action_spaces))
        }
        super().__init__(rl_module_specs=specs, inference_only=False)
        self.critic = AttentionCritic(observation_spaces, action_spaces, hidden, mode)
        self.critic2 = AttentionCritic(observation_spaces, action_spaces, hidden, mode) if twin else None

    @property
    def actors(self):
        return list(self._rl_modules.values())

    def to(self, device):
        for actor in self.actors:
            actor.to(device)
        self.critic.to(device)
        if self.critic2:
            self.critic2.to(device)
        return self

    def get_state(self, *args, **kwargs):
        state = {
            "actors": {k: m.state_dict() for k, m in self._rl_modules.items()},
            "critic": self.critic.state_dict(),
        }
        if self.critic2:
            state["critic2"] = self.critic2.state_dict()
        return state

    def set_state(self, state):
        for k, v in state["actors"].items():
            self._rl_modules[k].load_state_dict(v, strict=True)
        self.critic.load_state_dict(state["critic"], strict=True)
        if self.critic2:
            self.critic2.load_state_dict(state["critic2"], strict=True)
