"""Actor-only artifacts contain no critic, value, target or environment."""

import importlib
from pathlib import Path

import torch

from .common.models import ActorModule


def export_actor_bundle(algo, path):
    policies = [algo.config.agent_to_module[aid] for aid in algo.config.agent_order]
    modules = [algo.get_module(mid) for mid in policies]
    data = {
        "schema": 2,
        "agent_order": algo.config.agent_order,
        "module_types": [(type(m).__module__, type(m).__name__) for m in modules],
        "observation_spaces": [
            m.observation_space["local"] if getattr(m, "central", False) else m.observation_space
            for m in modules
        ],
        "action_spaces": [m.action_space for m in modules],
        "model_configs": [m.model_config for m in modules],
        "actors": [
            {
                key: value.detach().cpu().clone()
                for key, value in m.state_dict().items()
                if key.startswith(("encoder.", "head."))
            }
            for m in modules
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    return str(path)


class ActorBundle:
    def __init__(self, path, device="cpu"):
        data = torch.load(path, map_location=device, weights_only=False)
        self.agent_order = data["agent_order"]
        self.device = torch.device(device)
        self.actors = []
        types = data.get(
            "module_types", [(ActorModule.__module__, ActorModule.__name__)] * len(self.agent_order)
        )
        configs = data.get(
            "model_configs", [{"hidden_dim": data.get("hidden_dim", 64)}] * len(self.agent_order)
        )
        for space, action, state, kind, config in zip(
            data["observation_spaces"], data["action_spaces"], data["actors"], types, configs, strict=True
        ):
            cls = getattr(importlib.import_module(kind[0]), kind[1])
            # PPO export has the local space only, so use the policy-only base.
            if getattr(cls, "central", False) or hasattr(cls, "value_class"):
                cls = ActorModule
            actor = cls(
                observation_space=space, action_space=action, inference_only=True, model_config=config
            )
            actor.load_state_dict(state, strict=True)
            self.actors.append(actor.to(self.device))

    def actions(self, observations, deterministic=True):
        with torch.no_grad():
            result = {}
            for aid, actor in zip(self.agent_order, self.actors):
                x = actor.encoder.layout.tensor([observations[aid]], self.device)
                if hasattr(actor, "policy_action"):
                    features = actor.policy_action(x)
                else:
                    features = actor.distribution(x).draw(deterministic=deterministic)[0]
                tree = actor.layout.decode_features(features)

                def unpack(v):
                    if isinstance(v, dict):
                        return {key: unpack(value) for key, value in v.items()}
                    if isinstance(v, tuple):
                        return tuple(unpack(value) for value in v)
                    a = v[0].cpu().numpy()
                    return a.item() if a.ndim == 0 else a

                result[aid] = unpack(tree)
            return result
