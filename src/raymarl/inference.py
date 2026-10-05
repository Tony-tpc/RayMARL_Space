"""Actor-only inference artifact; no environment or critic construction."""

import torch

from .networks import ActorModule


def export_actor_bundle(algo, path):
    """Export identity-mapped PPO policies without value-network weights."""
    if hasattr(algo, "export_actors"):
        return algo.export_actors(path)
    from pathlib import Path

    policies = sorted(algo.config.policies, key=str)
    modules = [algo.get_module(a) for a in policies]
    data = {
        "agent_order": tuple(policies),
        "observation_spaces": [
            m.observation_space["local"] if getattr(m, "central", False) else m.observation_space
            for m in modules
        ],
        "action_spaces": [m.action_space for m in modules],
        "hidden_dim": modules[0].model_config.get("hidden_dim", 64),
        "actors": [
            {
                k: v.detach().cpu().clone()
                for k, v in m.state_dict().items()
                if k.startswith(("encoder.", "head."))
            }
            for m in modules
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)


class ActorBundle:
    def __init__(self, path, device="cpu"):
        # Only load locally trusted PyTorch artifacts.
        data = torch.load(path, map_location=device, weights_only=False)
        self.agent_order = data["agent_order"]
        self.device = torch.device(device)
        self.actors = []
        for o, a, state in zip(
            data["observation_spaces"], data["action_spaces"], data["actors"], strict=True
        ):
            actor = ActorModule(
                observation_space=o,
                action_space=a,
                inference_only=True,
                model_config={"hidden_dim": data["hidden_dim"]},
            )
            actor.load_state_dict(state, strict=True)
            self.actors.append(actor.to(self.device))

    def actions(self, observations, deterministic=True):
        with torch.no_grad():
            result = {}
            for aid, actor in zip(self.agent_order, self.actors):
                x = actor.encoder.layout.tensor([observations[aid]], self.device)
                encoded = actor.distribution(x).draw(deterministic=deterministic)[0]
                tree = actor.layout.decode_features(encoded)

                def unpack(v):
                    if isinstance(v, dict):
                        return {k: unpack(x) for k, x in v.items()}
                    if isinstance(v, tuple):
                        return tuple(unpack(x) for x in v)
                    x = v[0].cpu().numpy()
                    return x.item() if x.ndim == 0 else x

                result[aid] = unpack(tree)
            return result
