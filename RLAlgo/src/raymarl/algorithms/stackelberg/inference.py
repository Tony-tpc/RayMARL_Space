"""Portable actor parameters with the same policy-output probability interface."""

from pathlib import Path

import torch

from .rl_module import StackelbergRLModule


def export_actors(algorithm, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    modules = {}
    for key in algorithm.config.agent_order:
        module = algorithm.env_runner.module[key]
        modules[key] = {
            "observation_space": module.observation_space,
            "action_space": module.action_space,
            "model_config": module.model_config,
            "actor": {
                name: value.detach().cpu().clone() for name, value in module.actor.state_dict().items()
            },
        }
    torch.save({"kind": "stackelberg-actors", "modules": modules}, path / "actors.pt")
    return str(path)


class ActorBundle:
    def __init__(self, path, device="cpu"):
        data = torch.load(Path(path) / "actors.pt", map_location=device, weights_only=False)
        if data["kind"] != "stackelberg-actors":
            raise ValueError("Expected a current Stackelberg actor artifact")
        self.device = torch.device(device)
        self.modules = {}
        for key, item in data["modules"].items():
            module = StackelbergRLModule(
                observation_space=item["observation_space"],
                action_space=item["action_space"],
                model_config=item["model_config"],
                inference_only=True,
            )
            module.actor.load_state_dict(item["actor"], strict=True)
            self.modules[key] = module.to(self.device).eval()

    def sample(self, agent_id, observation, bounds=None, *, deterministic=True):
        with torch.no_grad():
            obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            legal = (
                None
                if bounds is None
                else torch.as_tensor(bounds, dtype=torch.float64, device=self.device).unsqueeze(0)
            )
            sample = self.modules[agent_id].draw(obs, legal, deterministic=deterministic)
        return {key: value[0].cpu().numpy().copy() for key, value in sample.items()}

    def uc_controls(self, observation, *, deterministic=True):
        return self.sample("uc", observation, deterministic=deterministic)["raw_latents"]

    def consumer_action(self, agent_id, observation, bounds, *, deterministic=True):
        return self.sample(agent_id, observation, bounds, deterministic=deterministic)["actions"]
