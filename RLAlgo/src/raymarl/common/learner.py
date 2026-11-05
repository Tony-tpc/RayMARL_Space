"""Native TorchLearner plumbing; algorithm losses live in individual packages."""

import copy
import random

import numpy as np
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.torch.torch_learner import TorchLearner
from ray.rllib.utils.metrics import ALL_MODULES

from .connectors import CRITIC_ID, validate_joint_rows
from .targets import validate_finite


class PhasedTorchLearner(TorchLearner):
    """Critic then actor updates using RLlib's registered optimizer/gradient lifecycle."""

    optimizer_type = torch.optim.Adam

    def build(self):
        super().build()
        self.steps = 0
        self._phase = "critic"

    def configure_optimizers_for_module(self, module_id, config=None):
        module = self.module[module_id]
        actor_params = module.online_parameters() if hasattr(module, "online_parameters") else []
        if actor_params:
            self.register_optimizer(
                module_id=module_id,
                optimizer_name="actor",
                optimizer=self.optimizer_type(actor_params, lr=config.actor_lr),
                params=actor_params,
                lr_or_lr_schedule=config.actor_lr,
            )
        critic_params = list(module.critic.parameters()) if hasattr(module, "critic") else []
        if critic_params:
            self.register_optimizer(
                module_id=module_id,
                optimizer_name="critic",
                optimizer=self.optimizer_type(critic_params, lr=config.critic_lr),
                params=critic_params,
                lr_or_lr_schedule=config.critic_lr,
            )

    def _uncompiled_update(self, batch, **kwargs):
        self._phase = "critic"
        _, critic_losses, _ = super()._uncompiled_update(batch, **kwargs)
        self._phase = "actor"
        frozen = [
            p
            for module in self.module.values()
            if hasattr(module, "critic")
            for p in module.critic.parameters()
        ]
        for p in frozen:
            p.requires_grad_(False)
        try:
            fwd, actor_losses, info = super()._uncompiled_update(batch, **kwargs)
        finally:
            for p in frozen:
                p.requires_grad_(True)
            self._phase = "critic"
        for module in self.module.values():
            if hasattr(module, "update_targets"):
                module.update_targets(self.config.tau)
        self.steps += 1
        self.metrics.log_value((ALL_MODULES, "num_updates"), self.steps, window=1)
        return (
            fwd,
            {
                mid: critic_losses.get(mid, 0) + actor_losses.get(mid, 0)
                for mid in set(critic_losses) | set(actor_losses)
            },
            info,
        )

    def compute_gradients(self, loss_per_module, **kwargs):
        if any(not torch.isfinite(loss).all() for loss in loss_per_module.values()):
            raise FloatingPointError("Non-finite loss; optimizer step rejected")
        gradients = super().compute_gradients(loss_per_module, **kwargs)
        if any(g is not None and not torch.isfinite(g).all() for g in gradients.values()):
            raise FloatingPointError("Non-finite gradient; optimizer step rejected")
        return gradients

    def postprocess_gradients(self, gradients_dict):
        clip = self.config.critic_grad_clip if self._phase == "critic" else self.config.actor_grad_clip
        for name, optimizer in self._named_optimizers.items():
            params = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
            if params:
                norm = torch.nn.utils.clip_grad_norm_(params, clip, error_if_nonfinite=True)
                self.metrics.log_value(
                    (self._optimizer_name_to_module[name], self._phase + "_grad_norm"),
                    norm.detach(),
                    window=1,
                )
        return {pid: p.grad for pid, p in self._params.items() if p.grad is not None}

    def get_state(self, *args, **kwargs):
        state = super().get_state(*args, **kwargs)
        if not kwargs.get("inference_only", False):
            state["raymarl_signature"] = self.signature()
            state["raymarl_runtime"] = {
                "steps": self.steps,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if self._device.type == "cuda" else None,
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
            }
        return copy.deepcopy(state)

    def signature(self):
        return (
            type(self).__name__,
            tuple(
                (mid, module.observation_space, module.action_space, module.model_config)
                for mid, module in self.module.items()
            ),
        )

    def set_state(self, state):
        if "raymarl_signature" in state and state["raymarl_signature"] != self.signature():
            raise ValueError("Checkpoint algorithm/model layout mismatch")
        validate_finite(state)
        state = copy.deepcopy(state)
        saved_seq = state.get("weights_seq_no", 0)
        # Explicit restores must also restore older weights, not only newer broadcasts.
        state["weights_seq_no"] = 0
        super().set_state(state)
        self._weights_seq_no = saved_seq
        runtime = state.get("raymarl_runtime")
        if runtime:
            self.steps = runtime["steps"]
            torch.set_rng_state(torch.as_tensor(runtime["torch_rng"], dtype=torch.uint8).cpu())
            if runtime["cuda_rng"] and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(
                    [torch.as_tensor(x, dtype=torch.uint8).cpu() for x in runtime["cuda_rng"]]
                )
            np.random.set_state(runtime["numpy_rng"])
            random.setstate(runtime["python_rng"])


class JointTorchLearner(PhasedTorchLearner):
    def compute_loss_for_module(self, **kwargs):
        raise NotImplementedError("Joint algorithms implement compute_losses")

    @property
    def actor_ids(self):
        return self.module[CRITIC_ID].actor_ids

    def unpack_joint(self, batch):
        validate_joint_rows(batch, self.actor_ids)
        actors = [self.module[mid] for mid in self.actor_ids]
        obs = [
            a.encoder.layout.torch_batch(batch[mid][Columns.OBS]) for mid, a in zip(self.actor_ids, actors)
        ]
        next_obs = [
            a.encoder.layout.torch_batch(batch[mid][Columns.NEXT_OBS])
            for mid, a in zip(self.actor_ids, actors)
        ]
        actions = [
            a.layout.encode(batch[mid][Columns.ACTIONS], self._device)
            for mid, a in zip(self.actor_ids, actors)
        ]
        rewards = torch.stack([batch[mid][Columns.REWARDS] for mid in self.actor_ids], -1)
        terms = torch.stack([batch[mid][Columns.TERMINATEDS] for mid in self.actor_ids], -1).float()
        return actors, obs, actions, rewards, next_obs, terms

    def log_loss(self, mid, name, loss):
        self.metrics.log_value((mid, name), loss.detach(), window=1)
        return loss
