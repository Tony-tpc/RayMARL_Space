"""Native RLlib PPO updates for selected independent policies."""

import copy

import numpy as np
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.torch.torch_learner import TorchLearner


def clipped_policy_loss(new_logp, old_logp, advantages, clip_param):
    ratio = (new_logp - old_logp).exp()
    surrogate = torch.minimum(ratio * advantages, ratio.clamp(1 - clip_param, 1 + clip_param) * advantages)
    return -surrogate.mean(), ratio


class StackelbergTorchLearner(TorchLearner):
    """Actor and value have separate optimizers; absent module batches stay frozen."""

    def configure_optimizers_for_module(self, module_id, config=None):
        module = self.module[module_id]
        for name, network, learning_rate in (
            ("actor", module.actor, config.actor_lr),
            ("value", module.value, config.critic_lr),
        ):
            parameters = list(network.parameters())
            self.register_optimizer(
                module_id=module_id,
                optimizer_name=name,
                optimizer=torch.optim.Adam(parameters, lr=learning_rate),
                params=parameters,
                lr_or_lr_schedule=learning_rate,
            )

    def _convert_batch_type(self, batch, **kwargs):
        converted = super()._convert_batch_type(batch, **kwargs)
        # RLlib normally casts float64 to float32. Preserve physical boundaries
        # and their behavior log probabilities across collection and updating.
        for module_id, original in batch.policy_batches.items():
            for key, value in original.items():
                if getattr(value, "dtype", None) in (np.dtype("float64"), torch.float64):
                    converted.policy_batches[module_id][key] = torch.as_tensor(value, device=self._device)
        return converted

    def compute_loss_for_module(self, *, module_id, config, batch, fwd_out):
        advantages = batch["advantages"].detach()
        actor_loss, ratio = clipped_policy_loss(
            fwd_out[Columns.ACTION_LOGP], batch[Columns.ACTION_LOGP].detach(), advantages, config.clip_param
        )
        value_loss = 0.5 * (fwd_out[Columns.VF_PREDS] - batch["value_targets"].detach()).square().mean()
        entropy = fwd_out["entropy"].mean()
        if config.entropy_coeff:
            entropy = self.module[module_id].unwrapped().policy_entropy(batch).mean()
        loss = actor_loss + config.vf_loss_coeff * value_loss - config.entropy_coeff * entropy
        statistics = {
            "policy_loss": actor_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "clip_fraction": ((ratio - 1).abs() > config.clip_param).float().mean(),
            "approx_kl": ((ratio - 1) - ratio.log()).mean(),
        }
        for key, value in statistics.items():
            self.metrics.log_value((module_id, key), value.detach(), window=1)
        return loss

    def compute_gradients(self, loss_per_module, **kwargs):
        if any(not torch.isfinite(loss).all() for loss in loss_per_module.values()):
            raise FloatingPointError("Non-finite PPO loss; optimizer update rejected")
        return super().compute_gradients(loss_per_module, **kwargs)

    def postprocess_gradients(self, gradients_dict):
        for name, optimizer in self._named_optimizers.items():
            parameters = [
                p for group in optimizer.param_groups for p in group["params"] if p.grad is not None
            ]
            if parameters:
                norm = torch.nn.utils.clip_grad_norm_(
                    parameters, self.config.grad_clip, error_if_nonfinite=True
                )
                self.metrics.log_value(
                    (self._optimizer_name_to_module[name], "grad_norm"), norm.detach(), window=1
                )
        return {key: value for key, value in gradients_dict.items() if value is not None}

    def set_state(self, state):
        # Candidate rollback must restore older snapshots, not reject them as
        # stale broadcasts under RLlib's monotonically increasing weight index.
        state = copy.deepcopy(state)
        sequence = state.get("weights_seq_no", 0)
        state["weights_seq_no"] = 0
        super().set_state(state)
        self._weights_seq_no = sequence
