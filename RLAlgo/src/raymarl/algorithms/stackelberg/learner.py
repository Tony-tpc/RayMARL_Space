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


def value_fit_statistics(predictions, targets, scale):
    """Fit in fixed value units, while reporting errors in reward units."""
    error = predictions - targets
    loss = 0.5 * (error / scale).square().mean()
    variance = targets.var(unbiased=False)
    explained = torch.where(variance > 0, 1.0 - error.var(unbiased=False) / variance.clamp_min(1e-12), 0.0)
    return loss, error.square().mean().sqrt(), explained


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
        value_loss, value_rmse, value_explained = value_fit_statistics(
            fwd_out[Columns.VF_PREDS],
            batch["value_targets"].detach(),
            self.module[module_id].unwrapped().value_scale,
        )
        entropy = fwd_out["entropy"].mean()
        if config.entropy_coeff:
            entropy = self.module[module_id].unwrapped().policy_entropy(batch).mean()
        loss = actor_loss + config.vf_loss_coeff * value_loss - config.entropy_coeff * entropy
        statistics = {
            "policy_loss": actor_loss,
            "value_loss": value_loss,
            "value_rmse": value_rmse,
            "value_explained_variance": value_explained,
            "entropy": entropy,
            "clip_fraction": ((ratio - 1).abs() > config.clip_param).float().mean(),
            "approx_kl": ((ratio - 1) - ratio.log()).mean(),
        }
        for key, value in statistics.items():
            self.metrics.log_value((module_id, key), value.detach(), window=1)
        return loss

    def fit_values(self, batches, *, epochs, minibatch_size):
        """Supervise only Value on fixed original-unit targets supplied by the caller.

        Each module batch contains ``obs``, ``critic_obs`` and ``value_targets``.
        The existing Value optimizer and checkpoint state are reused; Actor
        parameters and its optimizer are never included in this update.
        """
        if epochs < 1 or minibatch_size < 1:
            raise ValueError("Value fitting needs positive epochs and minibatch_size")
        results = {}
        for module_id, source in batches.items():
            module = self.module[module_id].unwrapped()
            batch = {
                key: torch.as_tensor(source[key], device=self._device, dtype=torch.float32).detach()
                for key in (Columns.OBS, "critic_obs", "value_targets")
            }
            count = len(batch["value_targets"])
            if count == 0:
                raise ValueError("Value fitting needs nonempty target batches")
            optimizer = self.get_optimizer(module_id, "value")
            parameters = list(module.value.parameters())
            with torch.no_grad():
                _, before, _ = value_fit_statistics(
                    module.compute_values(batch), batch["value_targets"], module.value_scale
                )
            steps = 0
            maximum_norm = 0.0
            for _ in range(epochs):
                order = torch.randperm(count, device=self._device)
                for indices in order.split(minibatch_size):
                    mini = {key: value[indices] for key, value in batch.items()}
                    optimizer.zero_grad(set_to_none=True)
                    loss, _, _ = value_fit_statistics(
                        module.compute_values(mini), mini["value_targets"], module.value_scale
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite Value loss; optimizer update rejected")
                    (self.config.vf_loss_coeff * loss).backward()
                    norm = torch.nn.utils.clip_grad_norm_(
                        parameters, self.config.grad_clip, error_if_nonfinite=True
                    )
                    maximum_norm = max(maximum_norm, float(norm))
                    optimizer.step()
                    steps += 1
            with torch.no_grad():
                _, after, explained = value_fit_statistics(
                    module.compute_values(batch), batch["value_targets"], module.value_scale
                )
            results[module_id] = {
                "samples": count,
                "epochs": epochs,
                "optimizer_steps": steps,
                "rmse_before": float(before),
                "rmse_after": float(after),
                "explained_variance_after": float(explained),
                "maximum_unclipped_grad_norm": maximum_norm,
            }
        self._weights_seq_no += 1
        return {
            "modules": results,
            "optimizer_steps": sum(item["optimizer_steps"] for item in results.values()),
        }

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
