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

    def update(self, *args, update_phase="joint", **kwargs):
        """Select gradient ownership without changing native minibatching/Adam.

        Unused parameters receive no gradient (not zero gradients), so Adam
        preserves their parameters, moments and step counters even after prior
        updates. The scope also resets on failure and is not checkpoint state.
        """
        if update_phase not in {"joint", "critic", "actor"}:
            raise ValueError("update_phase must be joint, critic or actor")
        previous = getattr(self, "_update_phase", "joint")
        self._update_phase = update_phase
        if update_phase == "critic":
            self._critic_gradient_norms = {}
        try:
            return super().update(*args, **kwargs)
        finally:
            self._update_phase = previous

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

    @torch.no_grad()
    def batch_diagnostics(self, batch, *, actor_ids=()):
        """Measure a full fixed batch without optimizer, metric or RNG updates.

        Policy loss is the clipped PPO surrogate (entropy excluded). Frozen
        actors return None, never a cached metric from a previous update.
        """
        converted = self._convert_batch_type(batch)
        results = {}
        for key, data in converted.policy_batches.items():
            module = self.module[key].unwrapped()
            predictions = module.compute_values(data)
            loss, rmse, _ = value_fit_statistics(
                predictions, data["value_targets"], module.value_scale
            )
            inputs = data[Columns.OBS] if module.role == "uc" else data["critic_obs"]
            diagnostics = {"tanh_saturation": module.value.saturation(inputs.float())}
            series = {"prediction": predictions, "target": data["value_targets"]}
            if "value_bootstrap" in data:
                series.update(
                    bootstrap=data["value_bootstrap"],
                    realized_return=data["value_targets"] - data["value_bootstrap"],
                    reward=data[Columns.REWARDS],
                )
            for name, values in series.items():
                diagnostics[name] = {
                    "mean": float(values.mean()), "std": float(values.std(unbiased=False)),
                    "min": float(values.min()), "max": float(values.max()),
                }
            policy_loss = None
            if key in actor_ids:
                policy_loss, _ = clipped_policy_loss(
                    module.evaluate_actions(data)[Columns.ACTION_LOGP],
                    data[Columns.ACTION_LOGP], data["advantages"], self.config.clip_param,
                )
                policy_loss = float(policy_loss)
            results[key] = {
                "critic_loss": float(loss), "critic_rmse": float(rmse),
                "policy_loss": policy_loss, "value_scale": module.value_scale,
                "value_diagnostics": diagnostics,
                "critic_gradient_norms": getattr(self, "_critic_gradient_norms", {}).get(key, []),
            }
        return results

    def compute_loss_for_module(self, *, module_id, config, batch, fwd_out):
        phase = getattr(self, "_update_phase", "joint")
        if phase != "actor":
            value_loss, value_rmse, value_explained = value_fit_statistics(
                fwd_out[Columns.VF_PREDS],
                batch["value_targets"].detach(),
                self.module[module_id].unwrapped().value_scale,
            )
            for name, value in {
                "value_loss": value_loss,
                "value_rmse": value_rmse,
                "value_explained_variance": value_explained,
            }.items():
                self.metrics.log_value((module_id, name), value.detach(), window=1)
            if phase == "critic":
                return config.vf_loss_coeff * value_loss
        advantages = batch["advantages"].detach()
        actor_loss, ratio = clipped_policy_loss(
            fwd_out[Columns.ACTION_LOGP], batch[Columns.ACTION_LOGP].detach(), advantages, config.clip_param
        )
        entropy = fwd_out["entropy"].mean()
        if config.entropy_coeff:
            entropy = self.module[module_id].unwrapped().policy_entropy(batch).mean()
        loss = actor_loss - config.entropy_coeff * entropy
        if phase == "joint":
            loss = loss + config.vf_loss_coeff * value_loss
        statistics = {
            "policy_loss": actor_loss,
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
                if self._update_phase == "critic":
                    key = self._optimizer_name_to_module[name]
                    self._critic_gradient_norms.setdefault(key, []).append(float(norm))
        return {key: value for key, value in gradients_dict.items() if value is not None}

    def set_state(self, state):
        # Candidate rollback must restore older snapshots, not reject them as
        # stale broadcasts under RLlib's monotonically increasing weight index.
        state = copy.deepcopy(state)
        sequence = state.get("weights_seq_no", 0)
        state["weights_seq_no"] = 0
        super().set_state(state)
        self._weights_seq_no = sequence
