"""Native RLlib PPO updates for selected independent policies."""

import copy

import numpy as np
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.torch.torch_learner import TorchLearner

from .mode_diagnostics import gradient_norms, mode_statistics, soc_advantages, summarize_minibatches
from .popart import update_popart
from .runner import generalized_advantages


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

    def update(self, *args, update_phase="joint", popart_targets=None, **kwargs):
        """Select gradient ownership without changing native minibatching/Adam.

        Unused parameters receive no gradient (not zero gradients), so Adam
        preserves their parameters, moments and step counters even after prior
        updates. The scope also resets on failure and is not checkpoint state.
        """
        if update_phase not in {"joint", "critic", "actor"}:
            raise ValueError("update_phase must be joint, critic or actor")
        previous = getattr(self, "_update_phase", "joint")
        self._update_phase = update_phase
        gradient_flags = []
        try:
            if update_phase in {"actor", "joint"} and getattr(self.config, "mode_credit", None) is not None:
                for module in self.module.values():
                    module = module.unwrapped()
                    if module.role == "uc":
                        for name, parameter in module.named_parameters():
                            if name.startswith(("actor.mode.", "actor.mode_soc.")):
                                gradient_flags.append((parameter, parameter.requires_grad))
                                parameter.requires_grad_(False)
            self._mode_diagnostics = {}
            self._pending_mode_diagnostics = {}
            if update_phase == "critic":
                self._critic_gradient_norms = {}
            if update_phase in {"critic", "joint"}:
                batch = kwargs.get("batch", args[0] if args else None)
                if batch is None:
                    batch = getattr(kwargs.get("training_data"), "batch", None)
                if batch is not None:
                    for key, data in batch.policy_batches.items():
                        value = self.module[key].unwrapped().value
                        if hasattr(value, "popart_scale"):
                            targets = (popart_targets or {}).get(key, data["value_targets"])
                            update_popart(value, targets)
            if update_phase == "actor" and getattr(self.config, "mode_credit", None) is None:
                batch = kwargs.get("batch", args[0] if args else None)
                # LearnerGroup wraps local batches in RLlib TrainingData before forwarding.
                if batch is None:
                    batch = getattr(kwargs.get("training_data"), "batch", None)
                if batch is not None:
                    for key, data in self._convert_batch_type(batch).policy_batches.items():
                        module = self.module[key].unwrapped()
                        if module.role == "uc":
                            self._mode_diagnostics[key] = {
                                "unique_samples": len(data["advantages"]),
                                "soc_advantages": soc_advantages(module, data), "minibatches": [],
                            }
            return super().update(*args, **kwargs)
        finally:
            for parameter, enabled in gradient_flags:
                parameter.requires_grad_(enabled)
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

    @torch.no_grad()
    def update_mode_head(self, module_id, batch, *, target_kl=0.01, damping=0.001):
        """One conditional-Q natural-gradient step on the linear mode head.

        Enumerating the three modes gives its exact empirical Fisher. As in
        TRPO, accept a step only when the sampled surrogate improves within
        the KL limit. Inputs and Q must come from this current joint policy.
        This stateless update leaves all Adam states and other weights intact;
        the ordinary actor phase remains a separate caller-owned operation.
        """
        if not (np.isfinite(target_kl) and target_kl > 0 and np.isfinite(damping) and damping > 0):
            raise ValueError("Mode KL limit and damping must be finite and positive")
        module = self.module[module_id].unwrapped()
        if module.role != "uc":
            raise ValueError("A mode-head update requires a UC module")
        data = {key: torch.as_tensor(value, device=self._device) for key, value in batch.items()}
        obs, base = data["obs"], data["base_latents"]
        features = module._features(obs)
        distribution, valid = module._uc_mode(
            features, base, data["bounds"].double(), module.centered_soc(obs),
        )
        p = distribution._dist.probs
        if not len(p) or data["mode_q_values"].shape != p.shape or data["mode_old_probs"].shape != p.shape:
            raise ValueError("Mode returns and probabilities require one row per state and three columns")
        q = data["mode_q_values"].double().masked_fill(~valid, 0.0)
        if not torch.equal(valid, data["mode_mask"]):
            raise ValueError("Counterfactual mode feasibility changed")
        if not torch.isfinite(q).all() or not torch.isfinite(data["mode_old_probs"]).all():
            raise ValueError("Mode probabilities and Q values must be finite")
        if not torch.allclose(p, data["mode_old_probs"].double(), atol=2e-7, rtol=0):
            # Probes run one state at a time on CPU. Float32 batching and CUDA
            # kernels can exceed this tolerance even with identical weights.
            # Replay the sampling path before declaring the policy stale;
            # keep the original tolerance and the learner's update arithmetic.
            reference = copy.deepcopy(module).cpu()
            cpu_obs, cpu_base = obs.cpu(), base.cpu()
            cpu_bounds = data["bounds"].double().cpu()
            replay = torch.cat([
                reference._uc_mode(
                    reference._features(cpu_obs[i:i+1]), cpu_base[i:i+1],
                    cpu_bounds[i:i+1], reference.centered_soc(cpu_obs[i:i+1]),
                )[0]._dist.probs for i in range(len(p))
            ])
            if not torch.allclose(replay, data["mode_old_probs"].double().cpu(), atol=2e-7, rtol=0):
                raise ValueError("Mode return batch is stale for the current policy")
        head = module.actor["mode"]
        parameters = [head.weight, head.bias]
        columns = [features, base.clamp(0, 1).float(), torch.ones_like(p[:, :1])]
        weights = [head.weight.double(), head.bias[:, None].double()]
        if module.soc_input is not None:
            parameters.append(module.actor["mode_soc"]["weight"])
            columns.append(module.centered_soc(obs)[:, None])
            weights.append(parameters[-1].double())
        x, w = torch.cat(columns, -1).double(), torch.cat(weights, -1)
        q = q - (p * q).sum(-1, keepdim=True)
        gradient = torch.einsum("nc,nd->cd", p * q, x) / len(x)
        covariance = torch.diag_embed(p) - p[:, :, None] * p[:, None, :]
        fisher = torch.einsum("nac,nb,nd->abcd", covariance, x, x).reshape(w.numel(), w.numel()) / len(x)
        direction = torch.linalg.solve(
            fisher + damping * torch.eye(w.numel(), dtype=fisher.dtype, device=fisher.device),
            gradient.flatten(),
        )
        curvature = float(direction @ fisher @ direction)
        result = {"accepted": False, "samples": len(x), "mode_q_gain": 0.0, "mode_kl": 0.0}
        if curvature <= 1e-20:
            return result
        step = (2 * target_kl / curvature) ** 0.5
        originals = [parameter.clone() for parameter in parameters]
        try:
            for backtrack in range(20):
                candidate = w + step * direction.reshape_as(w)
                head.weight.copy_(candidate[:, :head.in_features])
                head.bias.copy_(candidate[:, head.in_features])
                if module.soc_input is not None:
                    parameters[-1].copy_(candidate[:, -1:])
                current, _ = module._uc_mode(features, base, data["bounds"].double(), module.centered_soc(obs))
                new = current._dist.probs
                kl = float((p * (p.clamp_min(1e-100).log() - new.clamp_min(1e-100).log())).sum(-1).mean())
                gain = float(((new - p) * q).sum(-1).mean())
                if np.isfinite(kl) and np.isfinite(gain) and kl <= target_kl and gain > 0:
                    result.update(accepted=True, mode_q_gain=gain, mode_kl=kl, backtracks=backtrack)
                    self._weights_seq_no += 1
                    return result
                step *= 0.8
            return result
        finally:
            if not result["accepted"]:
                for parameter, original in zip(parameters, originals, strict=True):
                    parameter.copy_(original)

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
    def recompute_advantages(self, batch):
        """Refresh GAE once after value fitting, using the original next inputs.

        This only evaluates values: no action sampling, RNG use, optimizer steps,
        or changes to the fixed Critic targets and PPO behavior probabilities.
        """
        converted = self._convert_batch_type(batch)
        advantages = {}
        for key, data in converted.policy_batches.items():
            module = self.module[key].unwrapped()
            values = module.compute_values(data).cpu().numpy()
            inputs = data["next_value_obs"]
            next_values = module.compute_values({"obs": inputs, "critic_obs": inputs}).cpu().numpy()
            original = batch.policy_batches[key]
            advantages[key], _ = generalized_advantages(
                original["rewards"], values, next_values, original["terminateds"],
                self.config.gamma, self.config.lambda_gae,
                original.get("segment_end"),
            )
        return advantages

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
            inputs = data["critic_obs"]
            diagnostics = {"tanh_saturation": module.value.saturation(inputs.float())}
            if hasattr(module.value, "popart_scale"):
                diagnostics["popart"] = {
                    name: float(getattr(module.value, "popart_" + name))
                    for name in ("shift", "scale", "mass", "updates")
                }
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
                "mode_diagnostics": copy.deepcopy(getattr(self, "_mode_diagnostics", {}).get(key))
                    if key in actor_ids else None,
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
        if phase == "actor" and module_id in self._mode_diagnostics:
            diagnostics = self._mode_diagnostics[module_id]
            entry = mode_statistics(self.module[module_id].unwrapped(), batch, fwd_out, config.clip_param)
            visits = sum(row["samples"] for row in diagnostics["minibatches"])
            entry.update(minibatch=len(diagnostics["minibatches"]) + 1,
                         epoch=visits // diagnostics["unique_samples"] + 1,
                         entropy_coeff=config.entropy_coeff)
            diagnostics["minibatches"].append(entry)
            self._pending_mode_diagnostics[module_id] = entry
        logp = fwd_out[Columns.ACTION_LOGP]
        if getattr(self.config, "mode_credit", None) is not None and self.module[module_id].unwrapped().role == "uc":
            parts = fwd_out["logp_parts"]
            # Preserve the joint importance ratio, but remove the unreliable
            # mode GAE gradient, including its path through the shared encoder.
            logp = parts[:, 0] + parts[:, 1].detach() + parts[:, 2]
        actor_loss, ratio = clipped_policy_loss(logp, batch[Columns.ACTION_LOGP].detach(), advantages, config.clip_param)
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
                key = self._optimizer_name_to_module[name]
                module = self.module[key].unwrapped()
                entry = getattr(self, "_pending_mode_diagnostics", {}).get(key)
                mode_update = entry is not None and any(p is module.actor["mode"].weight for p in parameters)
                if mode_update:
                    entry["gradient_norm_before"] = gradient_norms(module)
                norm = torch.nn.utils.clip_grad_norm_(
                    parameters, self.config.grad_clip, error_if_nonfinite=True
                )
                if mode_update:
                    entry["gradient_norm_after"] = gradient_norms(module)
                    entry["actor_global_norm"] = float(norm)
                    entry["global_clip_multiplier"] = min(1.0, self.config.grad_clip / (float(norm) + 1e-6))
                self.metrics.log_value(
                    (self._optimizer_name_to_module[name], "grad_norm"), norm.detach(), window=1
                )
                if self._update_phase == "critic":
                    key = self._optimizer_name_to_module[name]
                    self._critic_gradient_norms.setdefault(key, []).append(float(norm))
        return {key: value for key, value in gradients_dict.items() if value is not None}

    def apply_gradients(self, gradients_dict):
        # Observe native Adam; diagnostics never call backward/step or enter checkpoint state.
        before = {
            key: {name: p.detach().clone() for name, p in self.module[key].actor.named_parameters()
                  if name.startswith(("mode.", "mode_soc."))}
            for key in getattr(self, "_pending_mode_diagnostics", {})
        }
        optimizers = {
            key: next(opt for opt in self._named_optimizers.values() if any(
                p is self.module[key].actor["mode"].weight
                for group in opt.param_groups for p in group["params"]
            )) for key in before
        }
        for key, optimizer in optimizers.items():
            state = optimizer.state.get(self.module[key].actor["mode"].weight, {})
            self._pending_mode_diagnostics[key]["adam_step_before"] = int(state.get("step", 0))
        super().apply_gradients(gradients_dict)
        with torch.no_grad():
            for key, saved in before.items():
                current = dict(self.module[key].actor.named_parameters())
                entry = self._pending_mode_diagnostics[key]
                entry["adam_step_after"] = int(optimizers[key].state[current["mode.weight"]]["step"])
                delta = {name: current[name] - value for name, value in saved.items()}
                entry["parameter_delta_norm"] = {name: float(value.double().norm()) for name, value in delta.items()}
                entry["mode_bias_delta"] = delta["mode.bias"].tolist()
                if "mode_soc.weight" in current:
                    entry["soc_weight_delta"] = delta["mode_soc.weight"].flatten().tolist()
                    entry["soc_weights_after"] = current["mode_soc.weight"].flatten().tolist()
                diagnostics = self._mode_diagnostics[key]
                diagnostics["summary"] = summarize_minibatches(diagnostics["minibatches"])
        self._pending_mode_diagnostics.clear()

    def set_state(self, state):
        # Candidate rollback must restore older snapshots, not reject them as
        # stale broadcasts under RLlib's monotonically increasing weight index.
        state = copy.deepcopy(state)
        sequence = state.get("weights_seq_no", 0)
        state["weights_seq_no"] = 0
        super().set_state(state)
        self._weights_seq_no = sequence
