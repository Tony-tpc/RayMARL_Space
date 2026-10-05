"""RLlib TorchLearner with one owner for every shared critic parameter."""

import copy
import math

import torch
from ray.rllib.core.learner.torch.torch_learner import TorchLearner


def soft_update(target, online, tau):
    if not math.isfinite(tau) or not 0 < tau <= 1:
        raise ValueError("tau must be finite and in (0,1]")
    with torch.no_grad():
        for t, s in zip(target.parameters(), online.parameters(), strict=True):
            t.lerp_(s, tau)


def soft_target(rewards, terminated, next_q, next_logp, gamma, temperature):
    return rewards + gamma * (1 - terminated) * (next_q - temperature * next_logp)


def score_function_loss(logp, q, baseline, temperature):
    return (logp * (temperature * logp - q + baseline).detach()).mean()


class JointTorchLearner(TorchLearner):
    """Local joint learner. Multi-GPU/DDP is deliberately rejected in v0.1."""

    def compute_loss_for_module(self, **kwargs):
        raise NotImplementedError("Joint losses require update_joint with aligned agent batches")

    def build(self):
        if self.config.num_learners != 0:
            raise ValueError("v0.1 joint learner supports num_learners=0")
        self._device = torch.device("cuda" if self.config.num_gpus_per_learner else "cpu")
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA learner requested but CUDA is unavailable")
        self._module = self._module_obj.to(self._device)
        self.target = copy.deepcopy(self._module).to(self._device)
        self._is_built = True
        self.steps = 0
        optimizer = torch.optim.RMSprop if self.config.variant in ("maac", "isac") else torch.optim.Adam
        self.actor_optimizers = [
            optimizer(a.parameters(), lr=self.config.actor_lr) for a in self.module.actors
        ]
        self.critic_parameters = list(self.module.critic.parameters())
        if self.module.critic2:
            self.critic_parameters += list(self.module.critic2.parameters())
        self.critic_optimizer = optimizer(self.critic_parameters, lr=self.config.critic_lr)

    def _step(self, loss, optimizer, parameters, clip):
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss; optimizer step rejected")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grads = [p.grad for p in parameters if p.grad is not None]
        if any(not torch.isfinite(g).all() for g in grads):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("Non-finite gradient; optimizer step rejected")
        norm = torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
        optimizer.step()
        return float(norm.detach())

    def update_joint(self, observations, actions, rewards, next_observations, terminated):
        cfg, module = self.config, self.module
        deterministic = cfg.variant == "maddpg"
        with torch.no_grad():
            target_actors = module.actors if cfg.variant == "masac" else self.target.actors
            draws = [
                a.distribution(o).draw(deterministic=deterministic)
                for a, o in zip(target_actors, next_observations)
            ]
            next_q = self.target.critic(next_observations, [d[0] for d in draws])
            if self.target.critic2:
                next_q = torch.minimum(next_q, self.target.critic2(next_observations, [d[0] for d in draws]))
            target = soft_target(
                rewards,
                terminated,
                next_q,
                torch.stack([d[1] for d in draws], -1),
                cfg.gamma,
                0.0 if deterministic else cfg.temperature,
            )
        q = module.critic(observations, actions)
        critic_loss = (q - target).square().mean(0).sum()
        if module.critic2:
            critic_loss = (
                critic_loss + (module.critic2(observations, actions) - target).square().mean(0).sum()
            )
        critic_norm = self._step(
            critic_loss, self.critic_optimizer, self.critic_parameters, cfg.critic_grad_clip
        )
        with torch.no_grad():
            other_actions = [
                a.distribution(o).draw(deterministic=deterministic)[0]
                for a, o in zip(module.actors, observations)
            ]
        metrics = {"critic_loss": float(critic_loss.detach()), "critic_grad_norm": critic_norm}
        for p in self.critic_parameters:
            p.requires_grad_(False)
        try:
            cached = module.critic.encode_observations(observations)
            cached2 = module.critic2.encode_observations(observations) if module.critic2 else None
            for i, (actor, obs, optimizer) in enumerate(
                zip(module.actors, observations, self.actor_optimizers)
            ):
                distribution = actor.distribution(obs)
                if deterministic:
                    # Standard MADDPG keeps the OTHER replay actions fixed.
                    joint = [a.detach() for a in actions]
                    joint[i] = distribution.draw(deterministic=True)[0]
                    loss = -module.critic(observations, joint, encoded=cached)[:, i].mean()
                elif cfg.variant == "masac":
                    loss = obs["current"].new_zeros(())
                    # Exact discrete expectation; continuous branches use rsample.
                    for choice in distribution.layout.discrete_choices():
                        action, lp, dlp = distribution.draw(reparameterize=True, discrete_choice=choice)
                        joint = list(other_actions)
                        joint[i] = action
                        q_i = module.critic(observations, joint, encoded=cached)[:, i]
                        q_i = torch.minimum(q_i, module.critic2(observations, joint, encoded=cached2)[:, i])
                        loss = loss + (dlp.exp() * (cfg.temperature * lp - q_i)).mean()
                else:
                    action, lp, _ = distribution.draw()
                    with torch.no_grad():
                        joint = list(other_actions)
                        joint[i] = action
                        q_i = module.critic(observations, joint, encoded=cached)[:, i]
                        baseline = torch.zeros_like(q_i)
                        if cfg.counterfactual:
                            from .distributions import HybridDistribution

                            k = cfg.baseline_samples
                            batch_size = distribution.logits.shape[0]
                            expanded = HybridDistribution(
                                distribution.logits.unsqueeze(0)
                                .expand(k, -1, -1)
                                .reshape(k * batch_size, -1),
                                action_space=actor.action_space,
                            )
                            expanded_actions = [a.repeat(k, 1) for a in other_actions]
                            expanded_features = [e.repeat(k, 1) for e in cached]
                            for choice in distribution.layout.discrete_choices():
                                # Vectorized independent samples; no reuse of the scored action.
                                candidate, _, dlp = expanded.draw(discrete_choice=choice)
                                expanded_actions[i] = candidate
                                values = module.critic(
                                    observations, expanded_actions, encoded=expanded_features
                                )[:, i]
                                baseline += (dlp.exp() * values).reshape(k, batch_size).mean(0)
                    loss = score_function_loss(lp, q_i, baseline, cfg.temperature)
                norm = self._step(loss, optimizer, list(actor.parameters()), cfg.actor_grad_clip)
                metrics[f"actor_{i}_loss"] = float(loss.detach())
                metrics[f"actor_{i}_grad_norm"] = norm
        finally:
            for p in self.critic_parameters:
                p.requires_grad_(True)
        soft_update(self.target.critic, module.critic, cfg.tau)
        if module.critic2:
            soft_update(self.target.critic2, module.critic2, cfg.tau)
        for target_actor, actor in zip(self.target.actors, module.actors):
            soft_update(target_actor, actor, cfg.tau)
        self.steps += 1
        return metrics

    def get_state(self, *args, **kwargs):
        return copy.deepcopy(
            {
                "signature": (
                    self.config.variant,
                    self.config.critic_mode,
                    self.config.hidden_dim,
                    self.config.temperature,
                    self.config.tau,
                    self.config.gamma,
                ),
                "module": self.module.get_state(),
                "target": self.target.get_state(),
                "actor_optimizers": [o.state_dict() for o in self.actor_optimizers],
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "steps": self.steps,
            }
        )

    def set_state(self, state):
        signature = (
            self.config.variant,
            self.config.critic_mode,
            self.config.hidden_dim,
            self.config.temperature,
            self.config.tau,
            self.config.gamma,
        )
        if state.get("signature") != signature:
            raise ValueError("Checkpoint algorithm/config signature mismatch")

        def finite(value):
            if (
                isinstance(value, torch.Tensor)
                and value.is_floating_point()
                and not torch.isfinite(value).all()
            ):
                raise ValueError("Non-finite checkpoint tensor")
            if isinstance(value, dict):
                for child in value.values():
                    finite(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    finite(child)

        finite(state)
        for field, module in (("module", self.module), ("target", self.target)):
            expected = module.get_state()

            def layout(actual, expected):
                if isinstance(expected, dict):
                    if actual.keys() != expected.keys():
                        raise ValueError("Checkpoint parameter keys mismatch")
                    for key in expected:
                        layout(actual[key], expected[key])
                elif isinstance(expected, torch.Tensor):
                    if actual.shape != expected.shape or actual.dtype != expected.dtype:
                        raise ValueError("Checkpoint parameter shape/dtype mismatch")

            layout(state[field], expected)
        if len(state["actor_optimizers"]) != len(self.actor_optimizers):
            raise ValueError("Checkpoint optimizer count mismatch")
        self.module.set_state(state["module"])
        self.target.set_state(state["target"])
        for optimizer, saved in zip(self.actor_optimizers, state["actor_optimizers"], strict=True):
            optimizer.load_state_dict(copy.deepcopy(saved))
        self.critic_optimizer.load_state_dict(copy.deepcopy(state["critic_optimizer"]))
        self.steps = state["steps"]
