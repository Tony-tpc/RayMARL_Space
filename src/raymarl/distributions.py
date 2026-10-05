"""Factorized policies with stable bounded-Gaussian log densities."""

import math

import torch
from ray.rllib.core.distribution.torch.torch_distribution import TorchDistribution
from torch.distributions import Bernoulli, Categorical, Normal, kl_divergence

from .spaces import ActionLayout


class HybridDistribution(TorchDistribution):
    def __init__(self, logits, *, action_space, action_latents=None):
        self.logits = logits
        self.layout = ActionLayout(action_space)
        self.action_latents = action_latents
        self.parts, offset = [], 0
        for b in self.layout.branches:
            if b.kind == "box":
                mean, logstd = logits[..., offset : offset + 2 * b.size].chunk(2, -1)
                d = Normal(mean, logstd.clamp(-20, 2).exp(), validate_args=False)
                offset += 2 * b.size
            else:
                p = logits[..., offset : offset + b.size]
                d = (
                    Bernoulli(logits=p, validate_args=False)
                    if b.kind == "binary"
                    else Categorical(logits=p, validate_args=False)
                )
                offset += b.size
            self.parts.append(d)

    def _get_torch_distribution(self, *args, **kwargs):
        return None

    @staticmethod
    def required_input_dim(space, **kwargs):
        return ActionLayout(space).param_dim

    def draw(self, *, reparameterize=False, deterministic=False, discrete_choice=None, return_latents=False):
        features, lp, discrete_lp = [], self.logits.new_zeros(self.logits.shape[:-1]), []
        choices = iter(discrete_choice) if discrete_choice is not None else None
        latents = []
        for b, d in zip(self.layout.branches, self.parts):
            if b.kind == "box":
                u = d.loc if deterministic else (d.rsample() if reparameterize else d.sample())
                latents.append(u)
                a = torch.tanh(u)
                log_jac = 2 * (math.log(2) - u - torch.nn.functional.softplus(-2 * u))
                scale = u.new_tensor((b.high - b.low) / 2)
                lp = lp + (d.log_prob(u) - log_jac - scale.log()).sum(-1)
            elif b.kind == "binary":
                if choices is not None:
                    a = self.logits.new_tensor(next(choices)).expand_as(d.logits)
                else:
                    a = (d.logits >= 0).float() if deterministic else d.sample()
                p = d.log_prob(a).sum(-1)
                discrete_lp.append(p)
                lp = lp + p
            else:
                if choices is not None:
                    index = torch.full_like(d.logits[..., 0], next(choices), dtype=torch.long)
                else:
                    index = d.logits.argmax(-1) if deterministic else d.sample()
                a = torch.nn.functional.one_hot(index, b.size).float()
                p = d.log_prob(index)
                discrete_lp.append(p)
                lp = lp + p
            features.append(a)
        disc_lp = sum(discrete_lp, self.logits.new_zeros(self.logits.shape[:-1]))
        result = (torch.cat(features, -1), lp, disc_lp)
        return (
            (
                *result,
                torch.cat(latents, -1) if latents else self.logits.new_empty((*self.logits.shape[:-1], 0)),
            )
            if return_latents
            else result
        )

    def sample(self, *, sample_shape=None):
        if sample_shape:
            raise NotImplementedError("Use batched logits, not sample_shape")
        x, lp, _ = self.draw()
        result = self.layout.decode_features(x)
        self._last_sample = result
        self._last_logp = lp
        return result

    def rsample(self, *, sample_shape=None):
        return self.layout.decode_features(self.draw(reparameterize=True)[0])

    def logp(self, value, **kwargs):
        if value is getattr(self, "_last_sample", None):
            return self._last_logp
        lp = self.logits.new_zeros(self.logits.shape[:-1])
        latent_offset = 0
        for b, d, action_value in zip(self.layout.branches, self.parts, self.layout.values(value)):
            v = torch.as_tensor(action_value, device=self.logits.device)
            if b.kind == "box":
                v = v.reshape(*self.logits.shape[:-1], b.size)
                lo, hi = v.new_tensor(b.low), v.new_tensor(b.high)
                scale = (hi - lo) / 2
                a = ((v - lo) / scale - 1).clamp(-1 + 1e-6, 1 - 1e-6)
                u = (
                    self.action_latents[..., latent_offset : latent_offset + b.size]
                    if self.action_latents is not None
                    else torch.atanh(a)
                )
                latent_offset += b.size
                jac = 2 * (math.log(2) - u - torch.nn.functional.softplus(-2 * u))
                lp = lp + (d.log_prob(u) - jac - scale.log()).sum(-1)
            elif b.kind == "binary":
                lp = lp + d.log_prob(v.reshape(*self.logits.shape[:-1], b.size).float()).sum(-1)
            else:
                lp = lp + d.log_prob(v.long() - b.start)
        return lp

    def entropy(self):
        total = self.logits.new_zeros(self.logits.shape[:-1])
        for b, d in zip(self.layout.branches, self.parts):
            if b.kind == "box":
                u = d.rsample()
                jac = 2 * (math.log(2) - u - torch.nn.functional.softplus(-2 * u))
                total = total + (d.entropy() + jac + u.new_tensor((b.high - b.low) / 2).log()).sum(-1)
            elif b.kind == "binary":
                total = total + d.entropy().sum(-1)
            else:
                total = total + d.entropy()
        return total

    def kl(self, other):
        return sum(
            (kl_divergence(a, b).sum(-1) if branch.kind != "categorical" else kl_divergence(a, b))
            for branch, a, b in zip(self.layout.branches, self.parts, other.parts)
        )

    def to_deterministic(self):
        return DeterministicHybrid(self)

    @classmethod
    def from_logits(cls, logits, **kwargs):
        return cls(logits, **kwargs)

    @classmethod
    def get_partial_dist_cls(parent, **fixed):
        class BoundDistribution(parent):
            @classmethod
            def from_logits(cls, logits, **kwargs):
                if set(kwargs) & set(fixed):
                    raise ValueError("Cannot override bound distribution arguments")
                return cls(logits, **fixed, **kwargs)

            @staticmethod
            def required_input_dim(space, **kwargs):
                return ActionLayout(space).param_dim

        return BoundDistribution


class DeterministicHybrid:
    def __init__(self, parent):
        self.parent = parent

    def sample(self, **kwargs):
        return self.parent.layout.decode_features(self.parent.draw(deterministic=True)[0])

    def logp(self, value):
        return self.parent.logp(value)

    def to_deterministic(self):
        return self


def bound_distribution(space):
    return HybridDistribution.get_partial_dist_cls(action_space=space)
