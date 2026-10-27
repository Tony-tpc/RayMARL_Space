"""Online value normalization; output compensation follows DeepMind RLax PopArt.

https://github.com/google-deepmind/rlax/blob/main/rlax/_src/pop_art.py
One debiased EMA update per phase batch, before the existing Critic epochs.
"""

import torch

POPART = {
    "beta_per_phase": 0.99999,
    "minimum_std": 0.1,
    "statistics": "debiased_EMA_once_per_phase",
    "optimizer": "preserve_Adam_moments_during_output_compensation",
    "values": "original_reward_units_for_GAE_bootstrap_and_evaluation",
}


def register_popart(value, initial_scale):
    for name, number in (("shift", 0.0), ("scale", initial_scale)):
        value.register_buffer("popart_" + name, torch.tensor(number, dtype=torch.float32))
    for name in ("mean_ema", "second_ema", "mass"):
        value.register_buffer("popart_" + name, torch.tensor(0.0, dtype=torch.float64))
    value.register_buffer("popart_updates", torch.tensor(0, dtype=torch.int64))


@torch.no_grad()
def update_popart(value, targets):
    target = torch.as_tensor(targets, device=value.popart_scale.device, dtype=torch.float64)
    if target.numel() == 0 or not torch.isfinite(target).all():
        raise ValueError("PopArt requires a nonempty finite phase target batch")
    beta = POPART["beta_per_phase"]
    value.popart_mean_ema.mul_(beta).add_(target.mean(), alpha=1 - beta)
    value.popart_second_ema.mul_(beta).add_(target.square().mean(), alpha=1 - beta)
    value.popart_mass.mul_(beta).add_(1 - beta)
    mean = value.popart_mean_ema / value.popart_mass
    variance = value.popart_second_ema / value.popart_mass - mean.square()
    shift = mean.to(value.popart_shift)
    scale = variance.clamp_min(POPART["minimum_std"] ** 2).sqrt().to(value.popart_scale)
    value.head.weight.mul_(value.popart_scale / scale)
    value.head.bias.mul_(value.popart_scale).add_(value.popart_shift - shift).div_(scale)
    value.popart_shift.copy_(shift)
    value.popart_scale.copy_(scale)
    value.popart_updates.add_(1)
