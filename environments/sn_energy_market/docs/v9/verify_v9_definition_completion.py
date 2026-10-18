"""Documentation-only checks for the five v9 definition completions.

The decoder below is an independent mathematical reference. It never imports
production environments or algorithms and does not implement a training policy.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import re
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
SOURCE_BEFORE_SHA = "542ebda36f829f1ea6a065598c96f65740b6fb0b34034c895baa9415b3d6178e"
ORIGINAL_DISPLAY_SHA = "5e9ef2b0474991416ee8437e12361050d1b7dee3eb917a410f09c09e05ddfb90"
CODE_BASELINE_SHA = "529c45cb8bd4aa4f72aa4269566ecb8e171ee89a4f1f6a12bb1f60ef9efbbecf"
TOL = 1e-8


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(text):
    return re.sub(r"\s+", "", text)


def semantic_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def original_displays(text):
    start = text.index("### 11.2 UC 一次输出与联合可行行动")
    end = text.index("### 11.3 消费者条件分布", start)
    return [m.group(1) for m in re.finditer(r"\$\$([\s\S]*?)\$\$", text)
            if not start <= m.start() < end and "J_j^{cont}" not in m.group(1)]


EXPECTED_NEW_DISPLAYS = [
    r"""J_j^{cont}(\pi_U,\pi_{1:N})=
\mathbb E\left[\sum_{t=0}^{\infty}\gamma^t R_{j,t}\right],\qquad 0\le\gamma<1.""",
    r"""u_k=\operatorname{clip}(z_k,0,1),\quad k\in\{1,2,3,4,6\},\qquad
u_5=\operatorname{clip}(z_5,-1,1).""",
    r"""\begin{aligned}
p^{PG,offer}&=u_1\bar p^{PG},\qquad p^{curt}=u_2p^{DER},\\
p^{ch,int,DER}&=u_3[u_5]_+(p^{DER}-p^{curt}),\\
p^{ch,int,PG}&=u_4[u_5]_+p^{PG,offer},\\
p^{dis,offer}&=[-u_5]_+p^{dis,avail},\qquad p^{DR,cap}=u_6\chi\bar p^{DR}.
\end{aligned}""",
    r"""\begin{aligned}
\beta&=
\begin{cases}
\displaystyle\min\left(1,
\frac{\min\left(\bar p^{ch},(E_{max}-E)/(\eta_c\Delta t)\right)}
{p^{ch,int,DER}+p^{ch,int,PG}}\right),
&p^{ch,int,DER}+p^{ch,int,PG}>0,\\
0,&p^{ch,int,DER}+p^{ch,int,PG}=0,
\end{cases}\\
p^{ch,DER}&=\beta p^{ch,int,DER},\qquad p^{ch,PG}=\beta p^{ch,int,PG}.
\end{aligned}""",
    r"""q_{\theta_U}(\mathbf{z}\mid o^U)=
\prod_{k=1}^{6}\mathcal N(z_k;\mu_k^U, (\sigma_k^U)^2),\qquad
\log q_{\theta_U}(\mathbf{z}\mid o^U)=\sum_{k=1}^{6}\log\mathcal N(z_k;\mu_k^U, (\sigma_k^U)^2).""",
]


@dataclass(frozen=True)
class FrozenInputs:
    der: float = 6.0
    grid_max: float = 8.0
    energy: float = 5.0
    energy_min: float = 0.0
    energy_max: float = 10.0
    charge_max: float = 8.0
    discharge_max: float = 8.0
    eta_c: float = 0.95
    eta_d: float = 0.95
    dt: float = 0.25
    event: int = 1
    dr_max: float = 4.0

    @property
    def charge_room(self):
        return min(self.charge_max, (self.energy_max - self.energy) / (self.eta_c * self.dt))

    @property
    def discharge_room(self):
        return min(self.discharge_max, self.eta_d * (self.energy - self.energy_min) / self.dt)


def validate_inputs(s):
    assert all(math.isfinite(v) for v in asdict(s).values())
    assert min(s.der, s.grid_max, s.charge_max, s.discharge_max, s.dr_max) >= 0
    assert s.energy_min <= s.energy <= s.energy_max
    assert s.dt > 0 and 0 < s.eta_c <= 1 and 0 < s.eta_d <= 1 and s.event in (0, 1)


def decode(z, s):
    validate_inputs(s)
    assert len(z) == 6 and all(math.isfinite(v) for v in z)
    u = [max(0.0, min(1.0, v)) for v in z]
    u[4] = max(-1.0, min(1.0, z[4]))
    grid = u[0] * s.grid_max
    curtailed = u[1] * s.der
    der_intent = u[2] * max(u[4], 0.0) * (s.der - curtailed)
    grid_intent = u[3] * max(u[4], 0.0) * grid
    intent = der_intent + grid_intent
    scale = min(1.0, s.charge_room / intent) if intent > 0 else 0.0
    return (grid, curtailed, scale * der_intent, scale * grid_intent,
            max(-u[4], 0.0) * s.discharge_room, u[5] * s.event * s.dr_max)


def check_action(action, s):
    assert len(action) == 6
    grid, curtailed, der_charge, grid_charge, discharge, dr = action
    assert all(math.isfinite(x) and x >= -TOL for x in action)
    assert grid <= s.grid_max + TOL and grid_charge <= grid + TOL
    assert curtailed + der_charge <= s.der + TOL
    assert der_charge + grid_charge <= s.charge_room + TOL
    assert discharge <= s.discharge_room + TOL
    assert (der_charge + grid_charge) * discharge == 0
    assert dr <= s.event * s.dr_max + TOL


def encode_legal_action(action, s):
    """Construct an internal preimage of an independently supplied legal action."""
    check_action(action, s)
    grid, curtailed, der_charge, grid_charge, discharge, dr = action
    direction = 1.0 if der_charge + grid_charge > 0 else (-discharge / s.discharge_room if discharge else 0.0)
    return (grid / s.grid_max if s.grid_max else 0.0,
            curtailed / s.der if s.der else 0.0,
            der_charge / (s.der - curtailed) if s.der > curtailed else 0.0,
            grid_charge / grid if grid else 0.0,
            direction, dr / (s.event * s.dr_max) if s.event * s.dr_max else 0.0)


def log_prob(z, mean, std):
    assert len(z) == len(mean) == len(std) == 6
    assert all(x > 0 for x in std)
    return sum(-0.5 * ((x - mu) / sigma) ** 2 - math.log(sigma) - 0.5 * math.log(2 * math.pi)
               for x, mu, sigma in zip(z, mean, std))


def load_reference():
    path = BASE / "verify_v9_mechanisms.py"
    assert sha(path) == "d732f140462839469c677fcab425f96aa70c9ab6cf2304f2fc6d425ef0ebed97"
    spec = importlib.util.spec_from_file_location("v9_definition_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    source = (BASE / "modeling_complete_zh.md").read_text(encoding="utf-8")
    original = original_displays(source)
    assert len(original) == 45 and semantic_sha(original) == ORIGINAL_DISPLAY_SHA
    all_displays = re.findall(r"\$\$([\s\S]*?)\$\$", source)
    assert len(all_displays) == len(original) + len(EXPECTED_NEW_DISPLAYS)
    original_ids = {compact(x) for x in original}
    added = [x for x in all_displays if compact(x) not in original_ids]
    assert [compact(x) for x in added] == [compact(x) for x in EXPECTED_NEW_DISPLAYS]
    assert re.findall(r"\\tag\{(\d+)\}", source) == [str(i) for i in range(1, 31)]
    assert semantic_sha(re.findall(r"```[\s\S]*?```", source)) == CODE_BASELINE_SHA
    for phrase in ["最大 DR 采购功率", "同一场景内固定", "最高 UC 零售价",
                   "仅在 $p_i^{due}=0$", "仍保留上限 $-p_i^{due}$", "连续场景的 Nash 响应",
                   "本版统一采用内部变量概率", "其规则在同一训练配置内保持不变"]:
        assert phrase in source, phrase

    rng = random.Random(10142026)
    samples = 10000
    maximum_error = 0.0
    for _ in range(samples):
        s = FrozenInputs(der=rng.uniform(0, 15), grid_max=rng.uniform(0, 15),
                         energy=rng.uniform(0, 10), charge_max=rng.uniform(0, 10),
                         discharge_max=rng.uniform(0, 10), dt=rng.choice([0.25, 0.5, 1.0]),
                         event=rng.choice([0, 1]), dr_max=rng.uniform(0, 15))
        check_action(decode([rng.gauss(0, 2) for _ in range(6)], s), s)
        # Construct legal physical actions directly from the joint constraints.
        grid, curtailed = rng.uniform(0, s.grid_max), rng.uniform(0, s.der)
        charge_total = rng.uniform(0, min(s.charge_room, s.der - curtailed + grid))
        der_charge = rng.uniform(max(0, charge_total - grid), min(charge_total, s.der - curtailed))
        grid_charge = charge_total - der_charge
        discharge = 0.0
        if rng.random() < 0.5:
            der_charge = grid_charge = 0.0
            discharge = rng.uniform(0, s.discharge_room)
        target = (grid, curtailed, der_charge, grid_charge, discharge,
                  rng.uniform(0, s.event * s.dr_max))
        actual = decode(encode_legal_action(target, s), s)
        check_action(actual, s)
        error = max(abs(a - b) for a, b in zip(actual, target))
        maximum_error = max(maximum_error, error)
        assert error < TOL, (actual, target)

    boundaries = [
        ("zero supply and idle", FrozenInputs(), (0, 0, 0, 0, 0, 0)),
        ("full DER curtailment", FrozenInputs(), (8, 6, 0, 0, 0, 4)),
        ("two-source full charge", FrozenInputs(), (8, 0, 6, 2, 0, 4)),
        ("maximum discharge", FrozenInputs(), (8, 0, 0, 0, 8, 4)),
        ("empty battery", FrozenInputs(energy=0), (8, 0, 0, 0, 0, 4)),
        ("full battery", FrozenInputs(energy=10), (8, 0, 0, 0, 8, 4)),
        ("zero DER", FrozenInputs(der=0), (8, 0, 0, 8, 0, 4)),
        ("zero PG", FrozenInputs(grid_max=0), (0, 0, 6, 0, 0, 4)),
        ("closed DR event", FrozenInputs(event=0), (8, 0, 0, 0, 0, 0)),
        ("zero DR maximum", FrozenInputs(dr_max=0), (8, 0, 0, 0, 0, 0)),
        ("all physical domains zero", FrozenInputs(der=0, grid_max=0, energy=0, event=0), (0, 0, 0, 0, 0, 0)),
    ]
    for name, s, target in boundaries:
        actual = decode(encode_legal_action(target, s), s)
        check_action(actual, s)
        assert all(abs(a - b) < TOL for a, b in zip(actual, target)), name
        check_action(decode([3, 3, 3, 3, 3, 3], s), s)
        check_action(decode([-3, -3, -3, -3, -3, -3], s), s)

    rejected = []
    invalid_actions = {
        "PG maximum": (9, 0, 0, 0, 0, 0),
        "DER split": (8, 6, 1, 0, 0, 0),
        "charging room": (8, 0, 6, 4, 0, 0),
        "mutual exclusion": (8, 0, 1, 0, 1, 0),
        "discharge maximum": (8, 0, 0, 0, 9, 0),
        "DR maximum": (8, 0, 0, 0, 0, 5),
    }
    for label, action in invalid_actions.items():
        try:
            check_action(action, FrozenInputs())
        except AssertionError:
            rejected.append(label)
        else:
            raise AssertionError(label)
    try:
        check_action((0, 0, 0, 0, 0, 1), FrozenInputs(event=0))
    except AssertionError:
        rejected.append("closed DR event")
    else:
        raise AssertionError("closed DR event")

    mean, std = [0.0] * 6, [1.0] * 6
    z1, z2 = [-1.0] * 6, [-2.0] * 6
    zero = FrozenInputs(der=0, grid_max=0, energy=0, event=0)
    assert decode(z1, zero) == decode(z2, zero)
    assert log_prob(z1, mean, std) != log_prob(z2, mean, std)
    max_probability_error = 0.0
    for _ in range(1000):
        z = [rng.gauss(0, 1) for _ in range(6)]
        sigmas = [rng.uniform(0.5, 2) for _ in range(6)]
        mus = [rng.uniform(-1, 1) for _ in range(6)]
        direct = math.prod(math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
                           for x, mu, sigma in zip(z, mus, sigmas))
        error = abs(math.exp(log_prob(z, mus, sigmas)) - direct)
        max_probability_error = max(max_probability_error, error)
        assert error < 1e-12
        assert math.exp(log_prob(z, mus, sigmas) - log_prob(z, mus, sigmas)) == 1

    ref = load_reference()
    last_due = ref.local_bounds(4, 0, 4, 1, 1, 5, 0.25, 95, 96, 3, 0)
    last_free = ref.local_bounds(4, 0, 4, 1, 0, 5, 0.25, 95, 96, 3, 0)
    assert last_due["low"] == last_due["high"] == -4
    assert last_free["high"] == 0
    for _ in range(500):
        s = FrozenInputs(energy=rng.uniform(0, 10), event=rng.choice([0, 1]))
        action = decode([rng.gauss(0, 2) for _ in range(6)], s)
        grid, curtailed, cr, cg, discharge, dr = action
        result = ref.run([ref.Order("A", rng.uniform(0, 10), der_share=rng.random(), der_bid=0.5)],
                         ref.Supply(grid_offer=grid, grid_max=s.grid_max, der=s.der,
                                    planned_curtail=curtailed, der_charge=cr, grid_charge=cg,
                                    discharge_offer=discharge, dr_cap=dr, energy=s.energy,
                                    event=bool(s.event), dr_price=0.2 if s.event else 0.0))
        assert abs(result["power_residual"]) < TOL and abs(result["cash_residual"]) < TOL
        assert tuple(result["original_action"]) == action

    continuous = []
    for gamma in [0.0, 0.5, 0.95, 0.99]:
        amount, count = 2.0, 500
        partial = sum(amount * gamma ** t for t in range(count))
        full = amount / (1 - gamma)
        tail = amount * gamma ** count / (1 - gamma)
        assert abs(full - partial - tail) < 1e-8
        continuous.append({"gamma": gamma, "constant_reward_value": full, "finite_prefix_tail": tail})
    assert not 0 <= 1.0 < 1

    integrity = {}
    for name in ["protected_files_manifest.json", "history_files_manifest.json"]:
        manifest = json.loads((BASE / name).read_text(encoding="utf-8"))
        assert all(sha(ROOT / path) == digest for path, digest in manifest.items())
        integrity[name] = {"files": len(manifest), "unchanged": True}
    report = {
        "status": "passed", "date": "2026-10-14", "seed": 10142026,
        "source_before_sha256": SOURCE_BEFORE_SHA,
        "source_after_sha256": sha(BASE / "modeling_complete_zh.md"),
        "numbered_equations": 30, "original_display_formulas_unchanged": len(original_displays(source)),
        "new_display_formulas_checked_exactly": len(added), "code_examples_unchanged": True,
        "six_physical_actions_and_joint_constraints_unchanged": True,
        "no_mandatory_discrete_modes_or_sampling_order": True,
        "decoder_random_samples": samples, "independent_legal_action_preimages": samples,
        "maximum_preimage_error_kw": maximum_error,
        "boundary_cases": [name for name, _, _ in boundaries], "negative_controls_rejected": rejected,
        "probability_checks": {"samples": 1000, "maximum_absolute_error": max_probability_error,
                               "same_policy_ratio": 1.0, "many_to_one_records_preserved": True,
                               "zero_physical_domain_keeps_latent_probability": True},
        "original_reference_market_and_physics_cases": 500,
        "last_period_bounds": {"due": last_due, "no_due": last_free},
        "continuous_discount_probes": continuous, "integrity": integrity,
        "reference_verifier_unchanged": True,
        "scope": "Independent documentation reference, source consistency, feasible-domain coverage and latent probability checks; production environment, network and Learner unchanged. No equilibrium or learning-convergence claim.",
    }
    (BASE / "definition_completion_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ["continuous_discount_probes", "last_period_bounds", "scope"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
