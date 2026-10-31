"""v9 fixed-rule/random-feasible whole-episode validation; no networks or trainers."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "environments/sn_energy_market/src"))
sys.dont_write_bytecode = True

from validation_helpers import execute_period, json_safe


def main():
    from snmarket import EnergyMarketEngine, EnvironmentConfig, load_profiles

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consumers", type=int)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--mode", choices=("fixed", "random"), default="fixed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profiles", type=Path)
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("D:/Codex/snmarket-validation"))
    args = parser.parse_args()
    base = EnvironmentConfig.from_directory(args.config_dir) if args.config_dir else EnvironmentConfig()
    count = args.consumers if args.consumers is not None else base.system.num_consumers if args.config_dir else 3
    horizon = args.horizon if args.horizon is not None else base.system.horizon_steps
    config = replace(base, system=replace(base.system, num_consumers=count, horizon_steps=horizon)).validate()
    engine = EnergyMarketEngine(config, load_profiles(args.profiles, config) if args.profiles else None)
    engine.reset(seed=args.seed, options={"episode_id": f"smoke-{args.seed}"})
    output = args.output.resolve() / f"v9-smoke-{uuid4().hex[:12]}"
    output.mkdir(parents=True)
    rng = np.random.default_rng(args.seed)
    coverage, maximums, total_rewards = Counter(), {}, Counter()
    with (output / "records.jsonl").open("w", encoding="utf-8") as stream:
        for period in range(horizon):
            record = execute_period(engine, rng, args.seed + period, coverage, maximums,
                                    fixed=args.mode == "fixed")
            total_rewards.update(record.rewards)
            stream.write(json.dumps(json_safe(record), ensure_ascii=False, allow_nan=False) + "\n")
    from validate_v9 import fingerprint
    report = {"status": "passed", "mechanism_version": config.market.mechanism_version,
              "mode": args.mode, "seed": args.seed, "sources": fingerprint(),
              "profile_sha256": engine.profiles.fingerprint(), "config": config.to_dict(),
              "physical_periods": horizon, "engine_action_calls": 2 * horizon,
              "consumer_records": horizon * count, "coverage": dict(coverage),
              "maximums": maximums, "returns": dict(total_rewards), "no_training": True}
    path = output / "report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": "passed", "physical_periods": horizon, "maximums": maximums}), flush=True)
    print(f"验证记录：{path}", flush=True)


if __name__ == "__main__":
    main()
