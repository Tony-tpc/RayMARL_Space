"""Ten-round Ausgrid storage experiment, passive traces and automatic reporting."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "RLAlgo/src"), str(ROOT / "environments/sn_energy_market/src")]

from experiments.sn_energy_market.batch_critic_pilot import (
    REFERENCE,
    configuration,
    round_budget,
    run_pilot,
)
from experiments.sn_energy_market.train import write_json

OUTPUT = ROOT / "outputs/ausgrid-hybrid-storage/20261022-long-horizon-10rounds"
OLD = ROOT / "outputs/ausgrid-year-study/20261020"
ROUNDS = 10


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def check_protocol(output):
    """Check data/calibration and record the explicitly increased sample budget."""
    old = json.loads((OLD / "report.json").read_text(encoding="utf-8"))["learning_contract"]
    cfg = configuration(output)
    new = cfg.learning_contract()
    budget = ("actor_lr", "critic_lr", "clip_param", "critic_epochs", "critic_lambda",
              "inner_updates", "minibatch_size", "num_epochs",
              "response_updates", "uc_candidates", "uc_updates_per_candidate",
              "seed", "train_seeds", "validation_seeds", "test_seeds", "deviation_updates")
    for key in budget:
        assert new[key] == old[key], (key, new[key], old[key])
    for key in ("manifest_sha256", "household_ids", "seed_windows", "resource_calibration", "loaded_splits"):
        assert new["scenario"][key] == old["scenario"][key], key
    old_config = old["scenario"]["config"]
    new_config = new["scenario"]["config"]
    assert old_config["system"] == new_config["system"]
    assert old_config["market"] == new_config["market"]
    for key, value in old_config["rewards"].items():
        if key != "terminal_energy_value_per_kwh":
            assert new_config["rewards"][key] == value
    assert new["gamma"] == 0.99 and new["evaluation_steps"] is None
    assert new["lambda_gae"] == 0.97 and new["rollout_length"] == 192
    assert new_config["rewards"]["terminal_energy_value_per_kwh"] > 0
    return {
        "outer_rounds": ROUNDS,
        **{key: value * ROUNDS for key, value in round_budget(cfg).items()},
        "seed": 42, "validation_rounds": [0, 3, ROUNDS], "test_loaded": False,
        "same_data_and_per_round_budget": False,
        "same_data_and_calibration": True,
        "reference": str(OLD), "manifest_sha256": new["scenario"]["manifest_sha256"],
        "physical_config": new_config,
        "changes_from_historical_run": [
            "masked idle/charge/discharge mode and conditional source powers",
            "gamma 0.9801 -> 0.99 (shared training and evaluation)",
            "rollout 64 -> 192; actor GAE lambda 0.9025 -> 0.97",
            "terminal inventory value 0 -> 0.3135 AUD/stored kWh",
            "candidate evaluation 96 periods -> complete months", "current Critic includes LayerNorm",
        ],
    }


def physical_row(env, before, samples, result):
    record = env.engine.latest_record
    p, ledger, action, sample = record.physical, record.ledger, record.uc_action, samples["uc"]
    mode = int(sample["storage_mode"])
    assert sample["mode_mask"][mode], "Unavailable storage mode sampled"
    assert not (sample["bounds"][4] == 0 and mode == 2), "Empty ESS discharge"
    return {
        "timestamp": record.timestamp.isoformat(), "period": record.period_id,
        "scenario_seed": int(sample["scenario_seed"]), "terminated": int(result.terminated),
        "mode": mode, "charge_available": int(sample["mode_mask"][1]),
        "discharge_available": int(sample["mode_mask"][2]),
        "baseline_kw": sum(o.baseline_kw for o in record.orders),
        "native_kw": sum(o.native_kw for o in record.orders),
        "requested_kw": sum(o.request_kw for o in record.orders),
        "native_lost_kw": sum(s.native_lost_kw for s in record.services.values()),
        "supply_shortfall_kw": ledger.supply_shortfall_kw,
        "grid_kw": p.grid_buy_kw, "der_kw": p.der_generation_kw,
        "der_to_consumer_kw": record.market.der_total_kw,
        "der_to_ess_kw": p.der_to_ess_kw, "grid_to_ess_kw": p.grid_to_ess_kw,
        "ess_discharge_kw": p.ess_discharge_kw, "der_spill_kw": p.der_spill_kw,
        "served_kw": p.supply_served_kw, "soc_before_kwh": before.ess_energy_kwh,
        "soc_kwh": p.ess_energy_after_kwh,
        "initial_soc_kwh": env.engine.state.initial_ess_energy_kwh,
        "grid_rate": float(env.engine.profiles.grid_cost_per_kwh[record.period_id]),
        "uc_cash": ledger.cash_profit, "uc_book": ledger.book_profit,
        "uc_reward": record.rewards["uc"], "terminal_energy_value": ledger.terminal_energy_value,
        "consumer_cost": sum(c.total_cost for c in ledger.consumers.values()),
        "consumer_net_bill": sum(c.net_bill for c in ledger.consumers.values()),
        "dr_payment": ledger.dr_payment, "dr_net_kw": record.dr.net_dr_kw,
        "remaining_backlog_kwh": sum(s.backlog_after_kwh for s in record.services.values()),
        "energy_residual": p.energy_residual_kwh, "power_residual": p.power_residual_kw,
        "cash_residual": ledger.cash_residual,
        **{"offer_" + key: value for key, value in vars(action).items()},
    }


class StudyObserver:
    """Append-only evidence; never samples an action or changes learner/RNG state."""

    def __init__(self, output, protocol):
        self.output, self.protocol = output, protocol
        self.stack, self.files = ExitStack(), {}
        self.algorithm = self.audit = None
        self.validation_round = None
        self.batch_id, self.active_batch = 0, None
        self.last_status_time = 0
        self.status = {"status": "starting", "phase": "setup", "pid": os.getpid(),
                       "started_at": utc_now(), "completed_rounds": 0, "target_rounds": ROUNDS,
                       "output": str(output)}

    def update_status(self, **fields):
        self.status.update(fields)
        self.status["updated_at"] = utc_now()
        temporary = self.output / "status.next.json"
        temporary.write_text(json.dumps(self.status, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.output / "status.json")
        self.last_status_time = time.monotonic()

    def append(self, name, row):
        if name not in self.files:
            path = self.output / "data" / name
            stream = path.open("x", encoding="utf-8-sig", newline="")
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            self.files[name] = (stream, writer)
        stream, writer = self.files[name]
        writer.writerow(row)
        if row["period"] % 48 == 47 or row["terminated"]:
            stream.flush()

    def on_ready(self, algorithm, audit):
        self.algorithm, self.audit = algorithm, audit
        (self.output / "data").mkdir()
        write_json(self.output / "protocol.json", self.protocol)
        # Preserve the actual dirty-worktree source used by this run.
        sources = json.loads((self.output / "experiment.json").read_text(encoding="utf-8"))["source_sha256"]
        from experiments.sn_energy_market.evaluation import source_hashes
        sources.update(source_hashes("environments/sn_energy_market/src"))
        for relative, digest in sources.items():
            raw = (ROOT / relative).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == digest
            target = self.output / "sources" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        write_json(self.output / "source_manifest.json", sources)
        runner = algorithm.env_runner
        original_period, original_collect = runner._period, runner.collect

        def period(**kwargs):
            before = runner.env.engine.state
            samples, result = original_period(**kwargs)
            if self.active_batch is not None:
                row = physical_row(runner.env, before, samples, result)
                self.append("training-all-branches.csv", {"batch_id": self.active_batch, **row})
            elif self.validation_round is not None:
                # The no-keyword call is full_value_validation's stochastic MC.
                # Its trajectory is the same as evaluate_pair(sampled); only save once.
                if not kwargs or kwargs.get("deterministic"):
                    kind = "mean" if kwargs.get("deterministic") else "sampled"
                    name = f"validation-{self.validation_round:04d}-{kind}-{runner._scenario_seed}.csv"
                    self.append(name, physical_row(runner.env, before, samples, result))
            if time.monotonic() - self.last_status_time > 10:
                self.update_status(phase=self.audit.phase, scenario_seed=runner._scenario_seed,
                                   current_timestamp=before.timestamp.isoformat(),
                                   ppo_sampled_periods=algorithm.sampled_periods,
                                   collected_batches=self.batch_id)
            return samples, result

        def collect(steps, **kwargs):
            active = self.audit.phase.startswith("round_")
            if active:
                assert steps == 64
                self.batch_id += 1
                self.active_batch = self.batch_id
            try:
                return original_collect(steps, **kwargs)
            finally:
                self.active_batch = None

        self.stack.enter_context(patch.object(runner, "_period", period))
        self.stack.enter_context(patch.object(runner, "collect", collect))
        self.update_status(status="running")

    def before_evaluation(self, number):
        self.validation_round = number
        self.update_status(phase=f"validation_{number}")

    def after_evaluation(self, number, evaluation):
        self.validation_round = None
        self.flush()
        # Independently reconcile saved physical records with the published totals.
        for episode in evaluation["critic"]["episodes"]:
            path = self.output / "data" / f"validation-{number:04d}-sampled-{episode['seed']}.csv"
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            assert len(rows) == episode["periods"]
            assert sum(int(r["terminated"]) for r in rows) == 1 and int(rows[-1]["terminated"]) == 1
            for column, key, scale in (("der_to_ess_kw", "der_charge_kwh", .5),
                                       ("grid_to_ess_kw", "grid_charge_kwh", .5),
                                       ("ess_discharge_kw", "ess_discharge_kwh", .5),
                                       ("native_lost_kw", "native_lost_kwh", .5),
                                       ("uc_cash", "uc_cash", 1), ("uc_book", "uc_book_profit", 1)):
                actual = sum(float(r[column]) * scale for r in rows)
                assert abs(actual - episode["metrics"][key]) < 1e-7, (column, actual)
            sampled = next(r for r in evaluation["performance"]["sampled"]["rows"] if r["seed"] == episode["seed"])
            assert abs(sampled["returns"]["uc"] - episode["returns"]["uc"]) < 1e-7
        self.update_status(last_validated_round=number, phase="training")

    def after_round(self, number, row):
        self.flush()
        assert self.batch_id == number * 8
        self.update_status(completed_rounds=number, ppo_sampled_periods=self.algorithm.sampled_periods,
                           main_periods=row["end"]["main_periods"], selected_candidate=row["round"]["selected_candidate"])

    def on_completed(self, report):
        self.close()
        self.update_status(status="postprocessing", phase="plots", checkpoint_exact_replay=report["checkpoint_exact_replay"])
        subprocess.run([str(ROOT / ".venv-plots/Scripts/python.exe"),
                        str(ROOT / "experiments/sn_energy_market/plot_hybrid_storage.py"),
                        str(self.output)], cwd=ROOT, check=True)
        self.update_status(status="completed", phase="completed", completed_at=utc_now(),
                           report=str(self.output / "comparison/report.html"),
                           summary=str(self.output / "comparison/summary.json"))

    def on_failure(self, exc):
        self.update_status(status="failed", error=repr(exc), failed_at=utc_now())

    def flush(self):
        for stream, _ in self.files.values():
            stream.flush()

    def close(self):
        self.stack.close()
        for stream, _ in self.files.values():
            stream.close()
        self.files.clear()


def main():
    protocol = check_protocol(OUTPUT)
    observer = StudyObserver(OUTPUT, protocol)
    run_pilot(OUTPUT, REFERENCE, outer_rounds=ROUNDS, observer=observer, checkpoint_each_round=True)


if __name__ == "__main__":
    main()
