"""Reconcile measured storage trajectories and plot the ten-round comparison."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.sn_energy_market.plot_batch_losses import load_batches

OLD = ROOT / "outputs/ausgrid-year-study/20261020"
SEEDS = (10042, 10045, 10048, 10051)
COLORS = {"old72": "#64748b", "new0": "#e8a031", "new3": "#7d56a3", "new10": "#087f8c"}
LABELS = {"old72": "旧版 · 72 轮", "new0": "新版 · 初始策略", "new3": "新版 · 3 轮", "new10": "新版 · 10 轮"}
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                     "axes.unicode_minus": False, "font.size": 10, "axes.titlesize": 12,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [{k: v if k in ("timestamp", "band") else float(v) for k, v in r.items()} for r in rows]


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def values(rows, key):
    return np.asarray([r[key] for r in rows], dtype=float)


def daily(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["timestamp"][:10]].append(row)
    result = []
    for date, day in grouped.items():
        result.append({"date": date, **{k + "_kwh": float(values(day, k + "_kw").sum() * .5)
                       for k in ("der_to_ess", "grid_to_ess", "ess_discharge", "native_lost", "native", "der_spill")},
                       **{k: float(values(day, k).sum()) for k in ("uc_cash", "uc_book", "uc_reward")},
                       "soc_mean_kwh": float(values(day, "soc_kwh").mean()),
                       "soc_min_kwh": float(values(day, "soc_kwh").min()),
                       "soc_max_kwh": float(values(day, "soc_kwh").max())})
    return result


def summarize(rows, system, inventory_rate, old=False):
    dt, initial = system["resolution_minutes"] / 60, system["ess_initial_energy_kwh"]
    groups = defaultdict(list)
    for row in rows:
        groups[row["timestamp"][:7]].append(row)
    stored_delta = sum(r[-1]["soc_kwh"] - initial for r in groups.values())
    end_value = inventory_rate * stored_delta
    energy = {k + "_kwh": float(values(rows, k + "_kw").sum() * dt)
              for k in ("der_to_ess", "grid_to_ess", "ess_discharge", "native_lost", "native", "der_spill", "der")}
    day = daily(rows)
    charge = values(rows, "der_to_ess_kw") + values(rows, "grid_to_ess_kw")
    near_empty = system["ess_energy_min_kwh"] + .01 * (system["ess_energy_capacity_kwh"] - system["ess_energy_min_kwh"])
    segments = {}
    for label, lo, hi in (("early_1_10", 1, 10), ("middle_11_20", 11, 20), ("late_21_end", 21, 31)):
        part = [r for r in rows if lo <= int(r["timestamp"][8:10]) <= hi]
        d = daily(part)
        segments[label] = {"days": len(d), "charge_kwh": float(sum(x["der_to_ess_kwh"] + x["grid_to_ess_kwh"] for x in d)),
                           "charging_days": sum(x["der_to_ess_kwh"] + x["grid_to_ess_kwh"] > 1e-6 for x in d),
                           "discharge_kwh": float(values(part, "ess_discharge_kw").sum() * dt),
                           "near_empty_fraction": float(np.mean(values(part, "soc_kwh") <= near_empty))}
    rewards = float(values(rows, "uc_reward").sum())
    result = {**energy, "periods": len(rows), "days": len(day),
              "charge_kwh": float(charge.sum() * dt),
              "uc_cash_per_day": float(values(rows, "uc_cash").sum() / len(day)),
              "uc_book_per_day": float(values(rows, "uc_book").sum() / len(day)),
              "native_reward_undiscounted": rewards,
              "common_inventory_adjusted_reward": rewards + end_value if old else rewards,
              "inventory_change_kwh": stored_delta, "terminal_inventory_adjustment": end_value,
              "service_rate": 1 - energy["native_lost_kwh"] / energy["native_kwh"],
              "der_utilization": 1 - energy["der_spill_kwh"] / energy["der_kwh"],
              "charging_period_fraction": float(np.mean(charge > 1e-6)),
              "charging_days": sum(x["der_to_ess_kwh"] + x["grid_to_ess_kwh"] > 1e-6 for x in day),
              "near_empty_fraction": float(np.mean(values(rows, "soc_kwh") <= near_empty)),
              "soc_mean_kwh": float(values(rows, "soc_kwh").mean()),
              "soc_end_by_month": {m: r[-1]["soc_kwh"] for m, r in groups.items()},
              "segments": segments}
    if not old:
        modes = values(rows, "mode")
        result.update(mode_fraction={str(i): float(np.mean(modes == i)) for i in range(3)},
                      empty_discharge_modes=int(np.sum((values(rows, "discharge_available") == 0) & (modes == 2))),
                      blocked_charge_modes=int(np.sum((values(rows, "charge_available") == 0) & (modes == 1))))
        assert result["empty_discharge_modes"] == result["blocked_charge_modes"] == 0
        assert abs(values(rows, "terminal_energy_value").sum() - end_value) < 1e-8
    return result


def verify_month(rows, system, *, new):
    assert len(rows) in (1440, 1488)
    assert values(rows, "period").tolist() == list(range(len(rows)))
    dt, eta_c, eta_d = .5, system["ess_charge_efficiency"], system["ess_discharge_efficiency"]
    previous = np.r_[system["ess_initial_energy_kwh"], values(rows, "soc_kwh")[:-1]]
    expected = previous + eta_c * dt * (values(rows, "der_to_ess_kw") + values(rows, "grid_to_ess_kw")) - dt * values(rows, "ess_discharge_kw") / eta_d
    assert np.max(np.abs(expected - values(rows, "soc_kwh"))) < 1e-8
    assert np.max((values(rows, "der_to_ess_kw") + values(rows, "grid_to_ess_kw")) * values(rows, "ess_discharge_kw")) < 1e-8
    for key in ("energy_residual", "power_residual", "cash_residual"):
        assert np.max(np.abs(values(rows, key))) < 1e-8
    if new:
        assert sum(r["terminated"] for r in rows) == 1 and rows[-1]["terminated"] == 1
    for row in rows:
        assert all(np.isfinite(v) for k, v in row.items() if k not in ("timestamp", "band"))


def save(fig, directory, name, title, figures):
    fig.suptitle(title, fontsize=15)
    for axis in fig.axes:
        axis.grid(alpha=.22)
    for ext in ("png", "svg"):
        fig.savefig(directory / f"{name}.{ext}", dpi=160)
    figures.append((name, title))
    plt.close(fig)


def main(output):
    output = Path(output)
    target = output / "comparison"
    target.mkdir(exist_ok=True)
    figures = []
    report, protocol = read_json(output / "report.json"), read_json(output / "protocol.json")
    assert report["status"] == "completed" and report["outer_rounds"] == 10
    assert report["ppo_sampled_periods"] == protocol["ppo_periods"]
    assert report["main_advanced_periods"] == protocol["main_periods"]
    assert report["logged_batches"] == protocol["batches"] and report["checkpoint_exact_replay"]
    assert report["critic_optimizer_steps_executed_per_module"] == protocol["critic_steps_per_module"]
    contract = report["learning_contract"]
    system = protocol["physical_config"]["system"]
    inventory = protocol["physical_config"]["rewards"]["terminal_energy_value_per_kwh"]
    paths, data, summaries = [], {}, {}
    for label, number in (("old72", 72), ("new0", 0), ("new3", 3), ("new10", 10)):
        data[label] = {}
        for seed in SEEDS:
            path = OLD / f"review-20261021/data/current-{seed}.csv" if label == "old72" else output / f"data/validation-{number:04d}-sampled-{seed}.csv"
            paths.append(path)
            rows = read_csv(path)
            verify_month(rows, system, new=label != "old72")
            data[label][seed] = rows
            if label != "old72":
                assert [r["timestamp"] for r in rows] == [r["timestamp"] for r in data["old72"][seed]]
        all_rows = [row for seed in SEEDS for row in data[label][seed]]
        summaries[label] = summarize(all_rows, system, inventory, old=label == "old72")
    # Reconcile historical detailed curves with the saved full-month report.
    evidence = read_json(OLD / "review-20261021/data/original_month_evidence.json")
    for seed in SEEDS:
        old = data["old72"][seed]
        expected = evidence["months"][f"current-{seed}"]
        assert abs(values(old, "uc_cash").sum() - expected["uc_cash"]) < 1e-7
        assert abs(values(old, "native_lost_kw").sum() * .5 - expected["native_lost_kwh"]) < 1e-7

    monthly, daily_rows, segment_rows = [], [], []
    for label, months in data.items():
        for seed, rows in months.items():
            stats = summarize(rows, system, inventory, old=label == "old72")
            monthly.append({"policy": label, "month": rows[0]["timestamp"][:7],
                            **{k: v for k, v in stats.items() if not isinstance(v, dict)}})
            daily_rows.extend({"policy": label, **d} for d in daily(rows))
        for segment, row in summaries[label]["segments"].items():
            segment_rows.append({"policy": label, "segment": segment, **row})
    # Keep CSV columns uniform: mode diagnostics are separate JSON fields.
    shared = set.intersection(*(set(row) for row in monthly))
    write_csv(target / "monthly_metrics.csv", [{k: v for k, v in r.items() if k in shared} for r in monthly])
    write_csv(target / "daily_metrics.csv", daily_rows)
    write_csv(target / "within_month_metrics.csv", segment_rows)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    keys = [("der_to_ess_kwh", "DER → ESS / kWh"), ("grid_to_ess_kwh", "电网 → ESS / kWh"),
            ("ess_discharge_kwh", "ESS 实际放电 / kWh"), ("near_empty_fraction", "接近空电时段比例"),
            ("uc_cash_per_day", "UC 现金 / AUD/日"), ("native_lost_kwh", "原生失供 / kWh")]
    for axis, (key, label) in zip(axes.flat, keys):
        bars = axis.bar([LABELS[k] for k in data], [summaries[k][key] for k in data], color=[COLORS[k] for k in data])
        axis.bar_label(bars, fmt="%.3f", fontsize=9, padding=3)
        axis.set_title(label)
        axis.tick_params(axis="x", labelrotation=15)
        axis.margins(y=.2)
    save(fig, target, "01-overview", "四个完整验证月：123 天、5,904 个半小时", figures)

    # One selected complete-month candidate score for each round. No short-window overlay.
    selected = [r["round"]["candidates"][r["round"]["selected_candidate"]]["rows"] for r in report["rounds"]]
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True, layout="constrained")
    trend_keys = [("der_charge_kwh", "DER 充电 / kWh"), ("grid_charge_kwh", "电网充电 / kWh"),
                  ("ess_discharge_kwh", "实际放电 / kWh"), ("native_lost_kwh", "原生失供 / kWh"),
                  ("uc_cash", "UC 现金 / AUD"), ("reward", "完整月库存调整回报 / AUD")]
    trend = []
    for n, candidates in enumerate(selected, 1):
        trend.append({"round": n, **{key: sum(r["returns"]["uc"] if key == "reward" else r["metrics"][key] for r in candidates) for key, _ in trend_keys}})
    write_csv(target / "round_metrics.csv", trend)
    for axis, (key, label) in zip(axes.flat, trend_keys):
        axis.plot(range(1, 11), [r[key] for r in trend], "o-", color=COLORS["new10"])
        axis.set_ylabel(label)
        axis.set_xticks(range(1, 11))
    for axis in axes[-1]:
        axis.set_xlabel("外层轮数")
    save(fig, target, "02-round-trends", "新版每轮被选候选：完整月份评分（含验证集选择效应）", figures)

    batches = load_batches(output / "batch_losses.jsonl")
    assert len(batches) == protocol["batches"] and all(r["branch_status"] != "pending" for r in batches)
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, layout="constrained")
    for axis, key, label in zip(axes, ("critic_loss", "critic_rmse", "policy_loss"), ("Critic loss", "Critic RMSE / AUD", "UC Actor PPO loss")):
        for stage, style in (("before", "--"), ("after", "-")):
            y = [r["modules"]["uc"][f"{key}_{stage}"] for r in batches]
            axis.plot(range(1, len(batches) + 1), [np.nan if x is None else x for x in y], style, label=stage)
        axis.set_ylabel(label)
        axis.legend()
    axes[-1].set_xlabel("实际采样批编号，保留全部候选分支；冻结 Actor 留空")
    save(fig, target, "03-ppo-loss", f"{len(batches)} 批实际更新：批间策略与目标变化，loss 不代表整体收敛", figures)

    for seed in SEEDS:
        month = data["old72"][seed][0]["timestamp"][:7]
        fig, axes = plt.subplots(5, 1, figsize=(15, 13), sharex=True, layout="constrained")
        for label in ("old72", "new10"):
            rows = data[label][seed]
            x = values(rows, "period") / 48 + 1
            for axis, key, unit in zip(axes, ("der_to_ess_kw", "grid_to_ess_kw", "ess_discharge_kw", "soc_kwh", "native_lost_kw"),
                                       ("DER 充电 / kW", "电网充电 / kW", "放电 / kW", "SOC（储能量）/ kWh", "原生失供 / kW")):
                axis.plot(x, values(rows, key), lw=.8, color=COLORS[label], alpha=.85, label=LABELS[label])
                axis.set_ylabel(unit)
        axes[3].set_ylim(-.05, system["ess_energy_capacity_kwh"] * 1.05)
        axes[0].legend(ncol=2)
        axes[-1].set_xlabel("月内日期，曲线保留全部半小时值")
        save(fig, target, f"storage-{month}", f"{month}：两路充电、放电、SOC 与失供", figures)

        fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True, layout="constrained")
        for label in ("old72", "new10"):
            rows = data[label][seed]
            x = values(rows, "period") / 48 + 1
            for axis, key, name in zip(axes, ("uc_cash", "uc_book", "uc_reward", "native_lost_kw"),
                                       ("累计现金 / AUD", "累计经营收益 / AUD", "累计统一库存调整回报 / AUD", "累计原生失供 / kWh")):
                y = np.cumsum(values(rows, key) * (.5 if key.endswith("_kw") else 1))
                if label == "old72" and key == "uc_reward":
                    y[-1] += inventory * (rows[-1]["soc_kwh"] - system["ess_initial_energy_kwh"])
                axis.plot(x, y, label=LABELS[label], color=COLORS[label])
                axis.set_ylabel(name)
        axes[0].legend()
        axes[-1].set_xlabel("月内日期；库存估值在真实月末计入")
        save(fig, target, f"economics-{month}", f"{month}：累计收益与失供（现金、经营收益、训练回报分别列出）", figures)

    fig, axes = plt.subplots(7, 1, figsize=(16, 17), sharex=True, layout="constrained")
    fields = ("der_to_ess_kwh", "grid_to_ess_kwh", "ess_discharge_kwh", "soc_mean_kwh", "uc_cash", "uc_book", "native_lost_kwh")
    labels = ("DER 充电 / kWh", "电网充电 / kWh", "放电 / kWh", "日均储能量 / kWh", "现金 / AUD", "经营收益 / AUD", "原生失供 / kWh")
    for label in data:
        rows = [r for r in daily_rows if r["policy"] == label]
        for axis, key, name in zip(axes, fields, labels):
            axis.plot(np.arange(123) + 1, [r[key] for r in rows], label=LABELS[label], color=COLORS[label], lw=1, alpha=.8)
            axis.set_ylabel(name)
            for boundary in (31.5, 62.5, 93.5):
                axis.axvline(boundary, color="grey", ls=":", lw=.7)
    axes[0].legend(ncol=4)
    axes[-1].set_xlabel("验证日序号：2011-07 / 2011-10 / 2012-01 / 2012-04；分隔线处重新初始化")
    save(fig, target, "daily-all-months", "123 个验证日：新版初始、3 轮、10 轮及旧版 72 轮", figures)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for label, months in data.items():
        rows = [r for month in months.values() for r in month]
        soc = np.sort(values(rows, "soc_kwh"))
        axes[0].plot(soc, np.arange(1, len(soc) + 1) / len(soc), color=COLORS[label], label=LABELS[label])
        segments = summaries[label]["segments"]
        axes[1].plot(range(3), [r["charge_kwh"] / r["days"] for r in segments.values()], "o-", label=LABELS[label], color=COLORS[label])
        axes[2].plot(range(3), [r["near_empty_fraction"] for r in segments.values()], "o-", color=COLORS[label])
    axes[0].set(xlabel="储能量 / kWh", ylabel="累计时段比例", title="SOC 分布")
    axes[0].legend()
    axes[1].set(ylabel="日均充电 / kWh", title="月初、中旬、月末是否持续充电")
    axes[2].set(ylabel="接近空电时段比例", title="接近空电：可用容量的 1% 以内")
    for axis in axes[1:]:
        axis.set_xticks(range(3), ["1–10 日", "11–20 日", "21 日至月末"])
    save(fig, target, "persistence-soc", "区分持续补能与只消耗月初电量", figures)

    # Predetermined selection: for every month show the old policy's largest-loss day.
    case_days = []
    fig, axes = plt.subplots(4, 3, figsize=(17, 14), layout="constrained")
    for i, seed in enumerate(SEEDS):
        old_days = daily(data["old72"][seed])
        date = max(old_days, key=lambda r: r["native_lost_kwh"])["date"]
        case_days.append(date)
        for label in ("old72", "new10"):
            rows = [r for r in data[label][seed] if r["timestamp"].startswith(date)]
            x = np.arange(48) / 2
            charge = values(rows, "der_to_ess_kw") + values(rows, "grid_to_ess_kw")
            axes[i, 0].plot(x, charge, label=LABELS[label] + " 充电", color=COLORS[label])
            axes[i, 0].plot(x, -values(rows, "ess_discharge_kw"), "--", color=COLORS[label], label=LABELS[label] + " 负号放电")
            axes[i, 1].plot(x, values(rows, "soc_kwh"), color=COLORS[label], label=LABELS[label])
            axes[i, 2].plot(x, values(rows, "native_lost_kw"), color=COLORS[label], label=LABELS[label])
        for j, ylabel in enumerate(("充放电 / kW", "储能量 / kWh", "原生失供 / kW")):
            axes[i, j].set(title=date, xlabel="小时", ylabel=ylabel)
    axes[0, 0].legend(fontsize=8)
    save(fig, target, "matched-loss-days", "各月旧策略失供最多的一天：固定同日比较", figures)

    # The main trajectory excludes proposals and the unselected response branch.
    training = read_csv(output / "data/training-all-branches.csv")
    assert len(training) == protocol["ppo_periods"]
    metadata = {r["batch_id"]: r for r in batches}
    retained = [r for r in training if metadata[int(r["batch_id"])]["sampling_retained"]]
    assert len(retained) == protocol["main_periods"]
    write_csv(target / "training-main-trajectory.csv", retained)
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True, layout="constrained")
    for axis, key, label in zip(axes, ("der_to_ess_kw", "grid_to_ess_kw", "ess_discharge_kw", "soc_kwh"),
                               ("DER 充电 / kW", "电网充电 / kW", "放电 / kW", "储能量 / kWh")):
        axis.plot(np.arange(len(retained)) / 48, values(retained, key), lw=.8, color=COLORS["new10"])
        axis.set_ylabel(label)
        for index in np.flatnonzero(values(retained, "period") == 0):
            axis.axvline(index / 48, color="grey", ls=":")
    axes[-1].set_xlabel("训练主轨迹累计天数；竖线为真实月初重置")
    save(fig, target, "training-storage", f"10 轮实际保留主轨迹：{len(retained):,} 时段，约 {len(retained) / 48:.1f} 天", figures)

    mean_rows = []
    for seed in SEEDS:
        rows = read_csv(output / f"data/validation-0010-mean-{seed}.csv")
        verify_month(rows, system, new=True)
        mean_rows.extend(rows)
    mean_summary = summarize(mean_rows, system, inventory)
    old, initial, new = summaries["old72"], summaries["new0"], summaries["new10"]
    continued = all(r["charge_kwh"] > 1e-3 for r in new["segments"].values())
    charged_months = sum(r["charge_kwh"] > 1e-3 for r in monthly if r["policy"] == "new10")
    verdict = ("新版在四个验证月的月初、中旬和月末均发生充电，持续不充电的现象已被打破。"
               if continued and charged_months == 4 else "新版仍存在充电不足：需要结合各月与月内分段曲线定位。")
    summary = {"status": "completed", "protocol": protocol, "summary": summaries,
               "deterministic_new10": mean_summary, "continued_charging_all_segments": continued,
               "matched_case_days": case_days, "verdict": verdict,
               "checks": {"budget": True, "checkpoint_replay": True, "matched_calendar": True,
                          "energy_balance": True, "charge_discharge_exclusion": True,
                          "historical_reconciliation": True, "training_main_periods": len(retained)},
               "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
               "figures": [name for name, _ in figures]}
    (target / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    table_fields = [("der_to_ess_kwh", "DER 充电 / kWh"), ("grid_to_ess_kwh", "电网充电 / kWh"),
                    ("ess_discharge_kwh", "实际放电 / kWh"), ("charging_days", "充电天数 / 123"),
                    ("near_empty_fraction", "接近空电时段比例"), ("uc_cash_per_day", "现金 / AUD·日⁻¹"),
                    ("uc_book_per_day", "经营收益 / AUD·日⁻¹"), ("native_lost_kwh", "原生失供 / kWh"),
                    ("service_rate", "原生供电率"), ("der_spill_kwh", "DER 弃电 / kWh"),
                    ("common_inventory_adjusted_reward", "统一库存调整回报 / AUD")]
    table = "<table><tr><th>指标</th>" + "".join(f"<th>{LABELS[k]}</th>" for k in data) + "</tr>"
    for key, title in table_fields:
        table += "<tr><td>" + title + "</td>" + "".join(f"<td>{summaries[k][key]:,.5f}</td>" for k in data) + "</tr>"
    table += "</table>"
    historical = []
    for n in (0, 3, 12, 24, 48, 72):
        validation = read_json(OLD / f"validation-{n:04d}.json")
        episodes = validation["critic"]["episodes"]
        lost = sum(r["metrics"]["native_lost_kwh"] for r in episodes)
        cash = sum(r["metrics"]["uc_cash"] for r in episodes) / 123
        historical.append(f"<tr><td>{n}</td><td>{cash:.5f}</td><td>{lost:.5f}</td></tr>")
    report_html = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>储能三模式：10 轮实测对比</title>
<style>body{{max-width:1320px;margin:36px auto;padding:0 24px;font:17px/1.8 'Microsoft YaHei',sans-serif;color:#16324a}}h1,h2{{line-height:1.35}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{padding:8px 12px;border:1px solid #dce4ec;text-align:right}}td:first-child,th:first-child{{text-align:left}}th{{background:#eaf4f6}}figure{{margin:40px 0}}img{{width:100%;height:auto}}a{{color:#087f8c}}.lead{{padding:20px;background:#edf7f7;border-left:5px solid #087f8c}}</style>
<h1>储能三模式：10 轮实测与旧年度实验对比</h1>
<p class="lead">{verdict}<br>新版总充电 {new['charge_kwh']:.4f} kWh，旧第 72 轮为 {old['charge_kwh']:.6f} kWh；新版充电覆盖 {new['charging_days']} / 123 天，接近空电比例 {new['near_empty_fraction']:.2%}。现金变化 {new['uc_cash_per_day']-old['uc_cash_per_day']:+.4f} AUD/日，原生失供变化 {new['native_lost_kwh']-old['native_lost_kwh']:+.4f} kWh。</p>
<h2>1. 充电恢复与训练效果分别看</h2><p>新版初始策略已充电 {initial['charge_kwh']:.4f} kWh，10 轮后为 {new['charge_kwh']:.4f} kWh。初始到第 10 轮的变化才表示这次学习带来的策略变化。充电增多的经济意义由现金、经营收益、弃电和失供共同判断。</p>
{table}
<p>随机策略按保存种子生成完整月份轨迹。第 10 轮确定性推理另测：充电 {mean_summary['charge_kwh']:.4f} kWh，放电 {mean_summary['ess_discharge_kwh']:.4f} kWh，现金 {mean_summary['uc_cash_per_day']:.4f} AUD/日，失供 {mean_summary['native_lost_kwh']:.4f} kWh。两种执行方式的原始数据均在 data 目录。</p>
<h2>2. 三个问题的检查</h2>
<p><b>来源控制：</b>DER 和电网功率分别记录，充电模式中各自由对应控制量决定，仍共同受电池充电上限约束。充电曲线与弃电指标可检查 DER 是否进入储能，以及电网充电是否增加采购成本。</p>
<p><b>空电放电：</b>第 10 轮完整验证中，禁用放电时仍选择放电为 {new['empty_discharge_modes']} 次，禁用充电时仍选择充电为 {new['blocked_charge_modes']} 次。这里按决策时可用动作掩码检查；接近空电定义为储能量低于可用容量的 1%，两个统计各有用途。</p>
<p><b>初始电量偏差：</b>每个验证月从 {system['ess_initial_energy_kwh']:.4f} kWh 开始，完整跑至月末。新版库存总变化 {new['inventory_change_kwh']:.4f} kWh，对回报的终端调整为 {new['terminal_inventory_adjustment']:.4f} AUD。按早、中、晚段计算充电，检验是否持续补能。</p>
<h2>3. 比较范围与预算</h2><p>本次 seed 42，10 轮、{protocol['batches']} 批、{protocol['ppo_periods']:,} 个 PPO 分支时段，主轨迹 {protocol['main_periods']:,} 时段，每个 Critic 实际执行 {protocol['critic_steps_per_module']:,} 次优化。沿用原 16 户、训练年、资源标定与四个验证月。每批 {protocol.get('phase_batch_periods', contract['rollout_length'])} 步，Actor GAE λ={contract['lambda_gae']}；与旧年度实验的每轮预算是否相同：{protocol['same_data_and_per_round_budget']}。主轨迹从训练年开头推进约 {len(retained) / 48:.1f} 天，测试集未加载。</p>
<p>当前三模式动作、完整月候选评价、gamma={contract['gamma']}、终端库存估值和 Critic LayerNorm，与旧年度实验存在协议差异。旧第 72 轮作为历史终态参照。旧实验保存的完整月验证节点列于下表；第 10 轮没有旧版完整月结果，因此不作相同轮数的效果归因。验证月参与候选选择，这些是验证集成绩。</p>
<table><tr><th>旧版轮次</th><th>现金 / AUD·日⁻¹</th><th>原生失供 / kWh</th></tr>{''.join(historical)}</table>
<p>现金为账本 cash_profit，经营收益为 book_profit，训练回报还含服务考核等项。统一库存调整回报将旧策略原始逐期奖励不折扣求和，再加 {inventory:.4f}×(期末库存−期初库存)；新版使用原始完整月回报。收益图将这三种口径分别展示。供电率的分母为各自策略的原生请求量。</p>
<h2>4. 全部曲线</h2><p>月曲线保留每个半小时点；日曲线只聚合当天实测值。案例日期固定选取旧策略各月失供最多的一天，不按新版效果选日。PNG 可直接引用，SVG 可编辑。</p>
"""
    for name, title in figures:
        report_html += f'<figure><h3>{html.escape(title)}</h3><a href="{name}.svg"><img src="{name}.png" loading="lazy" alt="{html.escape(title)}"></a></figure>'
    report_html += '<h2>5. 数据与核验</h2><p>预算、逐批编号、主轨迹筛选、SOC 递推、充放电互斥、账本守恒、真实月末结算、旧结果核对和 checkpoint 8 步精确重放均已通过。</p><p><a href="summary.json">机器摘要与来源哈希</a> · <a href="monthly_metrics.csv">逐月统计</a> · <a href="daily_metrics.csv">逐日统计</a> · <a href="within_month_metrics.csv">月内分段</a> · <a href="training-main-trajectory.csv">训练主轨迹</a> · <a href="../data/">全部逐时段数据</a></p></html>'
    (target / "report.html").write_text(report_html, encoding="utf-8")
    print(json.dumps({"status": "plots_completed", "figures": len(figures), "report": str(target / "report.html"), "verdict": verdict}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
