"""Compare the two measured ten-round runs; no simulator or optimizer is constructed."""

from __future__ import annotations

import hashlib
import html
import json
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.sn_energy_market.plot_batch_losses import load_batches
from experiments.sn_energy_market.plot_hybrid_storage import (
    SEEDS,
    daily,
    plt,
    read_csv,
    read_json,
    save,
    summarize,
    values,
    verify_month,
    write_csv,
)
from experiments.sn_energy_market.plot_mode_learning import add_mode_learning

NAMES = ("闲置", "充电", "放电")
MODE_KEYS = ("idle", "charge", "discharge")
LABELS = {"initial": "新初始策略", "previous_initial": "旧初始策略", "previous": "上一版 10 轮",
          "round3": "新 3 轮", "current": "新 10 轮"}
COLORS = {"initial": "#c78e35", "previous_initial": "#ab967b", "previous": "#73669a", "round3": "#75a49b", "current": "#007f86"}


def discounted_returns(rows, gamma):
    result = np.empty(len(rows))
    carry = 0.0
    for i in reversed(range(len(rows))):
        carry = rows[i]["uc_reward"] + gamma * carry * (not rows[i]["terminated"])
        result[i] = carry
    return result


def critic_metrics(rows, published, gamma):
    target = discounted_returns(rows, gamma)
    result = dict(published)
    result.update(gamma=gamma, target_std=float(target.std()),
                  rmse_over_target_std=float(published["rmse"] / target.std()),
                  rmse_over_mean_abs_target=float(published["rmse"] / np.mean(np.abs(target))))
    assert abs(result["return_mean"] - target.mean()) < 1e-5
    if "uc_value" in rows[0]:
        prediction = values(rows, "uc_value")
        assert abs(np.sqrt(np.mean((prediction - target) ** 2)) - result["rmse"]) < 1e-5
        result["prediction_std"] = float(prediction.std())
    return result


def probability_metrics(probs, mask):
    legal = mask.all(axis=1)
    p = probs[legal]
    assert np.allclose(probs.sum(axis=1), 1) and np.all(probs[~mask] == 0)
    return {"all_legal_states": int(legal.sum()), "mean": p.mean(0).tolist(),
            "std": p.std(0).tolist(), "argmax_counts": np.bincount(p.argmax(1), minlength=3).tolist(),
            "entropy": float(-(p * np.log(p)).sum(1).mean())}


def consumer_critic_metrics(modules):
    per_agent = {}
    for name, module in modules.items():
        if name == "uc":
            continue
        row = dict(module["all"])
        # MSE = Var(error) + bias^2; EV = 1 - Var(error) / Var(target).
        row["target_std"] = float(np.sqrt((row["rmse"] ** 2 - row["bias"] ** 2)
                                           / (1 - row["explained_variance"])))
        row["rmse_over_target_std"] = row["rmse"] / row["target_std"]
        per_agent[name] = row
    return {"per_agent": per_agent,
            "agent_mean": {key: float(np.mean([v[key] for v in per_agent.values()]))
                           for key in next(iter(per_agent.values()))}}


def table(headers, rows):
    return '<table><tr>' + ''.join(f'<th>{html.escape(str(h))}</th>' for h in headers) + '</tr>' + ''.join(
        '<tr>' + ''.join(f'<td>{html.escape(str(c))}</td>' for c in row) + '</tr>' for row in rows) + '</table>'


def daily_outcomes(rows, consumers):
    days = daily(rows)
    for day in days:
        selected = [r for r in rows if r["timestamp"].startswith(day["date"])]
        reward = -sum(r["consumer_cost"] for r in selected)
        if "consumer_reward" in selected[0]:
            assert abs(reward - sum(r["consumer_reward"] for r in selected)) < 1e-8
        day.update(consumer_reward=reward, consumer_reward_per_agent=reward / consumers)
    return days


def training_loss_comparison(target, output, previous, reports, figures):
    """Use real sampled-period counts so 40 and 80 batches share a budget axis."""
    paths, csv_rows = [], []
    for label, directory in (("previous", previous), ("current", output)):
        path = directory / "batch_losses.jsonl"
        paths.append(path)
        rows = load_batches(path)
        report = reports[label]
        assert len(rows) == report["logged_batches"]
        assert sum(r["samples"] for r in rows) == report["ppo_sampled_periods"]
        assert all(r["branch_status"] != "pending" for r in rows)
        periods = 0
        for row in rows:
            periods += row["samples"]
            assert periods == row["sampled_periods_lifetime"]
            for agent, measured in row["modules"].items():
                csv_rows.append({"policy": label, "agent": agent, "batch": row["batch_id"],
                                 "ppo_periods": periods, "samples": row["samples"],
                                 "phase": row["phase"], "branch_status": row["branch_status"],
                                 **{f"{key}_{stage}": measured[f"{key}_{stage}"]
                                    for key in ("critic_loss", "critic_rmse", "policy_loss")
                                    for stage in ("before", "after")},
                                 "critic_retained": row["retained_updates"][agent]["critic"],
                                 "actor_retained": row["retained_updates"][agent]["actor"]})
    write_csv(target / "training-losses.csv", csv_rows)
    for role, filename in (("uc", "12-uc-training-loss"), ("consumer", "13-consumer-training-loss")):
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, layout="constrained")
        for label in ("previous", "current"):
            selected = [r for r in csv_rows if r["policy"] == label and (r["agent"] == "uc") == (role == "uc")]
            xs = sorted({r["ppo_periods"] for r in selected})
            for axis, key, title in zip(axes, ("critic_loss", "critic_rmse", "policy_loss"),
                                        ("Critic loss", "Critic RMSE / 奖励单位", "Actor PPO loss")):
                for stage, style in (("before", "--"), ("after", "-")):
                    ys = []
                    rolled = []
                    for x in xs:
                        group = [r for r in selected if r["ppo_periods"] == x and r[f"{key}_{stage}"] is not None]
                        ys.append(float(np.mean([r[f"{key}_{stage}"] for r in group])) if group else np.nan)
                        retained_key = "actor_retained" if key == "policy_loss" else "critic_retained"
                        if stage == "after" and group and not any(r[retained_key] for r in group):
                            rolled.append((x, ys[-1]))
                    axis.plot(xs, ys, style, marker=".", markersize=4, lw=1,
                              color=COLORS[label], label=f"{LABELS[label]} · {stage}")
                    if rolled:
                        axis.scatter(*zip(*rolled), marker="x", color=COLORS[label], s=22)
                axis.set_ylabel(title)
                axis.grid(alpha=.2)
        axes[0].legend(fontsize=8, ncol=2)
        axis_scope = "有效路径的累计 PPO 分支采样时段" if any(r.get("recovery") for r in reports.values()) else "累计实际 PPO 分支采样时段"
        axes[-1].set_xlabel(axis_scope + "；叉号为回滚更新，冻结 Actor 留空")
        title = "UC" if role == "uc" else "16 位 Consumer 的逐批指标均值（逐户数值见 CSV）"
        save(fig, target, filename, title + "：相同采样预算轴上的训练 loss", figures)
    return paths


def main(output, previous):
    output, previous = Path(output).resolve(), Path(previous).resolve()
    report, old_report = read_json(output / "report.json"), read_json(previous / "report.json")
    rounds, old_rounds = report["outer_rounds"], old_report["outer_rounds"]
    target = output / f"comparison{rounds}"
    target.mkdir(exist_ok=True)
    protocol = read_json(output / "protocol.json")
    assert report["status"] == old_report["status"] == "completed"
    assert rounds in (10, 72) and old_rounds == 10
    assert report["ppo_sampled_periods"] == 1536 * rounds
    assert report["main_advanced_periods"] == 768 * rounds and report["logged_batches"] == protocol["batches"]
    effective_critic_steps = report.get("critic_optimizer_steps_effective_per_module",
                                       report["critic_optimizer_steps_executed_per_module"])
    assert effective_critic_steps == 192 * rounds and report["checkpoint_exact_replay"]
    assert not report["test_loaded"] and not report["test_evaluated"]
    system = protocol["physical_config"]["system"]
    equal_budget = report["ppo_sampled_periods"] == old_report["ppo_sampled_periods"]
    same_gamma = report["learning_contract"]["gamma"] == old_report["learning_contract"]["gamma"]
    if equal_budget:
        assert report["main_advanced_periods"] == old_report["main_advanced_periods"]
        assert effective_critic_steps == old_report.get("critic_optimizer_steps_effective_per_module", old_report["critic_optimizer_steps_executed_per_module"])
        changes = {k for k, v in report["learning_contract"].items()
                   if v != old_report["learning_contract"][k]}
        if "scenario" in changes:
            assert {k: v for k, v in report["learning_contract"]["scenario"].items() if k != "source_sha256"} == {
                k: v for k, v in old_report["learning_contract"]["scenario"].items() if k != "source_sha256"}
            changes.remove("scenario")
        assert changes <= {"update_protocol", "input_transform"}, changes
    same_inputs = report["learning_contract"]["input_transform"] == old_report["learning_contract"]["input_transform"]
    LABELS["previous"] = f'上一版 {old_rounds} 轮：{old_report["ppo_sampled_periods"] // old_report["logged_batches"]} 步/批'
    LABELS["current"] = f'新 {rounds} 轮：{report["ppo_sampled_periods"] // report["logged_batches"]} 步/批'
    inventory = protocol["physical_config"]["rewards"]["terminal_energy_value_per_kwh"]
    inputs = [output / "report.json", previous / "report.json", output / "protocol.json"]
    data, combined, summaries, daily_data, critic, consumer_critic = {}, {}, {}, {}, {}, {}
    specs = {"previous_initial": (previous, 0, old_report), "initial": (output, 0, report), "previous": (previous, old_rounds, old_report),
             "round3": (output, 3, report), "current": (output, rounds, report)}
    for label, (directory, number, result) in specs.items():
        data[label] = {}
        for seed in SEEDS:
            path = directory / f"data/validation-{number:04d}-sampled-{seed}.csv"
            rows = read_csv(path)
            verify_month(rows, system, new=True)
            inputs.append(path)
            data[label][seed] = rows
        combined[label] = [row for seed in SEEDS for row in data[label][seed]]
        assert len(combined[label]) == 5904
        summaries[label] = summarize(combined[label], system, inventory)
        daily_data[label] = daily_outcomes(combined[label], system["num_consumers"])
        summaries[label]["consumer_reward_per_day"] = float(np.mean([r["consumer_reward"] for r in daily_data[label]]))
        for episode in result["evaluations"][str(number)]["critic"]["episodes"]:
            assert abs(sum(r["consumer_cost"] for r in data[label][episode["seed"]]) - episode["metrics"]["consumer_cost"]) < 1e-7
        stats = result["evaluations"][str(number)]["critic"]["modules"]["uc"]["all"]
        critic[label] = critic_metrics(combined[label], stats, result["learning_contract"]["gamma"])
        consumer_critic[label] = consumer_critic_metrics(result["evaluations"][str(number)]["critic"]["modules"])
        summaries[label]["common_gamma_099_return"] = float(sum(
            discounted_returns(data[label][seed], .99)[0] for seed in SEEDS) / len(SEEDS))
    # Same seed does not imply identical initial policies after input-transform changes.
    initial_error = 0.0
    for seed in SEEDS:
        path = previous / f"data/validation-0000-sampled-{seed}.csv"
        inputs.append(path)
        old_rows = read_csv(path)
        new_rows = data["initial"][seed]
        assert len(old_rows) == len(new_rows)
        for left, right in zip(old_rows, new_rows):
            assert left["timestamp"] == right["timestamp"]
            initial_error = max(initial_error, *(abs(left[k] - right[k]) for k in left if k != "timestamp"))
    if same_inputs:
        assert initial_error < 1e-6, initial_error

    old_fixed_path = previous / f"data/fixed-state-probe-{old_rounds:04d}.npz"
    if old_fixed_path.exists():
        inputs.append(old_fixed_path)
        with np.load(old_fixed_path) as fixed:
            probabilities = {"previous": fixed["probs"]}
            previous_mask = fixed["mask"]
        old_fixed = None
    else:
        old_fixed_path = previous / "analysis/actor-mode-audit/fixed_state_probabilities.csv"
        old_fixed = read_csv(old_fixed_path)
        inputs.append(old_fixed_path)
        probabilities = {"previous": np.array([[r[f"final_{k}_prob"] for k in MODE_KEYS] for r in old_fixed])}
        previous_mask = None
    for label, number in (("initial", 0), ("round3", 3), ("current", rounds)):
        path = output / f"data/fixed-state-probe-{number:04d}.npz"
        inputs.append(path)
        with np.load(path) as fixed:
            probabilities[label] = fixed["probs"]
            mask = fixed["mask"]
    path = previous / "data/fixed-state-probe-0000.npz"
    if path.exists():
        inputs.append(path)
        with np.load(path) as fixed:
            probabilities["previous_initial"] = fixed["probs"]
    else:
        assert same_inputs
        probabilities["previous_initial"] = probabilities["initial"]
    if previous_mask is not None:
        np.testing.assert_array_equal(previous_mask, mask)
    mode_metrics = {label: probability_metrics(p, mask) for label, p in probabilities.items()}
    for old, row in zip(old_fixed or [], combined["previous"]):
        assert old["timestamp"] == row["timestamp"] and old["seed"] == row["scenario_seed"]
        row["uc_value"] = old["final_value"]
        for key in MODE_KEYS:
            row[f"prob_{key}"] = old[f"final_{key}_prob"]
    # Recheck predictions against the discount actually used by the reference.
    critic["previous"] = critic_metrics(combined["previous"],
                                        old_report["final"]["critic"]["modules"]["uc"]["all"],
                                        old_report["learning_contract"]["gamma"])
    deterministic = {}
    for label, directory in (("previous", previous), ("current", output)):
        rows = []
        for seed in SEEDS:
            number = old_rounds if label == "previous" else rounds
            path = directory / f"data/validation-{number:04d}-mean-{seed}.csv"
            inputs.append(path)
            part = read_csv(path)
            verify_month(part, system, new=True)
            rows.extend(part)
        deterministic[label] = summarize(rows, system, inventory)

    figures = []
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for i, label in enumerate(("initial", "previous", "current")):
        entry = mode_metrics[label]
        axes[0, 0].bar(np.arange(3) + (i - 1) * .24, np.array(entry["mean"]) * 100, .24,
                       label=LABELS[label], color=COLORS[label])
        axes[0, 1].bar(np.arange(3) + (i - 1) * .24, np.array(entry["std"]) * 100, .24, color=COLORS[label])
        axes[1, 0].bar(np.arange(3) + (i - 1) * .24, entry["argmax_counts"], .24, color=COLORS[label])
        axes[1, 1].bar(np.arange(3) + (i - 1) * .24,
                       [summaries[label]["mode_fraction"][str(m)] * 100 for m in range(3)], .24, color=COLORS[label])
    for ax, title in zip(axes.flat, ("相同状态：平均条件概率 / %", "相同状态：概率标准差 / 百分点",
                                    "相同状态与基础额度：模式 argmax 次数", "各自随机执行轨迹：实际模式频率 / %")):
        ax.set(xticks=np.arange(3), xticklabels=NAMES, title=title)
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].set_ylim(0, max(47, 115 * max(max(mode_metrics[k]["mean"]) for k in mode_metrics)))
    save(fig, target, "01-mode-probabilities", "网络概率与实际抽样频率分别比较；固定状态统计只含三模式均可用状态", figures)

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True, layout="constrained")
    for label in ("initial", "previous", "current"):
        days = daily_data[label]
        for ax, key in zip(axes, ("der_to_ess_kwh", "grid_to_ess_kwh", "ess_discharge_kwh", "soc_mean_kwh")):
            ax.plot([r[key] for r in days], color=COLORS[label], label=LABELS[label], alpha=.85)
    for ax, title in zip(axes, ("DER 充电 / kWh/日", "电网充电 / kWh/日", "实际放电 / kWh/日", "日平均储能量 / kWh")):
        ax.set_ylabel(title)
    axes[0].legend(ncol=3, fontsize=9)
    axes[-1].set_xlabel("123 个验证日：7 月、10 月、1 月、4 月")
    save(fig, target, "02-daily-storage", f"初始与 {old_rounds}/{rounds} 轮：两路充电、放电和 SOC", figures)

    fig, axes = plt.subplots(5, 2, figsize=(14, 15), sharex=True, layout="constrained")
    for label in ("initial", "previous", "current"):
        for row, key in enumerate(("uc_cash", "uc_book", "uc_reward", "consumer_reward_per_agent", "native_lost_kwh")):
            y = np.array([d[key] for d in daily_data[label]])
            axes[row, 0].plot(y, color=COLORS[label], label=LABELS[label])
            axes[row, 1].plot(np.cumsum(y), color=COLORS[label])
    for row, title in enumerate(("UC 现金 / AUD", "UC 经营收益 / AUD", "UC 奖励 / 元等值",
                                "Consumer 平均奖励 / 元等值/户", "原生失供 / kWh")):
        axes[row, 0].set_title("每日" + title)
        axes[row, 1].set_title("累计" + title)
    axes[0, 0].legend(fontsize=9)
    axes[-1, 0].set_xlabel("123 个验证日：7 月、10 月、1 月、4 月")
    axes[-1, 1].set_xlabel("123 个验证日：累计和，不跨月份计算折扣")
    save(fig, target, "03-profit-and-shortfall", "每日与累计 UC / Consumer 收益及失供：Consumer 奖励 = 负总成本", figures)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for ax, key, title in zip(axes.flat, ("rmse", "rmse_over_target_std", "explained_variance", "target_std"),
                              ("原单位 RMSE（同 γ）" if same_gamma else "原单位 RMSE（折扣目标不同）",
                               "RMSE / 目标标准差", "解释方差 EV（越高越好）", "MC 目标标准差")):
        ax.bar(range(len(specs)), [critic[k][key] for k in specs], color=[COLORS[k] for k in specs])
        ax.set(xticks=range(len(specs)), xticklabels=["旧初始", "新初始", f"旧 {old_rounds} 轮", "新 3 轮", f"新 {rounds} 轮"], title=title)
    save(fig, target, "04-critic-errors", "Critic：同时检查误差、解释方差和目标尺度，不能只看 loss 下降", figures)

    fig, axes = plt.subplots(4, 2, figsize=(15, 13), layout="constrained")
    for row, seed in enumerate(SEEDS):
        for col, label in enumerate(("previous", "current")):
            rows = data[label][seed]
            y = discounted_returns(rows, critic[label]["gamma"])
            axes[row, col].plot(np.arange(len(rows)) / 48 + 1, y, label="完整月 MC", lw=.8, color="#bb8336")
            axes[row, col].plot(np.arange(len(rows)) / 48 + 1, values(rows, "uc_value"), label="Critic V", lw=.9, color=COLORS[label])
            axes[row, col].set(title=f"{rows[0]['timestamp'][:7]} · {LABELS[label]}", ylabel="奖励单位", xlabel="月内日期")
    axes[0, 0].legend()
    save(fig, target, "05-critic-traces", "逐时段预测与完整月 MC 目标；同 γ=0.99" if same_gamma
         else "逐时段预测与各自折扣 MC 目标；折扣不同", figures)

    monthly = []
    for seed in SEEDS:
        fig, axes = plt.subplots(6, 1, figsize=(15, 14), sharex=True, layout="constrained")
        for label in ("previous", "current"):
            rows = data[label][seed]
            monthly.append({"policy": label, "month": rows[0]["timestamp"][:7], **summarize(rows, system, inventory)})
            for ax, key in zip(axes, ("der_to_ess_kw", "grid_to_ess_kw", "ess_discharge_kw", "soc_kwh", "grid_kw", "native_lost_kw")):
                ax.plot(np.arange(len(rows)) / 48 + 1, values(rows, key), lw=.65, color=COLORS[label], label=LABELS[label], alpha=.8)
        for ax, title in zip(axes, ("DER 充电 / kW", "电网充电 / kW", "放电 / kW", "储能量 / kWh", "实际购电 / kW", "原生失供 / kW")):
            ax.set_ylabel(title)
        axes[0].legend(fontsize=9)
        axes[-1].set_xlabel("月内日期；保留每个半小时时段")
        month = data["current"][seed][0]["timestamp"][:7]
        save(fig, target, "month-" + month, f"{month} 完整月：{old_rounds}/{rounds} 轮逐时段对比", figures)

    date = "2011-07-15"
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), layout="constrained")
    for label in ("previous", "current"):
        rows = [r for r in combined[label] if r["timestamp"].startswith(date)]
        assert len(rows) == 48
        write_csv(target / f"day-{date}-{label}.csv", rows)
        for ax, key in zip(axes.flat, ("soc_kwh", "der_to_ess_kw", "grid_to_ess_kw", "ess_discharge_kw", "grid_kw", "native_lost_kw")):
            if key == "soc_kwh":
                ax.plot(np.arange(49) / 2, np.r_[rows[0]["soc_before_kwh"], values(rows, key)],
                        color=COLORS[label], label=LABELS[label])
            else:
                ax.step(np.arange(49) / 2, np.r_[values(rows, key), rows[-1][key]], where="post",
                        color=COLORS[label], label=LABELS[label])
    for ax, title in zip(axes.flat, ("储能量 / kWh", "DER 充电 / kW", "电网充电 / kW", "实际放电 / kW", "实际购电 / kW", "原生失供 / kW")):
        ax.set(title=title, xlabel="当地小时", xlim=(0, 24))
    axes[0, 0].legend(fontsize=9)
    save(fig, target, "10-matched-day", "2011-07-15：沿用此前审查日期，不按新结果挑选", figures)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), layout="constrained")
    soc_rows = []
    for label in ("previous", "current"):
        rows = combined[label]
        fraction = (values(rows, "soc_before_kwh") - system["ess_energy_min_kwh"]) / (
            system["ess_energy_capacity_kwh"] - system["ess_energy_min_kwh"])
        errors = values(rows, "uc_value") - discounted_returns(rows, critic[label]["gamma"])
        bins = (0, .01, .1, .3, .6, 1.00001)
        for i, (lo, hi) in enumerate(pairwise(bins)):
            selected = (fraction >= lo) & (fraction < hi)
            soc_rows.append({"policy": label, "bin": i, "count": int(selected.sum()),
                             "rmse": float(np.sqrt(np.mean(errors[selected] ** 2))) if selected.any() else None})
        part = soc_rows[-5:]
        axes[0].plot(range(5), [r["count"] / len(rows) * 100 for r in part], 'o-', color=COLORS[label], label=LABELS[label])
        axes[1].plot(range(5), [r["rmse"] for r in part], 'o-', color=COLORS[label])
    for ax in axes:
        ax.set(xticks=range(5), xticklabels=("0–1%", "1–10%", "10–30%", "30–60%", "60–100%"), xlabel="决策前 SOC 区间")
    axes[0].set_ylabel("状态占比 / %")
    axes[1].set_ylabel("各自目标下的 Critic RMSE")
    axes[0].legend()
    save(fig, target, "11-soc-occupancy-and-errors", "检查长期空电状态及对应 Critic 误差", figures)
    inputs.extend(training_loss_comparison(target, output, previous,
                                          {"current": report, "previous": old_report}, figures))
    mode_learning, mode_inputs, mode_body = add_mode_learning(target, output, previous, figures)
    inputs.extend(mode_inputs)

    metrics = ("der_to_ess_kwh", "grid_to_ess_kwh", "ess_discharge_kwh", "soc_mean_kwh", "near_empty_fraction",
               "uc_cash_per_day", "uc_book_per_day", "native_lost_kwh", "consumer_reward_per_day")
    differences = {base: {k: {"difference": summaries["current"][k] - summaries[base][k],
                             "relative_percent": 100 * (summaries["current"][k] / summaries[base][k] - 1)
                             if summaries[base][k] else None} for k in metrics}
                   for base in ("initial", "previous")}
    payload = {"status": "completed", "summaries": summaries, "critic": critic, "consumer_critic": consumer_critic,
               "fixed_state_modes": mode_metrics,
               "deterministic": deterministic, "differences": differences, "monthly": monthly,
               "soc_stratified_errors": soc_rows,
               "initial_physical_replay_max_error": initial_error if same_inputs else None,
               "initial_trajectory_max_difference": initial_error, "same_initial_input_transform": same_inputs,
               "mode_learning": mode_learning,
               "figures": figures, "previous": str(previous), "current": str(output),
                "same_rounds_different_sample_budget": rounds == old_rounds and not equal_budget,
                "outer_rounds": {"current": rounds, "previous": old_rounds},
               "same_effective_sample_and_optimizer_budget": equal_budget,
               "same_sample_and_optimizer_budget": equal_budget and not report.get("recovery") and not old_report.get("recovery"),
               "previous_recovery": old_report.get("recovery"),
               "recovery": report.get("recovery"), "same_gamma": same_gamma,
               "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
    (target / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_csv(target / "daily.csv", [{"policy": k, **row} for k, days in daily_data.items() for row in days])
    metric_names = ("DER 充电 / kWh", "电网充电 / kWh", "实际放电 / kWh", "平均储能量 / kWh", "近空电比例",
                    "现金 / AUD/日", "经营收益 / AUD/日", "原生失供 / kWh", "Consumer 合计奖励 / 元等值/日")
    body = f'<h1>储能模式与经济调度：{rounds} 轮对比此前 {old_rounds} 轮</h1>'
    if equal_budget:
        body += (f'<p>同一 seed 42、16 户、训练年和四个验证月。上一版为 {old_report["logged_batches"]} 个学习批，'
                 f'本次为 {report["logged_batches"]} 个学习批；γ=0.99、λ=0.97、奖励及物理配置一致。'
                 '两版有效 10 轮轨迹均为 15,360 个 PPO 分支时段、7,680 个主轨迹时段，每个 Critic 对应 1,920 次更新。'
                 f'契约变化项：{html.escape(", ".join(protocol["changes_from_previous_ten_rounds"]))}。测试集未加载。</p>')
        if not same_inputs:
            body += ('<p>本次 UC Actor 改用中心化 SOC，并增加三个零初始化的 SOC 直连权重；'
                     '采样起点、384 步合批、Critic 后重算 GAE、联合 PPO 裁剪和总训练预算保持。'
                     '输入转换会改变初始动作，两版初始策略分别列出；不把同一 seed 称为完全相同初始策略。</p>')
    else:
        body += ('<p>同一 seed 42、16 户、训练年和四个验证月。各版本轮次和训练预算以实际记录为准，'
                 f'PPO 时段分别为 {old_report["ppo_sampled_periods"]:,} 和 {report["ppo_sampled_periods"]:,}，'
                 '不是等预算消融。详细学习契约保存在各自 report.json。</p>')
    if report.get("recovery"):
        body += ('<p><b>故障恢复开销：</b>从第 7 轮完整 checkpoint 恢复，原第 8 轮的 384 步学习批和 224 步未完成采样单独归档。'
                 '本报告曲线使用去除故障重复段的完整 10 轮路径；实际共采样 15,968 个训练时段，完成学习的时段为 15,744，'
                 '每个 Critic 实际执行 1,968 次更新。有效路径预算相同，总计算开销多于上一版。'
                 '恢复时网络、Adam 和随机状态逐项核对，checkpoint 采样精确重放通过。详细账本：../recovery/from-round-0007/ledger.json。</p>')
    if old_report.get("recovery"):
        body += ('<p>上一版曾故障恢复，其有效路径预算用于本对照；故障导致的额外学习和未完成采样另见上一版 recovery 账本。'
                 '有效预算相同不表示两次实际总计算开销相同。</p>')
    if report.get("continuation"):
        continued = report["continuation"]
        body += (f'<p>本次从第 {continued["source_round"]} 轮原生 checkpoint 继续，新增 '
                 f'{rounds - continued["source_round"]} 轮。前缀结果为此前实测，未重复训练。'
                 '第 11 轮起候选筛选使用冬、夏两个完整月，完整四个月报告验证只在第 30、72 轮执行；'
                 '网络、奖励、采样起点和每轮学习预算保持。减少筛选月份会改变接受统计口径，不能作为同筛选协议的消融。</p>')
    body += (f'<p><b>最终检查：</b>随机执行充电覆盖 {summaries["current"]["charging_days"]} / 123 天；'
             f'确定性执行闲置占比 {deterministic["current"]["mode_fraction"]["0"]*100:.4f}%，'
             f'实际充电 {deterministic["current"]["charge_kwh"]:.6f} kWh、放电 {deterministic["current"]["ess_discharge_kwh"]:.6f} kWh。'
             f'UC Critic RMSE 为 {critic["current"]["rmse"]:.4f}，解释方差为 {critic["current"]["explained_variance"]:.6f}。'
             '两种执行方式必须分别评价；持续随机充电并不能证明策略已学会按状态调度。</p>')
    body += table(["指标", "旧初始", "新初始", f"旧 {old_rounds} 轮", "新 3 轮", f"新 {rounds} 轮", "相对旧版变化"],
                  [[name, *(f'{summaries[label][key]:.6f}' for label in specs),
                    f'{differences["previous"][key]["difference"]:+.6f}'] for name, key in zip(metric_names, metrics)])
    body += '<p>近空电定义为可用容量最低 1%；所有电量与账本收益按完整月份未折扣累计。各策略原生请求可能变化，失供同时结合原生请求量和供电率判断。共同 γ=0.99 的策略回报另存 summary.json，只用于统一目标比较，不替换任一版本的训练折扣。</p>'
    body += '<h2>模式概率与确定性执行</h2>'
    body += table(["策略", "相同可行状态下闲置/充电/放电平均概率", "条件分类熵", "实际确定性充电 / kWh"],
                  [[LABELS[k], ' / '.join(f'{x*100:.4f}%' for x in mode_metrics[k]["mean"]),
                    f'{mode_metrics[k]["entropy"]:.6f}', f'{deterministic[k]["charge_kwh"]:.6f}'] for k in ("previous", "current")])
    body += '<p>固定状态曲线使用此前审查保存的同一组观测、可行边界和基础额度，隔离网络变化；实际模式频率与确定性完整月执行另列。概率偏离 1/3 本身不能证明经济策略改善，须结合收益、失供与持续充电。</p>'
    body += table(["策略", "跨状态概率标准差 / 百分点（闲/充/放）", "固定状态 argmax 次数（闲/充/放）", "随机执行频率（闲/充/放）"],
                  [[LABELS[k], ' / '.join(f'{x*100:.4f}' for x in mode_metrics[k]["std"]),
                    ' / '.join(str(x) for x in mode_metrics[k]["argmax_counts"]),
                    ' / '.join(f'{summaries[k]["mode_fraction"][str(m)]*100:.4f}%' for m in range(3))] for k in specs])
    body += '<h2>Critic 的误差与尺度</h2>'
    body += table(["策略", "γ", "原单位 RMSE", "RMSE/目标标准差", "解释方差", "目标均值", "预测均值"],
                  [[LABELS[k], *(f'{critic[k][v]:.6f}' for v in ("gamma", "rmse", "rmse_over_target_std", "explained_variance", "return_mean", "prediction_mean"))] for k in specs])
    body += ('<p>两版 Critic 均按 γ=0.99 的完整月 MC 评价，折扣口径相同。' if same_gamma else
             '<p>两版 Critic 的折扣目标不同，原始 RMSE 变化包含目标尺度变化。')
    body += ('各策略的访问状态与实际回报仍可能不同，结合归一化误差、解释方差和预测/目标曲线判断。'
             '每个状态只有一条冻结随机 MC 轨迹，报告的是该轨迹上的误差。每日 Consumer 奖励取总成本的负值，'
             '包含账单和模型内其他成本；不称为纯现金收入。UC 现金、经营收益、奖励分别列出。</p>')
    body += '<h3>16 个消费者 Critic</h3>'
    body += table(["策略", "平均原单位 RMSE", "平均 RMSE/目标标准差", "平均解释方差", "平均目标均值", "平均目标标准差"],
                  [[LABELS[k], *(f'{consumer_critic[k]["agent_mean"][v]:.6f}' for v in
                    ("rmse", "rmse_over_target_std", "explained_variance", "return_mean", "target_std"))] for k in specs])
    body += '<p>各列先逐消费者计算，再对 16 人取算术平均，不是把不同尺度的消费者混合成一个样本池。目标标准差由已保存的 MSE、bias 与 EV 恒等式还原；逐消费者数值存于 summary.json。</p>'
    body += '<h3>UC 候选接受记录</h3>'
    body += table(["轮次", "接受", "平均配对增益", "筛选下界", "保留 Actor 参数 RMS 变化"],
                  [[r["round"]["outer_round"], r["round"]["candidate_accepted"],
                    f'{r["round"]["candidates"][1]["paired_uc_gain"]["mean"]:.6f}',
                    f'{r["round"]["candidates"][1]["paired_uc_gain"]["lower"]:.6f}',
                    f'{r["round"]["policy_change"]["per_agent"]["uc"]:.8f}'] for r in report["rounds"]])
    body += '<p>筛选比较同轮配对策略的完整月折扣回报；接受门槛为平均增益减 1.96 倍标准误大于 0。现金、经营收益与原生失供使用全月未折扣账本，需分别判断。</p>'
    body += mode_body
    body += '<h2>完整曲线</h2>'
    for name, title in figures:
        body += f'<figure><h3>{html.escape(title)}</h3><img src="{name}.png"><figcaption><a href="{name}.svg">SVG</a></figcaption></figure>'
    body += '<p>固定日图的 SOC 在实际半小时时点绘制（含 00:00 初值和 24:00 终值）；功率为对应半小时区间值。完整月图的储能量记录为各时段结束值。</p>'
    if (output / "comparison/report.html").exists():
        body += '<p><a href="../comparison/report.html">旧 72 轮历史补充</a></p>'
    body += '<p><a href="summary.json">全部指标与源文件哈希</a> · <a href="daily.csv">逐日数据</a> · <a href="training-losses.csv">逐户训练 loss</a></p>'
    css = 'body{max-width:1380px;margin:32px auto;padding:0 24px;color:#183348;font:17px/1.8 "Microsoft YaHei",sans-serif}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #d2dce5;padding:9px;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#edf4f6}img{width:100%}figure{margin:36px 0}a{color:#087f8c}'
    (target / "report.html").write_text(f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{rounds} 轮储能对比</title><style>{css}</style>{body}</html>', encoding="utf-8")
    print(json.dumps({"report": str(target / "report.html"), "figures": len(figures)}), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
