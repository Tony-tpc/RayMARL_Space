"""SOC response and measured mode diagnostics for the storage comparison report."""

import numpy as np

from experiments.sn_energy_market.plot_batch_losses import load_batches
from experiments.sn_energy_market.plot_hybrid_storage import (
    plt,
    read_json,
    save,
    write_csv,
)

NAMES = ("闲置", "充电", "放电")


def add_mode_learning(target, output, previous, figures):
    path = output / "soc-mode-audit.json"
    if not path.exists():
        return {}, [], ""
    audit = read_json(path)
    inputs = [path, output / "batch_losses.jsonl", previous / "batch_losses.jsonl"]
    policies = audit["policies"]
    labels = {"previous_initial": "旧初始", "previous": "旧 384 步第 10 轮", "initial": "新初始",
              "round3": "新第 3 轮", "current": f"新第 {audit.get('outer_rounds', 10)} 轮"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, layout="constrained")
    for label, style in (("previous", "--"), ("initial", ":"), ("current", "-")):
        stats = policies[label]
        for mode in range(3):
            axes[0, mode].plot(stats["soc_fractions"], np.array(stats["probability_mean"])[:, mode] * 100,
                               style, label=labels[label])
            axes[1, mode].plot(stats["soc_fractions"], np.array(stats["argmax_counts"])[:, mode] / stats["states"] * 100,
                               style, label=labels[label])
            axes[0, mode].set_title(NAMES[mode] + "概率 / %")
            axes[1, mode].set(ylabel="argmax 占比 / %", xlabel="仅改变 SOC 比例（5%–95%）", ylim=(-2, 102))
    axes[0, 0].legend(fontsize=9)
    save(fig, target, "14-soc-mode-response", "固定其他观测、基础额度及可行掩码：SOC 是否改变模式排序（排除边界强制切换）", figures)

    batches = load_batches(output / "batch_losses.jsonl")
    rows, coverage = [], []
    for batch in batches:
        diagnostic = batch["modules"]["uc"].get("mode_diagnostics")
        if not diagnostic:
            continue
        assert diagnostic["unique_samples"] == batch["samples"]
        assert sum(r["count"] for r in diagnostic["soc_advantages"]) == batch["samples"]
        context = {"batch": batch["batch_id"], "ppo_periods": batch["sampled_periods_lifetime"],
                   "retained": batch["retained_updates"]["uc"]["actor"]}
        coverage.extend({**context, **r} for r in diagnostic["soc_advantages"])
        for entry in diagnostic["minibatches"]:
            assert entry["adam_step_after"] == entry["adam_step_before"] + 1
            rows.append({**context, **entry})
        assert sum(r["samples"] for r in diagnostic["minibatches"]) == 2 * batch["samples"]
    assert rows, "New SOC run must contain actual minibatch mode diagnostics"
    assert len(rows) == 24 * len({r["batch"] for r in rows})
    csv_rows = []
    for index, row in enumerate(rows, 1):
        csv_rows.append({"optimizer_execution": index, **{k: row[k] for k in
                         ("batch", "ppo_periods", "retained", "epoch", "minibatch", "samples",
                          "adam_step_before", "adam_step_after", "global_clip_multiplier",
                          "mode_blocked_by_joint_fraction", "removed_mode_gradient_mass_fraction")},
            **{f"{part}_{metric}": value for part, metrics in row["ratios"].items() for metric, value in metrics.items()},
            **{f"{part}_{metric}": value for metric in ("range_clip_fraction", "flat_clip_fraction")
               for part, value in row[metric].items()},
            **{f"{part}_{stage}": value for stage in ("gradient_norm_before", "gradient_norm_after", "parameter_delta_norm")
               for part, value in row[stage].items()},
            **{f"mode_soc_weight_{mode}": row["soc_weights_after"][mode] for mode in range(3)}})
    write_csv(target / "mode-minibatches.csv", csv_rows)
    write_csv(target / "mode-soc-advantages.csv", [{k: v for k, v in r.items() if k not in ("advantage", "raw_advantage")}
        | {f"{kind}_{key}": (r[kind] or {}).get(key) for kind in ("advantage", "raw_advantage")
           for key in ("mean", "std", "min", "max", "p05", "p50", "p95")} for r in coverage])
    x = np.arange(1, len(rows) + 1)
    rejected = np.array([not r["retained"] for r in rows])
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True, layout="constrained")
    for key, title in (("mode", "模式头"), ("mode_soc", "SOC 直连")):
        for stage, style in (("before", "--"), ("after", "-")):
            axes[0].plot(x, [r[f"gradient_norm_{stage}"][key] for r in rows], style, label=f"{title} {stage}")
    for key in ("joint", "unclipped", "separate_mode"):
        axes[1].plot(x, [r["policy_mode_gradient_norm"][key] for r in rows], label=key)
    for key in ("mode.weight", "mode.bias", "mode_soc.weight"):
        axes[2].plot(x, [r["parameter_delta_norm"][key] for r in rows], label=key)
    for mode in range(3):
        axes[3].plot(x, [r["soc_weights_after"][mode] for r in rows], label=NAMES[mode])
    for ax, title in zip(axes, ("实际梯度范数：全局裁剪前后", "策略模式梯度范数（分别裁剪仅是假设诊断）",
                               "实际 Adam 参数变化范数", "SOC 直连权重（含未接受提案）")):
        ax.set_ylabel(title, fontsize=9)
        ax.legend(fontsize=8, ncol=3)
        ax.fill_between(x, 0, 1, where=rejected, transform=ax.get_xaxis_transform(), color="#c78282", alpha=.12)
    axes[-1].set_xlabel("实际 UC Actor 优化执行序号；浅红背景为整轮未接受的提案")
    save(fig, target, "15-mode-gradients", "实际模式梯度、全局裁剪、Adam 变化与 SOC 权重；旧版未保存对应逐批指标", figures)

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True, layout="constrained")
    for key in ("joint", "mode", "continuous"):
        mean = np.array([r["ratios"][key]["mean"] for r in rows])
        axes[0].plot(x, mean, label=key)
        axes[0].fill_between(x, [r["ratios"][key]["p05"] for r in rows],
                             [r["ratios"][key]["p95"] for r in rows], alpha=.12)
        for ax, metric in zip(axes[1:3], ("range_clip_fraction", "flat_clip_fraction")):
            ax.plot(x, [100 * r[metric][key] for r in rows], label=key)
    for key, label in (("mode_blocked_by_joint_fraction", "模式未平坦但联合平坦 / %"),
                       ("removed_mode_gradient_mass_fraction", "被移除样本梯度范数质量 / %")):
        axes[3].plot(x, [r[key] * 100 for r in rows], label=label)
    for ax, title in zip(axes, ("概率比均值及各 minibatch 5%–95%", "超出裁剪区间 / %", "实际平坦分支 / %", "联合裁剪的模式影响")):
        ax.set_ylabel(title, fontsize=9)
        ax.legend(fontsize=8, ncol=3)
    axes[-1].set_xlabel("实际 UC Actor 优化执行序号；重复 epoch 样本按实际访问计数")
    save(fig, target, "16-mode-clipping", "联合概率比、真实平坦分支与模式信号损失；不是仅用超区间比例判断", figures)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    bins = sorted({(r["soc_low"], r["soc_high"]) for r in coverage})
    soc_summary = []
    for mode in range(3):
        counts, means, raw_means, contributions = [], [], [], []
        for lo, hi in bins:
            selected = [r for r in coverage if r["mode"] == mode and r["soc_low"] == lo and r["count"]]
            count = sum(r["count"] for r in selected)
            averages = {key: sum(r["count"] * r[key]["mean"] for r in selected) / count if count else None
                        for key in ("advantage", "raw_advantage")}
            contribution = sum(r["samples"] * g["mode_bias_ascent"][mode] for r in rows
                               for g in r["mode_bias_ascent_by_soc"] if g["soc_low"] == lo) / sum(r["samples"] for r in rows)
            soc_summary.append({"soc_low": lo, "soc_high": hi, "mode": mode, "unique_count": count,
                                **averages, "bias_ascent_contribution": contribution})
            counts.append(count)
            means.append(averages["advantage"])
            raw_means.append(averages["raw_advantage"])
            contributions.append(contribution)
        for ax, ys in zip(axes.flat, (counts, raw_means, means, contributions)):
            ax.plot(range(len(bins)), ys, "o-", label=NAMES[mode])
    for ax, title in zip(axes.flat, ("唯一提案样本数（不重复 epoch）", "重算后的原始 GAE 均值", "整批标准化优势均值", "实际联合裁剪后的模式偏置上升贡献")):
        ax.set(xticks=range(len(bins)), xticklabels=[f"{lo:.0%}–{hi:.0%}" for lo, hi in bins], title=title)
        ax.legend(fontsize=9)
        ax.axhline(0, color="gray", lw=.5)
    save(fig, target, "17-soc-advantages", "各 SOC 区间的样本覆盖、优势与模式梯度方向；未改变采样起点", figures)
    weights = np.array([r["samples"] for r in rows])
    summary = {"soc_response": policies, "optimizer_executions": len(rows),
               "unique_proposal_samples": sum(r["count"] for r in coverage), "sample_visits": int(weights.sum()),
               "soc_advantages": soc_summary, "previous_minibatch_diagnostics_available": any(
                   b["modules"]["uc"].get("mode_diagnostics") for b in load_batches(previous / "batch_losses.jsonl")),
               **{key: float(np.average([r[key] for r in rows], weights=weights)) for key in
                  ("mode_blocked_by_joint_fraction", "removed_mode_gradient_mass_fraction")},
               **{key: {part: float(np.average([r[key][part] for r in rows], weights=weights))
                        for part in ("joint", "mode", "continuous")} for key in ("range_clip_fraction", "flat_clip_fraction")}}
    body = ('<h2>SOC 模式切换与实际训练诊断</h2><p>固定其他观测、基础额度和原可行掩码，仅扫描内部 SOC 5%–95%；'
            '该探针测量 Actor 对 SOC 的条件响应，不作为物理反事实收益。排除空电/满电的强制掩码切换。'
            f'旧第 10 轮发生模式排序切换的状态比例为 {policies["previous"]["switch_fraction"]:.2%}，'
            f'新第 {audit.get("outer_rounds", 10)} 轮为 {policies["current"]["switch_fraction"]:.2%}。</p>'
            '<p>新运行记录每次实际 UC Actor 更新的联合/模式/连续概率比、范围裁剪与真实平坦比例、'
            '全局裁剪前后梯度和 Adam 参数变化，包含最终被回滚的提案。旧运行没有相同的逐 minibatch 历史诊断，'
            '不补造旧值；模式单独裁剪仅用于假设梯度诊断，实际优化仍使用联合裁剪。</p>'
            '<p><a href="mode-minibatches.csv">逐 minibatch 指标</a> · <a href="mode-soc-advantages.csv">SOC 分组优势</a></p>')
    return summary, inputs, body
