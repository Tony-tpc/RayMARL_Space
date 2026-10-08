"""完整使用流程：环境 -> 配置 -> 训练 -> 评估 -> 保存 -> 恢复 -> 独立推理。"""

import argparse
import json
import os
import secrets
import sys
from pathlib import Path
from uuid import uuid4

import gymnasium as gym
import numpy as np
import ray
import torch
from energy_env import MultiEnergyEnv, SingleEnergyEnv
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.train._internal.storage import StorageContext

ROOT = Path(__file__).resolve().parents[2]


def build_single(args, output):
    """单智能体使用原生 PPO；若带历史，则显式展平为普通 MLP 的输入。"""
    config = (
        PPOConfig()
        .environment(
            SingleEnergyEnv,
            env_config={"horizon": args.horizon, "history": args.history},
            normalize_actions=False,
            clip_actions=True,
        )
        .framework("torch")
        .env_runners(
            num_env_runners=0,
            rollout_fragment_length=min(16, args.horizon),
            env_to_module_connector=lambda env, spaces=None, device=None: FlattenObservations(),
        )
        .learners(num_learners=0, num_gpus_per_learner=0)
        .rl_module(model_config={"fcnet_hiddens": [32, 32]})
        .training(train_batch_size_per_learner=args.batch, minibatch_size=args.batch, num_epochs=1)
        .evaluation(evaluation_duration=1, evaluation_duration_unit="episodes", evaluation_num_env_runners=0)
        .debugging(seed=args.seed)
    )
    # 当前 RLlib 2.59 使用 StorageContext 指定输出位置。
    # 每次构建使用独立目录，恢复不会覆盖之前的运行。
    storage = StorageContext(
        storage_path=str(output.resolve()), experiment_dir_name="single-ppo", trial_dir_name=uuid4().hex[:12]
    )
    return PPO(config=config, storage=storage)


def build_multi(args, output, name):
    """MARL 外层流程相同，配置选择算法并自动为各住户建立独立策略。"""
    from raymarl import IPPOConfig, ISACConfig, MAACConfig, MADDPGConfig, MAPPOConfig, MASACConfig

    configs = {
        "maac": MAACConfig,
        "masac": MASACConfig,
        "maddpg": MADDPGConfig,
        "isac": ISACConfig,
        "mappo": MAPPOConfig,
        "ippo": IPPOConfig,
    }
    config = (
        configs[name]()
        .environment(
            MultiEnergyEnv,
            env_config={
                "horizon": args.horizon,
                "history": args.history,
                "households": 2,
                "reward_mode": args.reward,
            },
        )
        .env_runners(num_env_runners=0, rollout_fragment_length=min(16, args.horizon))
        .learners(num_learners=0, num_gpus_per_learner=0)
        .rl_module(model_config={"hidden_dim": 16})
        .training(storage_path=str(output), checkpoint_interval=0)
        .evaluation(evaluation_duration=1, evaluation_duration_unit="episodes", evaluation_num_env_runners=0)
        .debugging(seed=args.seed)
    )
    if name in ("mappo", "ippo"):
        config.training(train_batch_size_per_learner=args.batch, minibatch_size=args.batch, num_epochs=1)
    else:
        # 短示例也要真正发生参数更新，所以 warmup 采用较小的值。
        config.training(
            batch_size=args.batch,
            warmup_steps=min(16, args.horizon),
            update_interval=4,
            replay_capacity=1_000,
        )
        if name == "maac":
            config.training(baseline_samples=2)
    return config.build_algo()


def summary(result):
    runners = result.get("env_runners", {})
    value = runners.get("episode_return_mean")
    return {
        "iteration": int(result["training_iteration"]),
        "env_steps": int(result["num_env_steps_sampled_lifetime"]),
        "episode_return_mean": float(value) if value is not None else None,
        "updates": result.get("num_updates_lifetime"),
    }


def infer_single(module_path, args):
    # 从模型 checkpoint 加载，无需再次构建 PPO Algorithm 或学习器。
    # 这个原生 RLModule checkpoint 包含标准 PPO 模型；不宣称它是 Actor-only 文件。
    module = RLModule.from_checkpoint(module_path)
    env = SingleEnergyEnv({"horizon": args.horizon, "history": args.history})
    obs, _ = env.reset(seed=args.seed)
    total, steps = 0.0, 0
    try:
        while True:
            # 训练时用了 FlattenObservations，推理时必须进行相同的展开。
            flattened = gym.spaces.utils.flatten(env.observation_space, obs).astype(np.float32)
            batch = {Columns.OBS: torch.from_numpy(flattened).unsqueeze(0)}
            with torch.no_grad():
                out = module.forward_inference(batch)
                distribution = module.get_inference_action_dist_cls().from_logits(
                    out[Columns.ACTION_DIST_INPUTS]
                )
                action = distribution.to_deterministic().sample()[0].cpu().numpy()
            # 与训练时 clip_actions=True 的动作边界处理保持一致。
            action = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
            assert env.action_space.contains(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            steps += 1
            if terminated or truncated:
                break
        return {"steps": steps, "return": total}
    finally:
        env.close()


def infer_multi(actor_path, args):
    from raymarl.inference import ActorBundle

    # ActorBundle 只包含各住户的局部 Actor，不创建 Critic、优化器或学习器。
    actors = ActorBundle(actor_path, device="cpu")
    env = MultiEnergyEnv(
        {"horizon": args.horizon, "history": args.history, "households": 2, "reward_mode": args.reward}
    )
    obs, _ = env.reset(seed=args.seed)
    totals = {aid: 0.0 for aid in env.possible_agents}
    steps = 0
    try:
        while True:
            actions = actors.actions(obs, deterministic=True)
            assert all(env.get_action_space(aid).contains(action) for aid, action in actions.items())
            obs, rewards, terms, truncs, _ = env.step(actions)
            for aid, reward in rewards.items():
                totals[aid] += reward
            steps += 1
            if terms["__all__"] or truncs["__all__"]:
                break
        return {"steps": steps, "returns": totals}
    finally:
        env.close()


def run_case(args, output, kind, algorithm):
    """两种任务共用这段生命周期，差异集中在配置、环境接口及推理导出。"""

    def build():
        return build_single(args, output) if kind == "single" else build_multi(args, output, algorithm)

    algo = build()
    report = {"mode": kind, "algorithm": algorithm, "history": args.history, "training": []}
    checkpoint = output / f"{kind}-{algorithm}-checkpoint"
    exported = output / ("single-ppo-module" if kind == "single" else f"{algorithm}-actors.pt")
    try:
        # 训练：一个 algo.train() 管理全部智能体，不需要按住户分别启动训练。
        for _ in range(args.iterations):
            row = summary(algo.train())
            report["training"].append(row)
            print(json.dumps({"case": f"{kind}-{algorithm}", **row}, default=str), flush=True)
        evaluation = algo.evaluate()["env_runners"]["episode_return_mean"]
        assert np.isfinite(evaluation)
        report["evaluation_return"] = float(evaluation)
        # 保存：完整算法 checkpoint 可用于恢复训练。
        algo.save_to_path(str(checkpoint))
        if kind == "single":
            algo.get_module().save_to_path(str(exported))
        else:
            algo.export_actors(exported)
    finally:
        algo.stop()

    # 恢复：重新构建相同配置的算法，再加载状态并继续训练一次。
    restored = build()
    try:
        restored.restore_from_path(str(checkpoint))
        report["resumed_training"] = summary(restored.train())
    finally:
        restored.stop()
    # 推理展示使用首次训练结束时导出的文件，与 resumed_training 是两条分支。
    report["inference"] = infer_single(exported, args) if kind == "single" else infer_multi(exported, args)
    assert report["inference"]["steps"] == args.horizon
    report["checkpoint"] = str(checkpoint)
    report["export"] = str(exported)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("single", "marl", "both"), default="both")
    parser.add_argument(
        "--algorithm", choices=("maac", "masac", "maddpg", "isac", "mappo", "ippo", "all"), default="maac"
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--history", type=int, default=0)
    parser.add_argument("--reward", choices=("cooperative", "individual"), default="cooperative")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "usage-example")
    args = parser.parse_args()
    if args.iterations < 1 or args.horizon < 4 or args.batch < 4 or args.history < 0:
        parser.error("iterations >= 1; horizon/batch >= 4; history >= 0")

    # 从源码 checkout 运行时明确加载本项目版本，避免误用 venv 中的旧安装包。
    sys.path.insert(0, str(ROOT / "src"))
    run_id = uuid4().hex[:12]
    output = args.output.resolve() / run_id
    output.mkdir(parents=True)
    token = ROOT / ".ray" / "auth_token"
    token.parent.mkdir(exist_ok=True)
    if not token.exists():
        token.write_text(secrets.token_hex(32), encoding="utf-8")
    os.environ["RAY_AUTH_TOKEN_PATH"] = str(token)
    os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
    torch.set_num_threads(2)
    ray.init(
        num_cpus=2,
        include_dashboard=False,
        _node_ip_address="127.0.0.1",
        _temp_dir=str(ROOT / ".tmp" / "usage-example" / run_id / "ray"),
    )
    reports = []
    try:
        if args.mode in ("single", "both"):
            reports.append(run_case(args, output, "single", "ppo"))
        if args.mode in ("marl", "both"):
            names = (
                ("maac", "masac", "maddpg", "isac", "mappo", "ippo")
                if args.algorithm == "all"
                else (args.algorithm,)
            )
            for name in names:
                reports.append(run_case(args, output, "marl", name))
    finally:
        ray.shutdown()
    path = output / "report.json"
    path.write_text(json.dumps(reports, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"完成：{path}", flush=True)


if __name__ == "__main__":
    main()
