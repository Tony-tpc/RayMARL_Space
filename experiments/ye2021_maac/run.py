"""Auditable single-case runner. Budgets count joint ENV steps, never agent steps."""
import argparse
import copy
import json
from pathlib import Path
import time
import threading
import os
import numpy as np
import torch
from ray.rllib.core.columns import Columns
from raymarl import MAACConfig, MASACConfig, MADDPGConfig, ISACConfig, IPPOConfig, MAPPOConfig
from raymarl.algorithms import JointAlgorithm
from city_proxy import CityLearnProxy, ROOT
from runtime import initialize_ray


def factory(name, env_config, seed, gpu, parameters=None):
    types = {"maac": MAACConfig, "masac": MASACConfig, "maddpg": MADDPGConfig,
             "isac": ISACConfig, "ippo": IPPOConfig, "mappo": MAPPOConfig}
    environment=CityLearnProxy
    if name=='cen_sac':
        from central_reference import CentralReference
        environment=CentralReference
    if name.startswith('maac_'):name='maac'
    cfg = types['isac' if name=='cen_sac' else name]().environment(environment, env_config=env_config).debugging(seed=seed)
    cfg.learners(num_learners=0, num_gpus_per_learner=gpu)
    if name in ("ippo", "mappo"):
        cfg.multi_agent(policies=set(env_config["buildings"]),
            policy_mapping_fn=lambda agent, *args, **kw: agent)
        cfg.env_runners(num_env_runners=0, rollout_fragment_length=128, batch_mode="complete_episodes")
        cfg.rl_module(model_config={"hidden_dim": 64})
        cfg.training(train_batch_size_per_learner=512, minibatch_size=128, num_epochs=5,
                     gamma=.99, lambda_=.95, clip_param=.2, entropy_coeff=.01, lr=3e-4)
    else:
        cfg.training(storage_path=str(ROOT/"outputs"/"storage"/f"{name}-{seed}"))
    if parameters:
        if name in ('ippo','mappo'):
            cfg.training(lr=parameters['actor_lr'],entropy_coeff=parameters.get('entropy_coeff',.01))
        else:cfg.training(**parameters)
    return cfg.build_algo()


def actions(algo, env, observation, deterministic):
    if isinstance(algo, JointAlgorithm):
        return algo.actions(observation, deterministic=deterministic)
    state = None
    if isinstance(algo.config, MAPPOConfig):
        import gymnasium as gym
        state = np.concatenate([gym.spaces.utils.flatten(env.observation_spaces[a], observation[a])
                                for a in env.agents]).astype(np.float32)
    result = {}
    for aid in env.agents:
        module = algo.get_module(aid)
        value = {"local": observation[aid], "critic": state} if state is not None else observation[aid]
        device = next(module.parameters()).device
        def batch(v):
            if isinstance(v, dict):
                return {k: batch(x) for k, x in v.items()}
            return torch.as_tensor(v, device=device).unsqueeze(0)
        with torch.no_grad():
            output = module.forward_inference({Columns.OBS: batch(value)})
            distribution = module.get_inference_action_dist_cls().from_logits(output[Columns.ACTION_DIST_INPUTS])
            selected = distribution.to_deterministic().sample() if deterministic else distribution.sample()
        def unbatch(v):
            if isinstance(v, dict):
                return {k: unbatch(x) for k, x in v.items()}
            x = v[0].cpu().numpy()
            return x.item() if x.ndim == 0 else x
        result[aid] = unbatch(selected)
    return result


def evaluate(algo, config, deterministic=False, episodes=8, action_seed=100):
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    numpy_rng = np.random.get_state()
    torch.manual_seed(action_seed)
    if config.get('central_reference'):
        from central_reference import CentralReference
        env=CentralReference(config)
    else:env = CityLearnProxy(config)
    try:
        result = []
        for episode in range(episodes):
            obs, _ = env.reset(seed=action_seed)
            returns = {a: 0. for a in env.agents}
            while True:
                obs, reward, term, trunc, _ = env.step(actions(algo, env, obs, deterministic))
                for a, r in reward.items():
                    returns[a] += float(r)
                if term["__all__"] or trunc["__all__"]:
                    break
            result.append({"returns": dict(env.native_returns) if hasattr(env,'native_returns') else returns,
                           "kpis": env.evaluate()})
        return result
    finally:
        env.close()
        torch.set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        np.random.set_state(numpy_rng)


def run_case(algorithm, track, seed, budget, stage="smoke", gpu=1, episode_steps=168, evaluate_test=True,
             parameters=None,case_label=None):
    if algorithm == "maddpg" and track != "continuous":
        raise ValueError("Standard MADDPG is only eligible for the continuous track")
    buildings = ["Building_1", "Building_2", "Building_4"]
    if stage in ("formal","tune","ablation"):
        buildings += ["Building_3", "Building_5"]
    config = {"buildings": buildings, "seed": seed, "track": track, "split": "train",
              "episode_steps": episode_steps}
    config['central_reference']=algorithm=='cen_sac'
    parameters=dict(parameters or {})
    if algorithm=='maac_uniform':parameters['critic_mode']='uniform'
    if algorithm=='maac_no_baseline':parameters['counterfactual']=False
    if algorithm=='maac_no_history':config['history_enabled']=False
    out = ROOT/"outputs"/stage/(case_label or f"{algorithm}-{track}-seed{seed}")
    out.mkdir(parents=True, exist_ok=True)
    algo = factory(algorithm, config, seed, gpu,parameters)
    import psutil
    owner=psutil.Process(os.getpid());stop_sampling=threading.Event();peak_rss=[0]
    def sample_memory():
        while not stop_sampling.wait(.5):
            try:
                peak_rss[0]=max(peak_rss[0],sum(p.memory_info().rss for p in [owner,*owner.children(recursive=True)]
                                               if p.is_running()))
            except psutil.Error:pass
    monitor=threading.Thread(target=sample_memory,daemon=True);monitor.start()
    if gpu:torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    records = []
    best_score = -float("inf")
    next_validation = 5000
    try:
        steps = 0
        while steps < budget:
            r = algo.train()
            steps = int(r["num_env_steps_sampled_lifetime"])
            record = {"env_steps": steps, "return_mean": r["env_runners"]["episode_return_mean"],
                      "elapsed_seconds": time.perf_counter()-started}
            records.append(record)
            with (out/"training.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record)+"\n")
            print(algorithm, track, seed, steps, round(record["return_mean"], 4), flush=True)
            if steps >= next_validation and stage != "smoke":
                validation = evaluate(algo, {**config, "split": "validation", "weeks": [36,38,40,42]},
                                      episodes=4, action_seed=100)
                score = float(np.mean([v for r in validation for v in r["returns"].values()]))
                with (out/"validation.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"env_steps":steps,"mean_personal_return":score})+"\n")
                if score > best_score:
                    best_score = score
                    algo.save_to_path(str(out/"best-checkpoint"))
                next_validation += 5000
        checkpoint = algo.save_to_path(str(out/"checkpoint"))
        if best_score != -float("inf"):
            algo.restore_from_path(str(out/"best-checkpoint"))
        if isinstance(algo, JointAlgorithm):
            algo.export_actors(str(out/"actors.pt"))
        tests = []
        if evaluate_test:
            for action_seed in (100, 101, 102):
                tests.extend(evaluate(algo, {**config, "split": "test"}, episodes=8 if stage in ("formal","ablation") else 1,
                                      action_seed=action_seed))
        report = {"algorithm": algorithm, "track": track, "seed": seed, "stage": stage,
            "requested_joint_env_steps": budget, "actual_joint_env_steps": steps,
            "episode_time_rows": episode_steps,
            "note": "CityLearn 168 time rows yield 167 transitions; complete-episode budgets may overshoot",
            "elapsed_seconds": time.perf_counter()-started, "checkpoint": str(checkpoint),
            "training_last": records[-1], "evaluation": tests, "env_config": config}
        report["validation_best_mean_personal_return"] = best_score if best_score != -float("inf") else None
        report['parameters']=parameters
        from suite import code_hash
        report['source_sha256']=code_hash()
        report['peak_process_tree_rss_bytes']=peak_rss[0]
        report['peak_torch_cuda_allocated_bytes']=torch.cuda.max_memory_allocated() if gpu else 0
        report['training_iterations']=len(records)
        report['joint_learner_updates']=algo.joint_learner.steps if isinstance(algo,JointAlgorithm) else None
        (out/"result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        stop_sampling.set();monitor.join(timeout=2)
        algo.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=("maac", "masac", "maddpg", "isac", "mappo", "ippo"), required=True)
    parser.add_argument("--track", choices=("continuous", "hybrid"), default="hybrid")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget-env-steps", type=int, default=1000)
    parser.add_argument("--stage", choices=("smoke", "debug", "formal"), default="smoke")
    parser.add_argument("--episode-steps", type=int, default=168)
    parser.add_argument("--gpus", type=int, choices=(0,1), default=1)
    args = parser.parse_args()
    initialize_ray()
    import ray
    try:
        run_case(args.algorithm,args.track,args.seed,args.budget_env_steps,args.stage,args.gpus,args.episode_steps)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
