"""Compare frozen baseline code against optimized execution on round 24."""
import copy
import hashlib
import importlib.util
import json
import pickle
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'RLAlgo/src'), str(ROOT / 'environments/sn_energy_market/src')]
import numpy as np
import torch
from ray.rllib.core.learner.learner_group import LearnerGroup
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from raymarl.algorithms.stackelberg.algorithm import Stackelberg
from raymarl.algorithms.stackelberg.batch_log import BatchLossLog
from raymarl.algorithms.stackelberg.runner import (
    StackelbergEnvRunner,
    random_state,
    restore_random_state,
)

from experiments.sn_energy_market.batch_critic_pilot import configuration
from experiments.sn_energy_market.evaluation import source_hashes
from experiments.sn_energy_market.training_audit import TrainingAudit, assert_same_state

SOURCE = ROOT / 'outputs/ausgrid-hybrid-storage/20261023-soc-mode-72rounds'
CHECKPOINT = Path(json.loads((SOURCE/'optimization-stop.json').read_text(encoding='utf-8-sig'))['checkpoint'])
DEST = ROOT / 'outputs/diagnostics/optimization-parity-20261023'
DEST.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
cfg = configuration(DEST)
cfg.validation_seeds = (10042, 10048)
assert cfg.learning_contract() == json.loads((CHECKPOINT/'learning_contract.json').read_text(encoding='utf-8'))
cfg.prepare_policies()
runner = StackelbergEnvRunner(config=cfg)
group = LearnerGroup(config=cfg, module_spec=cfg.get_multi_rl_module_spec())
saved_runner = pickle.loads((CHECKPOINT/'env_runner/state.pkl').read_bytes())
saved_learner = pickle.loads((CHECKPOINT/'learner_group/learner/state.pkl').read_bytes())
weights = {k: pickle.loads((CHECKPOINT/f'learner_group/learner/rl_module/{k}/module_state.pkl').read_bytes()) for k in cfg.agent_order}
learner = group._learner

def load(relative, module_name):
    path = SOURCE / 'sources' / relative
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

old_runner = load('RLAlgo/src/raymarl/algorithms/stackelberg/runner.py', 'raymarl.algorithms.stackelberg.baseline_runner')
old_algorithm = load('RLAlgo/src/raymarl/algorithms/stackelberg/algorithm.py', 'raymarl.algorithms.stackelberg.baseline_algorithm')
old_audit = load('experiments/sn_energy_market/training_audit.py', 'experiments.sn_energy_market.baseline_audit')
initial_rng = pickle.loads((CHECKPOINT/'algorithm_state.pkl').read_bytes())['raymarl_rng']
results, evidence = [], []
for baseline in (True, False):
    with ExitStack() as stack:
        if baseline:
            for method in ('get_state', 'set_state', '_period', 'evaluate'):
                stack.enter_context(patch.object(StackelbergEnvRunner, method, getattr(old_runner.StackelbergEnvRunner, method)))
        runner.set_state(copy.deepcopy(saved_runner))
        learner.set_state(copy.deepcopy(saved_learner))
        learner.module.set_state(copy.deepcopy(weights))
        runner.module.set_state(copy.deepcopy(weights))
        restore_random_state(initial_rng)
        for k in cfg.agent_order:
            assert_same_state(weights[k], learner.module[k].get_state())
        assert_same_state(saved_learner['optimizer'], learner.get_state()['optimizer'])
        label = 'baseline' if baseline else 'optimized'
        algorithm = SimpleNamespace(config=cfg, env_runner=runner, learner_group=group, metrics=MetricsLogger(),
                                    sampled_periods=24*1536, completed_updates=96, outer_round=24,
                                    batch_loss_log=BatchLossLog(DEST/f'{label}-batches.jsonl'))
        cls = old_algorithm.Stackelberg if baseline else Stackelberg
        for name in ('_sync', '_batch_diagnostics', '_adapt', '_snapshot', '_restore'):
            setattr(algorithm, name, MethodType(getattr(cls, name), algorithm))
        audit_class = old_audit.TrainingAudit if baseline else TrainingAudit
        audit = audit_class(algorithm, DEST, full_period_checks=False).install()
        audit.phase = 'round_parity'
        batches = []
        collect = runner.collect
        def capture(*args, _collect=collect, _batches=batches, **kwargs):
            batch = _collect(*args, **kwargs)
            _batches.append(copy.deepcopy(batch))
            return batch
        with patch.object(runner, 'collect', capture):
            started = time.perf_counter()
            algorithm._adapt(tuple(k for k in cfg.agent_order if k != 'uc'), 2, phase='consumer_adaptation')
            consumer_seconds = time.perf_counter()-started
            started = time.perf_counter()
            algorithm._adapt(('uc',), 2, phase='uc_proposal', candidate_index=1)
            uc_seconds = time.perf_counter()-started
        audit.close()
        start = time.perf_counter()
        snapshot = algorithm._snapshot()
        snapshot_seconds = time.perf_counter()-start
        immutable = copy.deepcopy(snapshot)
        start = time.perf_counter()
        algorithm._restore(snapshot)
        restore_seconds = time.perf_counter()-start
        start = time.perf_counter()
        scores = runner.evaluate((10042,10048), **({} if baseline else {'actions_only': True}))
        evaluation_seconds = time.perf_counter()-start
        assert_same_state(snapshot, immutable)
        results.append({'batches': batches, 'learner': copy.deepcopy(learner.get_state()),
                        'rng': random_state(), 'scores': scores, 'runner': runner.get_state()})
        row={'label':label,'consumer_batch_seconds':consumer_seconds,'uc_batch_seconds':uc_seconds,
             'snapshot_seconds':snapshot_seconds,'restore_seconds':restore_seconds,
             'two_month_evaluation_seconds':evaluation_seconds}
        evidence.append(row)
        print(json.dumps(row), flush=True)

for left,right in zip(results[0]['batches'],results[1]['batches'],strict=True):
    for k, rows in left.policy_batches.items():
        for name, value in rows.items():
            np.testing.assert_array_equal(value,right[k][name])
for key in ('learner','rng','scores','runner'):
    assert_same_state(results[0][key], results[1][key])
sources={**source_hashes('RLAlgo/src/raymarl/algorithms/stackelberg'),
         **source_hashes('environments/sn_energy_market/src')}
for name in ('training_audit.py','batch_critic_pilot.py','resume_optimized_study.py','training_env.py'):
    path=ROOT/'experiments/sn_energy_market'/name
    sources[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
report={'passed':True,'checkpoint':str(CHECKPOINT),'baseline_git':'58d8cd8',
        'learning_contract_unchanged':True,'restored_weights_and_adam_exact':True,
        'two_384_step_batches_exact':True,'post_update_weights_adam_rng_exact':True,
        'two_full_month_scores_exact':True,'snapshot_isolation_exact':True,
        'timings':evidence,'optimized_source_sha256':sources}
(DEST/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({'passed':True,'report':str(DEST/'report.json')}),flush=True)
group.shutdown()
