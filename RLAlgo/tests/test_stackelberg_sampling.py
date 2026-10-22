"""Decision-time, physical discount and reproducible sampling checks."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

from raymarl.algorithms.stackelberg.runner import StackelbergEnvRunner, generalized_advantages


class ToyModule(torch.nn.Module):
    def __init__(self, leader):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.leader = leader

    def draw(self, obs, bounds=None, deterministic=False):
        width = 6 if self.leader else 4
        raw = torch.zeros((1, width)) if deterministic else torch.randn((1, width))
        return {
            "storage_mode": torch.zeros(1, dtype=torch.int64),
            "actions": raw.clone(),
            "raw_latents": raw,
            "action_logp": -(raw.square().sum(-1)),
            "conditional_bounds": torch.zeros((1, width, 2)),
            "zero_width_mask": torch.zeros((1, width), dtype=torch.bool),
        }

    def compute_values(self, batch):
        if self.leader:
            return batch["obs"][:, 0] * 10 + self.anchor
        return batch["critic_obs"][:, 0] + 100 * batch["critic_obs"][:, 1] + self.anchor


class ToyModules(torch.nn.ModuleDict):
    def __init__(self):
        super().__init__({"uc": ToyModule(True), "consumer_0": ToyModule(False)})

    def get_state(self):
        return copy.deepcopy(self.state_dict())

    def set_state(self, state):
        self.load_state_dict(state)


class ToySpec:
    def build(self):
        return ToyModules()


class ToyEnv:
    uc_observation_dim = consumer_observation_dim = consumer_value_dim = 2
    consumer_ids = ("consumer_0",)

    def __init__(self, terminal_at=5, data_until=100):
        self.terminal_at, self.data_until = terminal_at, data_until

    def reset(self, seed):
        self.t, self.phase = 0, "UC"
        self.announcement = 0.0
        return np.array([0, 0], dtype=np.float32)

    def uc_bounds(self):
        return np.ones(6)

    def publish(self, raw):
        raw = raw[1:]
        assert self.phase == "UC"
        self.phase, self.announcement = "C", float(raw[0])
        obs = np.array([self.t, self.announcement], dtype=np.float32)
        return SimpleNamespace(
            uc_observation=np.array([self.t, 0], dtype=np.float32),
            uc_action=np.asarray(raw).copy(),
            mapping_context={"t": self.t},
            consumer_observations={"consumer_0": obs},
            consumer_bounds={"consumer_0": np.zeros(13)},
            consumer_value_state=obs,
            period_id=self.t,
        )

    def step(self, actions):
        assert self.phase == "C"
        self.t += 1
        terminal = self.terminal_at is not None and self.t == self.terminal_at
        self.phase = "DONE" if terminal else "UC"
        exhausted = self.t == self.data_until and not terminal
        obs = None if terminal or exhausted else np.array([self.t, 0], dtype=np.float32)
        return SimpleNamespace(
            rewards={"uc": 1.0, "consumer_0": 2.0},
            terminated=terminal,
            next_uc_observation=obs,
            data_exhausted=exhausted,
            metrics={"uc_cash": 1.0, "consumer_cost": -2.0, "served_kwh": 3.0},
        )

    def get_state(self):
        return self.t, self.phase, self.announcement

    def set_state(self, state):
        self.t, self.phase, self.announcement = state

    def append_profiles(self, count):
        self.data_until += count
        return np.array([self.t, 0], dtype=np.float32)


class ToyFactory:
    def __init__(self, terminal_at=5, data_until=100):
        self.terminal_at, self.data_until = terminal_at, data_until

    def __call__(self, seed):
        return ToyEnv(self.terminal_at, self.data_until)


class ToyConfig(AlgorithmConfig):
    def __init__(self, terminal_at=5, data_until=100):
        super().__init__()
        self.train_seeds = (7, 8)
        self.scenario_factory = ToyFactory(terminal_at, data_until)
        self.lambda_gae, self.gamma, self.rollout_length = 0.8, 0.9, 4
        self.critic_lambda = 1.0

    def get_multi_rl_module_spec(self, **kwargs):
        return ToySpec()


def make_runner(**kwargs):
    return StackelbergEnvRunner(config=ToyConfig(**kwargs), worker_index=0)


def assert_batches_equal(left, right):
    for key in left.policy_batches:
        for column in left[key]:
            np.testing.assert_array_equal(left[key][column], right[key][column])


def test_gae_physical_discount_terminal_and_sampling_cut():
    # Last nonterminal row bootstraps, with no fictional second gamma for U -> C.
    advantages, targets = generalized_advantages([1, 2], [3, 4], [4, 5], [False, False], 0.9, 0.8)
    np.testing.assert_allclose(advantages, [1 + 0.9 * 4 - 3 + 0.9 * 0.8 * (2 + 0.9 * 5 - 4), 2.5])
    np.testing.assert_allclose(targets, advantages + [3, 4])
    terminal, _ = generalized_advantages([1, 2, 9], [3, 4, 0], [4, 999, 100], [False, True, False], 0.9, 0.8)
    assert terminal[1] == -2
    assert terminal[0] == pytest.approx(1 + 0.9 * 4 - 3 + 0.9 * 0.8 * -2)


@pytest.mark.parametrize("terminal_at", [2, 3, 8])
def test_separate_critic_targets_follow_rewards_until_terminal_or_cut(terminal_at):
    runner = make_runner(terminal_at=terminal_at)
    saved = runner.get_state()
    batch = runner.collect(5)
    for data in batch.policy_batches.values():
        expected = []
        for start in range(5):
            total, discount = 0.0, 1.0
            for end in range(start, 5):
                total += discount * data["rewards"][end]
                discount *= runner.config.gamma
                if data["terminateds"][end]:
                    break
            else:
                total += discount * data["next_vf_preds"][-1]
            expected.append(total)
        np.testing.assert_allclose(data["value_targets"], expected, atol=1e-12)
    runner.set_state(saved)
    runner.config.critic_lambda = runner.config.lambda_gae
    legacy = runner.collect(5)
    for key in batch.policy_batches:
        # Changing only the critic lambda changes neither samples nor Actor GAE.
        for name in ("actions", "advantages", "vf_preds", "next_vf_preds"):
            np.testing.assert_array_equal(batch[key][name], legacy[key][name])
        np.testing.assert_allclose(
            legacy[key]["value_targets"], legacy[key]["advantages"] + legacy[key]["vf_preds"],
            atol=1e-12,
        )
    fixed = copy.deepcopy(batch)
    with torch.no_grad():
        for module in runner.module.values():
            module.anchor.add_(100)
    assert_batches_equal(batch, fixed)


def test_consumer_bootstrap_uses_next_announced_state_and_preserves_rng():
    runner = make_runner()
    batch = runner.collect(2)
    assert runner.env.phase == "UC" and runner.env.t == 2
    consumer = batch["consumer_0"]
    assert consumer["next_vf_preds"][0] == consumer["vf_preds"][1]
    assert batch["uc"]["next_vf_preds"][-1] == 20
    next_batch = runner.collect(1)
    assert consumer["next_vf_preds"][-1] == next_batch["consumer_0"]["vf_preds"][0]
    # The private critic state never enters either UC input.
    np.testing.assert_array_equal(batch["uc"]["obs"], batch["uc"]["critic_obs"])
    assert np.all(batch["uc"]["obs"][:, 1] == 0)


def test_cut_does_not_consume_random_actions_or_reset_the_environment():
    runner = make_runner(terminal_at=8)
    saved = runner.get_state()
    first, second = runner.collect(2), runner.collect(2)
    runner.set_state(saved)
    whole = runner.collect(4)
    for key in whole.policy_batches:
        for name in ("actions", "raw_latents", "action_logp", "vf_preds", "next_vf_preds", "rewards"):
            np.testing.assert_array_equal(
                np.concatenate([first[key][name], second[key][name]]), whole[key][name]
            )


def test_true_terminal_zero_bootstrap_and_reset_never_join_episode_advantages():
    runner = make_runner(terminal_at=2)
    batch = runner.collect(4)
    for key in batch.policy_batches:
        np.testing.assert_array_equal(batch[key]["terminateds"], [False, True, False, True])
        np.testing.assert_array_equal(batch[key]["next_vf_preds"][[1, 3]], [0, 0])
        for index in (1, 3):
            assert batch[key]["advantages"][index] == pytest.approx(
                batch[key]["rewards"][index] - batch[key]["vf_preds"][index]
            )
    assert runner.num_env_steps_sampled_lifetime == 4


def test_snapshot_restores_complete_sampling_trajectory_and_partial_weights_keep_state():
    runner = make_runner()
    runner.collect(2)
    saved = runner.get_state()
    expected = runner.collect(2)
    runner.set_state(saved)
    assert runner.num_env_steps_sampled_lifetime == 2
    runner.set_state({"rl_module": saved["rl_module"], "weights_seq_no": 3})
    assert runner.env.t == 2
    actual = runner.collect(2)
    assert_batches_equal(expected, actual)


def test_framework_counter_broadcast_does_not_erase_branch_progress():
    from ray.rllib.utils.metrics import NUM_ENV_STEPS_SAMPLED_LIFETIME

    runner = make_runner(terminal_at=8)
    runner.collect(2)
    runner.set_state({NUM_ENV_STEPS_SAMPLED_LIFETIME: 0})
    assert runner.trajectory_periods == 2
    saved = runner.get_state(components="sampling")
    runner.collect(1)
    assert runner.trajectory_periods == 3
    runner.set_state(saved)
    assert runner.trajectory_periods == 2
    assert runner.env.t == 2


def test_native_runner_checkpoint_restores_environment_and_action_stream(tmp_path):
    runner = make_runner(terminal_at=8)
    runner.collect(2)
    path = runner.save_to_path((tmp_path / "runner-checkpoint").as_posix())
    expected = runner.collect(3)
    runner.restore_from_path(path)
    assert runner.env.t == 2
    assert_batches_equal(expected, runner.collect(3))


@pytest.mark.parametrize("gamma", [0.9, 0.99])
def test_evaluation_is_matched_seed_and_does_not_mutate_training_state(gamma):
    runner = make_runner()
    runner.config.gamma = gamma
    runner.collect(2)
    before = runner.get_state()
    rows = runner.evaluate((100, 101))
    assert all(row["physical_periods"] == 5 and row["terminated"] for row in rows)
    assert rows[0]["returns"]["uc"] == pytest.approx(sum(gamma**t for t in range(5)))
    assert rows[0]["cash_profit"] == 5
    assert runner.num_env_steps_sampled_lifetime == 2 and runner.env.t == 2
    assert rows == runner.evaluate((100, 101))
    actual = runner.collect(2)
    runner.set_state(before)
    assert_batches_equal(actual, runner.collect(2))


@pytest.mark.parametrize("terminal_at", [None, 5])
def test_window_evaluation_score_ignores_critic_only_changes(terminal_at):
    from raymarl.algorithms.stackelberg.evaluation import paired_gain, select_candidate

    runner = make_runner(terminal_at=terminal_at)
    baseline = runner.evaluate((100, 101), steps=2, deterministic=True)
    with torch.no_grad():
        runner.module["uc"].anchor.add_(100)
        runner.module["consumer_0"].anchor.sub_(200)
    changed = runner.evaluate((100, 101), steps=2, deterministic=True)
    for before, after in zip(baseline, changed, strict=True):
        assert before["returns"] == after["returns"] == {"uc": 1.9, "consumer_0": 3.8}
        assert before["metrics"] == after["metrics"]
        assert not before["terminated"] and not after["terminated"]
        assert after["bootstrap_values"]["uc"] - before["bootstrap_values"]["uc"] == 100
        assert (
            after["bootstrap_estimated_returns"]["uc"] - before["bootstrap_estimated_returns"]["uc"]
        ) == pytest.approx(81)
        assert (
            after["bootstrap_estimated_returns"]["consumer_0"]
            - before["bootstrap_estimated_returns"]["consumer_0"]
        ) == pytest.approx(-162)
    assert all(not row["score_complete"] for row in baseline + changed)
    with pytest.raises(ValueError, match="complete terminal returns"):
        select_candidate([baseline, changed])
    with pytest.raises(ValueError, match="complete terminal returns"):
        paired_gain(changed, baseline, agent="consumer_0")


def test_continuous_exhaustion_requires_append_and_has_no_terminal_value():
    runner = make_runner(terminal_at=None, data_until=2)
    with pytest.raises(RuntimeError, match="append profiles"):
        runner.collect(2)
    assert runner.env.t == 2 and not runner._terminated
    runner.append_profiles(3)
    batch = runner.collect(2)
    assert batch.env_steps() == 2
    assert not batch["uc"]["terminateds"].any()
    assert batch["uc"]["next_vf_preds"][-1] == 40
