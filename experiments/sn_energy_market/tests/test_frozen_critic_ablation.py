from datetime import datetime

import numpy as np
import torch
from raymarl.algorithms.stackelberg.critic_probe import CriticProbe

from experiments.sn_energy_market.frozen_critic_ablation import (
    chunks_for,
    pack_batch,
    public_calendar,
    schedule_for,
    validate_episode,
)


def test_public_time_features_use_decision_time_and_known_horizon():
    stamp = datetime.fromisoformat("2011-07-01T06:00:00+10:00")
    features = public_calendar(stamp, 12, 1488)
    np.testing.assert_allclose(features[:4], [12 / 1488, 1476 / 1488, 1, 0], atol=1e-7)
    np.testing.assert_allclose(features[-2:], [0, -1], atol=1e-7)


def test_extra_time_inputs_start_at_same_function_as_current_mlp():
    obs, times = torch.randn(4, 7, 68), torch.randn(4, 7, 8)
    base, extra = CriticProbe("mlp", 68, seed=11), CriticProbe("time", 68, seed=11)
    torch.testing.assert_close(base(obs)[0], extra(obs, times)[0], rtol=0, atol=1e-5)


def test_exact_prefix_replay_matches_whole_episode_and_has_no_gradient():
    model = CriticProbe("gru", 5, seed=29)
    sequences = [torch.randn(47, 5), torch.randn(25, 5), torch.randn(15, 5)]
    starts = [17, 9, 0]
    hidden = model.prefix_hidden([sequence[:start] for sequence, start in zip(sequences, starts, strict=True)])
    assert not hidden.requires_grad
    for i, (sequence, start) in enumerate(zip(sequences, starts, strict=True)):
        expected, _ = model(sequence[None])
        actual, _ = model(sequence[None, start:], hidden=hidden[:, i:i + 1])
        torch.testing.assert_close(actual, expected[:, start:], rtol=1e-5, atol=1e-5)
    # Context is rebuilt after parameters change, rather than reusing stale state.
    with torch.no_grad():
        model.gru.weight_ih_l0.add_(0.05)
    changed = model.prefix_hidden([sequences[0][:17]])
    assert not torch.equal(changed, hidden[:, :1])


def test_recurrent_predictions_are_causal_and_reset_per_episode():
    model = CriticProbe("gru", 5, seed=47)
    original, times = torch.randn(70, 5), torch.zeros(70, 8)
    changed = original.clone()
    changed[31:] = 100
    a = model.predict_episode(original, times, chunk_length=13)
    b = model.predict_episode(changed, times, chunk_length=32)
    torch.testing.assert_close(a[:31], b[:31])
    model.predict_episode(torch.randn(19, 5), torch.zeros(19, 8))
    torch.testing.assert_close(a, model.predict_episode(original, times, chunk_length=70))


def test_loss_chunks_cover_each_label_once_without_crossing_months():
    episodes = [{"obs": torch.zeros(n, 5), "calendar": torch.zeros(n, 8),
                 "targets": torch.arange(n).float()} for n in (70, 45)]
    chunks = chunks_for(episodes)
    assert chunks == [(0, 0, 32), (0, 32, 64), (0, 64, 70), (1, 0, 32), (1, 32, 45)]
    assert sum(end - start for _, start, end in chunks) == 115
    for order in schedule_for(chunks, 11, epochs=3):
        assert sorted(order) == list(range(len(chunks)))
    batch = pack_batch(episodes, [chunks[2], chunks[4]])
    assert int(batch["mask"].sum()) == 19
    np.testing.assert_array_equal(batch["targets"][batch["mask"]], [*range(64, 70), *range(32, 45)])


def test_mc_integrity_check_rejects_truncated_or_contaminated_labels():
    import pytest

    gamma = 0.5
    data = {"periods": np.arange(2), "terminated": np.array([False, True]),
            "rewards": np.array([1.0, 2.0]), "targets": np.array([2.0, 2.0]),
            "obs": np.zeros((2, 5)), "uc_actions": np.zeros((2, 6)),
            "consumer_actions": np.zeros((2, 3, 4)),
            "timestamps": np.array(["2011-07-01T00:00:00+10:00", "2011-07-01T00:30:00+10:00"])}
    data["calendar"] = np.stack([public_calendar(datetime.fromisoformat(t), i, 2)
                                 for i, t in enumerate(data["timestamps"])])
    validate_episode(data, gamma)
    data["targets"][0] += 0.1
    with pytest.raises(AssertionError):
        validate_episode(data, gamma)
    data["targets"][0] -= 0.1
    data["terminated"][-1] = False
    with pytest.raises(AssertionError):
        validate_episode(data, gamma)
