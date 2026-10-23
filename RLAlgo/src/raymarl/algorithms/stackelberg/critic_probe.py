"""Offline value probes; these models do not alter the live PPO modules."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from .rl_module import ValueNetwork, signed_log1p


class CriticProbe(nn.Module):
    """Current value MLP, extra public inputs, or an observation-history GRU.

    Values use the existing fixed output scale. Recurrent prefixes are replayed
    with current weights, without gradients; BPTT applies only to the loss chunk.
    No state or recurrent context is shared between episodes or model instances.
    """

    def __init__(self, kind, observation_dim, *, hidden_dim=32, extra_dim=8, scale=20.0, seed=0):
        super().__init__()
        if kind not in ("mlp", "time", "gru"):
            raise ValueError(kind)
        self.kind, self.scale, self.hidden_dim = kind, scale, hidden_dim
        # All three groups receive identical initial parameters where shapes agree.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            base = ValueNetwork(observation_dim, hidden_dim)
            self.value = base
            if kind == "time":
                expanded = ValueNetwork(observation_dim + extra_dim, hidden_dim)
                state = expanded.state_dict()
                for key, tensor in base.state_dict().items():
                    if key == "encoder.0.weight":
                        state[key].zero_()
                        state[key][:, :observation_dim].copy_(tensor)
                    else:
                        state[key].copy_(tensor)
                expanded.load_state_dict(state)
                self.value = expanded
            if kind == "gru":
                self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, observations, calendar=None, hidden=None):
        inputs = torch.cat((observations, calendar), -1) if self.kind == "time" else observations
        if self.kind != "gru":
            return self.value(inputs) * self.scale, None
        encoded = self.value.encoder(signed_log1p(inputs))
        sequence, hidden = self.gru(encoded, hidden)
        return self.value.head(sequence).squeeze(-1) * self.scale, hidden

    @torch.no_grad()
    def prefix_hidden(self, prefixes):
        """Exact forward context from each real episode start, at current weights."""
        device = next(self.parameters()).device
        hidden = torch.zeros(1, len(prefixes), self.hidden_dim, device=device)
        indices = [i for i, prefix in enumerate(prefixes) if len(prefix)]
        if indices:
            selected = [prefixes[i].to(device) for i in indices]
            lengths = torch.tensor([len(prefix) for prefix in selected], device="cpu")
            encoded = self.value.encoder(signed_log1p(pad_sequence(selected, batch_first=True)))
            packed = pack_padded_sequence(encoded, lengths, batch_first=True, enforce_sorted=False)
            _, context = self.gru(packed)
            hidden[:, indices] = context
        return hidden

    @torch.no_grad()
    def predict_episode(self, observations, calendar, chunk_length=256):
        predictions, hidden = [], None
        for start in range(0, len(observations), chunk_length):
            values, hidden = self(
                observations[None, start:start + chunk_length],
                calendar[None, start:start + chunk_length], hidden,
            )
            predictions.append(values.squeeze(0))
        return torch.cat(predictions)
