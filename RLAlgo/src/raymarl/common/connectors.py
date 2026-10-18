"""Learner connectors give every agent the same joint-transition row identity."""

import numpy as np
import torch
from ray.rllib.connectors.connector_v2 import ConnectorV2

CRITIC_ID = "central_critic"
JOINT_ROW = "joint_row"


class JointRowIdentity(ConnectorV2):
    def __call__(self, *, rl_module, batch, episodes, **kwargs):
        position = 0
        for episode in episodes:
            for sa in episode.agent_episodes.values():
                if len(sa) != len(episode):
                    raise ValueError("Joint learning requires synchronous episode transitions")
                self.add_n_batch_items(
                    batch,
                    JOINT_ROW,
                    items_to_add=np.arange(position, position + len(episode)),
                    num_items=len(sa),
                    single_agent_episode=sa,
                )
            position += len(episode)
        return batch


def validate_joint_rows(batch, ids):
    rows = [batch[mid][JOINT_ROW] for mid in ids]
    if any(not torch.equal(rows[0], row) for row in rows[1:]):
        raise ValueError("Joint agent batches have different transition row identities")


class JointBatch(ConnectorV2):
    def __call__(self, *, rl_module, batch, episodes, **kwargs):
        if CRITIC_ID not in rl_module:
            return batch
        ids = rl_module[CRITIC_ID].actor_ids
        if set(batch) != set(ids):
            raise ValueError("Joint learning requires one synchronized batch for every actor")
        validate_joint_rows(batch, ids)
        batch[CRITIC_ID] = dict(batch[ids[0]])
        return batch
