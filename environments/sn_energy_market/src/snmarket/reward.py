"""Each agent receives one reward per committed physical period."""


def calculate_rewards(ledger):
    return {"uc": ledger.uc_reward, **{i: -entry.total_cost for i, entry in ledger.consumers.items()}}

