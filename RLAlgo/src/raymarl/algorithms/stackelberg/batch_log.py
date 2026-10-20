"""Append-only batch measurements and later candidate rollback decisions."""

import json
from pathlib import Path


class BatchLossLog:
    """Audit IDs belong to the file, not to reversible training/checkpoint state."""

    def __init__(self, path):
        self.path = Path(path)
        self.next_id = 1
        self.pending = []
        if self.path.exists():
            with self.path.open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    self.next_id = max(self.next_id, row["batch_id"] + 1)

    def _append(self, row):
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()

    @staticmethod
    def outcome(row, selected):
        phase = row["phase"]
        proposal = phase == "uc_proposal"
        retained = phase not in {"uc_proposal", "response", "deviation"} or (
            phase == "response" and row["candidate_index"] == selected
        )
        roles = {}
        for key in row["modules"]:
            keep = retained or (proposal and key == "uc" and row["candidate_index"] <= selected)
            roles[key] = {"actor": keep and key in row["actor_ids"], "critic": keep}
        any_kept = any(any(values.values()) for values in roles.values())
        return {
            "branch_status": "retained" if retained else "partial" if any_kept else "rolled_back",
            "retained_updates": roles,
            "sampling_retained": retained,
        }

    def record(self, row):
        row = {"event": "batch", "batch_id": self.next_id, **row}
        self.next_id += 1
        if row["phase"] in {"uc_proposal", "response"}:
            row["branch_status"] = "pending"
            self.pending.append(row)
        else:
            row.update(self.outcome(row, selected=0))
        self._append(row)

    def begin_round(self):
        # Interrupted rounds stay pending in the file. A later round must not
        # label their branches using a different candidate selection.
        self.pending.clear()

    def resolve(self, selected):
        # Proposals are cumulative. Only UC actor/critic progress through the
        # chosen proposal survives; every proposal's consumer critic is reset.
        for row in self.pending:
            self._append({
                "event": "branch_resolution", "batch_id": row["batch_id"],
                "selected_candidate": selected, **self.outcome(row, selected),
            })
        self.pending.clear()
