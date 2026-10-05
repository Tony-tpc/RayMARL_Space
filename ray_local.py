"""Configure Ray authentication within this workspace."""

import os
from pathlib import Path
import secrets


def configure():
    root = Path(__file__).resolve().parent
    token_path = root / ".ray" / "auth_token"
    if "RAY_AUTH_TOKEN" not in os.environ and "RAY_AUTH_TOKEN_PATH" not in os.environ:
        token_path.parent.mkdir(exist_ok=True)
        if not token_path.exists():
            token_path.write_text(secrets.token_hex(32), encoding="ascii")
        os.environ["RAY_AUTH_TOKEN_PATH"] = str(token_path)
    os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")


if __name__ == "__main__":
    configure()
