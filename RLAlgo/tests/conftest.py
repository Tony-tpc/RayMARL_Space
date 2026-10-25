import os
import secrets
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def ray_runtime(tmp_path_factory):
    import ray

    root = Path(__file__).resolve().parents[2]
    secret_path = root / ".ray" / "auth_token"
    secret_path.parent.mkdir(exist_ok=True)
    if not secret_path.exists():
        secret_path.write_text(secrets.token_hex(32))
    os.environ["RAY_AUTH_TOKEN_PATH"] = str(secret_path)
    os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
    ray.init(
        num_cpus=2,
        include_dashboard=False,
        _node_ip_address="127.0.0.1",
        _temp_dir=str(tmp_path_factory.mktemp("ray")),
    )
    yield
    ray.shutdown()
