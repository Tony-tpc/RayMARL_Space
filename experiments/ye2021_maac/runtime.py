"""Workspace runtime configuration, intentionally excluded from the library."""
import os
from pathlib import Path
import secrets

ROOT = Path(__file__).resolve().parents[2]


def initialize_ray():
    token = ROOT/".ray"/"auth_token"
    token.parent.mkdir(exist_ok=True)
    if not token.exists():
        token.write_text(secrets.token_hex(32), encoding="ascii")
    os.environ["RAY_AUTH_TOKEN_PATH"] = str(token)
    os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
    import ray
    import torch
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    ray.init(num_cpus=2, include_dashboard=False, _node_ip_address="127.0.0.1",
             _temp_dir=str(ROOT/".tmp"/"experiment-ray"))


def stop_owned_ray():
    import ray
    import psutil
    owned = {}
    for p in psutil.process_iter(["pid", "name"]):
        try:
            cmd = " ".join(p.cmdline()).casefold()
            if str(ROOT/".tmp").casefold() in cmd and str(ROOT/".venv"/"Lib"/"site-packages"/"ray").casefold() in cmd:
                owned[p.pid] = p
                for child in p.children(recursive=True):
                    owned[child.pid] = child
        except psutil.Error:
            continue
    for p in owned.values():
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(list(owned.values()), timeout=5)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    return len(owned)
