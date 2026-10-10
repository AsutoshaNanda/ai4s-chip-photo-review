"""Run bookkeeping. Every experiment run gets its own folder with its settings, software versions,
machine details, log, outputs, and measured wall-clock time. A shared registry.csv lists all runs."""
import csv
import datetime as dt
import hashlib
import json
import platform
import socket
import sys
import time
from pathlib import Path

import os

from .paths import CODE_ROOT, RUNS_ROOT


def n_workers():
    """Worker processes: AI4S_WORKERS (set by the Modal launcher to the reserved CPUs) or all-but-two local CPUs."""
    w = os.environ.get("AI4S_WORKERS")
    return int(w) if w else max(1, (os.cpu_count() or 2) - 2)


def latest_run(tag):
    """Newest run folder whose name contains `tag`, searching new runs (RUNS_ROOT) and the Mac's record (code/runs)."""
    found = []
    for root in {RUNS_ROOT, CODE_ROOT / "runs"}:
        if root.exists():
            found += [p for p in root.iterdir() if p.is_dir() and tag in p.name]
    if not found:
        raise FileNotFoundError(f"no run folder matching {tag}")
    # same name in both places (cloud copy of a Mac run): prefer RUNS_ROOT, which holds the large files
    return sorted(found, key=lambda p: (p.name, p.parent == RUNS_ROOT))[-1]

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))  # one clock everywhere (Mac and cloud machines)


def now_ist():
    return dt.datetime.now(IST).replace(tzinfo=None)


REGISTRY_FIELDS = ["run_id", "plan_id", "name", "status", "started", "finished", "seconds", "machine", "notes"]


def file_md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def env_info():
    """Software versions and hardware facts for the current process (measured, not assumed)."""
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
    }
    versions = {}
    for mod in ["numpy", "pandas", "scipy", "sklearn", "skimage", "torch", "torchvision", "timm",
                "tifffile", "PIL", "gymnasium", "stable_baselines3"]:
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "?")
        except Exception:
            versions[mod] = None
    info["versions"] = versions
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        pass
    return info


def _append_registry(row):
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    reg = RUNS_ROOT / "registry.csv"
    rows = []
    if reg.exists():
        with open(reg, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["run_id"] != row["run_id"]]
    rows.append({k: row.get(k, "") for k in REGISTRY_FIELDS})
    with open(reg, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
        w.writeheader()
        w.writerows(rows)


class Run:
    """Usage:
        run = Run("G01", "bbbc039_check", config={...})
        run.log("message"); run.save_json("result.json", obj); run.finish("measured", notes="...")
    """

    def __init__(self, plan_id, name, config=None):
        stamp = now_ist().strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{stamp}_{plan_id}_{name}"
        self.plan_id, self.name = plan_id, name
        self.dir = RUNS_ROOT / self.run_id
        self.dir.mkdir(parents=True, exist_ok=False)
        self.t0 = time.time()
        self.started = now_ist().isoformat(timespec="seconds")
        self.config = config or {}
        self.save_json("config.json", self.config)
        self.env = env_info()
        self.save_json("env.json", self.env)
        _append_registry({"run_id": self.run_id, "plan_id": plan_id, "name": name, "status": "running",
                          "started": self.started, "machine": self.env.get("hostname", "")})
        self.log(f"run {self.run_id} started")

    def log(self, msg):
        line = f"[{now_ist().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        with open(self.dir / "log.txt", "a") as f:
            f.write(line + "\n")

    def path(self, name):
        p = self.dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def save_json(self, name, obj):
        with open(self.path(name), "w") as f:
            json.dump(obj, f, indent=2, default=str)

    def save_df(self, name, df):
        df.to_csv(self.path(name), index=False)

    def finish(self, status="measured", notes=""):
        seconds = round(time.time() - self.t0, 2)
        finished = now_ist().isoformat(timespec="seconds")
        self.save_json("timing.json", {"started": self.started, "finished": finished, "wall_seconds": seconds})
        _append_registry({"run_id": self.run_id, "plan_id": self.plan_id, "name": self.name, "status": status,
                          "started": self.started, "finished": finished, "seconds": seconds,
                          "machine": self.env.get("hostname", ""), "notes": notes})
        self.log(f"run finished: status={status}, wall_seconds={seconds}")
        return seconds
