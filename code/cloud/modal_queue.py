"""Crash-tolerant job queue on Modal (runs detached: keeps going with the Mac off).

Launch:  .venv/bin/modal run --detach cloud/modal_queue.py --lane cpu32 --session s1 --used-before 4.6 \
             --jobs "scripts/x.py --a 1;wait:TAG|scripts/y.py"
User rules (6 Oct 2026), checked EVERY 30 SECONDS while jobs run (not only between jobs):
  1. money: once (used_before + live cost of every lane in this session) reaches CAP_USD = $20, i.e. $10 of the
     free $30 left, write a STOP flag and kill the running job; every lane sees the flag within 30 s and stops too.
     used_before = Modal's "Metered Cost" (modal billing summary) at launch plus a small margin for billing lag.
  2. time: a job running over 1 hour is killed, a STOP flag is written, every lane stops (until the user is back).
  3. a job that crashes (non-zero exit) is recorded and the queue moves on to the next job.
Live cost: each lane writes /data/queue/<session>/hb/<lane id>.json every 30 s (start, last, hourly rate);
the total = used_before + sum over heartbeats of (last - start) x rate. Rates = Modal list prices for the
reserved CPU/memory/GPU (modal.com/pricing, read 5 Oct 2026).
"""
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import modal

CODE = Path(__file__).resolve().parents[1]
PKGS = ['numpy==2.5.3', 'pandas==3.0.6', 'scipy==1.18.1', 'scikit-learn==1.9.1', 'scikit-image==0.26.0',
        'tifffile==2026.9.20', 'pillow==12.3.0', 'openpyxl==3.1.5', 'matplotlib==3.11.2', 'pyarrow==25.0.1',
        'torch==2.14.1', 'torchvision==0.29.1', 'timm==1.0.30', 'gymnasium==1.4.0', 'stable-baselines3==2.9.0', 'psutil', 'czifile==2026.8.16', 'imagecodecs==2026.8.16']
HEAVY = ["**/*.npz", "**/*.npy", "**/*.pt", "**/*.pth", "**/*.parquet", "**/*.zip", "**/*.png", "**/*.log", "*.log"]
CAP_USD = 20.0                      # stop when $10 of the free $30 would be left
JOB_LIMIT_S = 3600                  # 1 hour per job
MAX_WAIT_S = 3600                   # 1 hour waiting for another experiment
CHECK_S = 30
RATE = {"cpu32": 32 * 0.0472 + 64 * 0.008, "cpu8": 8 * 0.0472 + 32 * 0.008, "a100": 2.10 + 8 * 0.0472 + 32 * 0.008}

app = modal.App("ai4s-queue")
vol = modal.Volume.from_name("ai4s-data", create_if_missing=True)
# Heartbeats and the STOP flag live in a shared Dict, NOT on the volume: refreshing the volume while a job runs
# wipes files the job has written but not yet saved (seen 6 Oct: A04 smoke lost its log). The volume is only
# refreshed between jobs (nothing open), and saved (commit, which never deletes) every 30 s.
state = modal.Dict.from_name("ai4s-queue-state", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install(*PKGS)
         .add_local_dir(str(CODE / "src"), "/root/code/src")
         .add_local_dir(str(CODE / "scripts"), "/root/code/scripts")
         .add_local_dir(str(CODE / "splits"), "/root/code/splits")
         .add_local_dir(str(CODE / "runs"), "/root/code/runs", ignore=HEAVY))


def _reload():
    """Refresh the shared storage; if a file is busy right now, skip this round (next check is 30 s later)."""
    try:
        vol.reload()
    except Exception:  # noqa: BLE001
        pass


def _commit():
    try:
        vol.commit()
    except Exception:  # noqa: BLE001
        pass


class Lane:
    def __init__(self, lane, session, used_before):
        self.lane, self.used_before, self.session = lane, used_before, session
        self.dir = f"/data/queue/{session}"
        os.makedirs(f"{self.dir}/hb", exist_ok=True)
        self.stop_path = f"{self.dir}/STOP"
        self.id = f"{lane}_{time.strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.start = time.time()
        self.log = f"{self.dir}/{self.id}.log"
        self.beat()

    def say(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')} UTC] {msg}"
        print(line, flush=True)
        with open(self.log, "a") as f:
            f.write(line + "\n")
        _commit()

    def beat(self):
        state[f"{self.session}|hb|{self.id}"] = {"start": self.start, "last": time.time(), "rate": RATE[self.lane]}
        _commit()

    def total_usd(self):
        tot = self.used_before
        for k, h in state.items():
            if k.startswith(f"{self.session}|hb|"):
                tot += (h["last"] - h["start"]) / 3600 * h["rate"]
        return tot

    def stop_reason(self):
        """None, or why everything must stop. Writes the STOP flag when this lane detects the money cap."""
        why = state.get(f"{self.session}|stop")
        if why:
            return why
        tot = self.total_usd()
        if tot >= CAP_USD:
            self.flag(f"money: about ${tot:.2f} of $30 used (stop at ${CAP_USD:.0f}, i.e. $10 left)")
            return state.get(f"{self.session}|stop")
        return None

    def flag(self, reason):
        msg = f"{reason} [{self.id} at {time.strftime('%H:%M:%S')} UTC]"
        state[f"{self.session}|stop"] = msg
        with open(self.stop_path, "w") as f:          # record only; lanes read the Dict
            f.write(msg)
        _commit()


def _run_queue(lane, jobs, workers, session, used_before):
    L = Lane(lane, session, used_before)
    env = dict(os.environ, AI4S_DATA="/data", AI4S_RUNS="/data/runs", AI4S_WORKERS=str(workers), PYTHONUNBUFFERED="1")
    L.say(f"lane {L.id}: {len(jobs)} jobs; used before launch ${used_before:.2f}; cap ${CAP_USD}; job limit {JOB_LIMIT_S // 60} min")
    missing = set()
    for job in jobs:
        why = L.stop_reason()
        if why:
            L.say(f"STOP ({why}); not starting: {job}")
            break
        wait_tag = None
        if job.startswith("wait:"):
            wait_tag, job = job[5:].split("|", 1)
        if wait_tag:
            if wait_tag in missing:
                L.say(f"SKIPPED ({wait_tag} never finished): {job}")
                continue
            t0, found = time.time(), False
            while time.time() - t0 < MAX_WAIT_S and not L.stop_reason():
                _reload()                           # safe: waiting, no job running here
                if any(wait_tag in d and os.path.exists(f"/data/runs/{d}/timing.json") for d in os.listdir("/data/runs")):
                    found = True
                    break
                L.beat()
                time.sleep(CHECK_S)
            if L.stop_reason():
                L.say(f"STOP ({L.stop_reason()}) while waiting; not starting: {job}")
                break
            if not found:
                missing.add(wait_tag)
                L.say(f"SKIPPED (waited {MAX_WAIT_S // 60} min for {wait_tag}): {job}")
                continue
        parts = job.split()
        _reload()                                   # safe: no job is running in this container right now
        L.say(f"START {job}")
        t0 = time.time()
        out = open(f"/tmp/{L.id}_job_{int(t0)}.out", "w")      # local disk: storage must not hold open files
        p = subprocess.Popen([sys.executable, f"/root/code/{parts[0]}", *parts[1:]], env=env, cwd="/root/code",
                             stdout=out, stderr=subprocess.STDOUT, text=True)
        status = None
        while status is None:
            try:
                rc = p.wait(timeout=CHECK_S)
                status = "OK" if rc == 0 else f"CRASHED (exit {rc}) -> moving to next job"
            except subprocess.TimeoutExpired:
                L.beat()
                why = L.stop_reason()
                if time.time() - t0 > JOB_LIMIT_S:
                    L.flag(f"time: {job} ran over {JOB_LIMIT_S // 60} min")
                    why = "time limit"
                if why:
                    p.kill()
                    p.wait()
                    status = f"KILLED ({why})"
        out.close()
        text = open(out.name).read()
        with open(f"{L.dir}/{os.path.basename(out.name)}", "w") as f:
            f.write(text)
        L.beat()
        tail = text[-2500:] if not status.startswith("OK") else ""
        L.say(f"{status}: {job} | {(time.time() - t0) / 60:.1f} min | session total about ${L.total_usd():.2f}\n{tail}")
        if status.startswith("KILLED"):
            break
    L.beat()
    L.say(f"lane finished; session total about ${L.total_usd():.2f}")


@app.function(image=image, volumes={"/data": vol}, cpu=32, memory=65536, timeout=20 * 3600)
def cpu32(jobs: list, session: str, used_before: float):
    _run_queue("cpu32", jobs, 32, session, used_before)


@app.function(image=image, volumes={"/data": vol}, cpu=8, memory=32768, timeout=20 * 3600)
def cpu8(jobs: list, session: str, used_before: float):
    _run_queue("cpu8", jobs, 8, session, used_before)


@app.function(image=image, volumes={"/data": vol}, gpu="A100-40GB", cpu=8, memory=32768, timeout=20 * 3600)
def a100(jobs: list, session: str, used_before: float):
    _run_queue("a100", jobs, 8, session, used_before)


@app.local_entrypoint()
def main(lane: str, jobs: str, session: str, used_before: float):
    fn = {"cpu32": cpu32, "cpu8": cpu8, "a100": a100}[lane]
    fn.remote([j.strip() for j in jobs.split(";") if j.strip()], session, used_before)
