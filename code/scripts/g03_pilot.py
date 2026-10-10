"""G03 real-data pilot (playbook requirement; audit gap): for each pipeline, 16 real training examples, one
short training pass, save -> reload -> identical predictions; data-preparation time measured separately
from prediction time; memory and versions recorded. Runs on whatever machine it is launched on (Modal).

  A (chip photos): 16 practice photos (prepared grayscale arrays), small pretrained model, one pass.
  B (nuclei): 16 practice images, small U-Net, one short pass.
  R (droplet control): PPO for a few thousand steps on the droplet environment, on CPU and on GPU.
"""
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ai4s import bbbc039 as B
from ai4s.paths import OOC_DIR, SPLITS_ROOT
from ai4s.runlog import Run

DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


def save_reload_same(model, make, x):
    """Predictions before saving vs after reloading into a fresh model: must be identical."""
    model.eval()
    with torch.no_grad():
        before = model(x).float().cpu()
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    size_mb = len(buf.getvalue()) / 1e6
    buf.seek(0)
    fresh = make().to(DEV)
    fresh.load_state_dict(torch.load(buf, map_location=DEV))
    fresh.eval()
    with torch.no_grad():
        after = fresh(x).float().cpu()
    return bool(torch.equal(before, after)), float((before - after).abs().max()), size_mb


def pilot_chip(run):
    import timm
    sp = pd.read_csv(SPLITS_ROOT / "ooc_splits.csv", dtype={"image_id": str})
    prac = sp[sp.role == "practice"]
    pick = pd.concat([prac[prac.label_bad == 0].sample(8, random_state=0), prac[prac.label_bad == 1].sample(8, random_state=0)])
    t0 = time.perf_counter()
    imgs = []
    for iid in pick["image_id"]:
        g = np.load(OOC_DIR / "gray1036" / f"{iid}.npy")
        x = torch.from_numpy(g).float().div(255)[None, None]
        x = F.interpolate(x, size=(336, 448), mode="bilinear", align_corners=False)   # long side 448
        imgs.append(x.expand(-1, 3, -1, -1))
    x = torch.cat(imgs).to(DEV)
    x = (x - torch.tensor([0.485, 0.456, 0.406], device=DEV)[:, None, None]) / torch.tensor([0.229, 0.224, 0.225], device=DEV)[:, None, None]
    y = torch.tensor(pick["label_bad"].to_numpy(), device=DEV)
    sync()
    prep_s = time.perf_counter() - t0

    def make():
        return timm.create_model("resnet18.tv_in1k", pretrained=False, num_classes=2)
    m = timm.create_model("resnet18.tv_in1k", pretrained=True, num_classes=2).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    t1 = time.perf_counter()
    m.train()
    for i in range(0, 16, 8):
        opt.zero_grad()
        F.cross_entropy(m(x[i:i + 8]), y[i:i + 8]).backward()
        opt.step()
    sync()
    train_s = time.perf_counter() - t1
    t2 = time.perf_counter()
    same, maxdiff, mb = save_reload_same(m, make, x)
    sync()
    pred_s = (time.perf_counter() - t2) / 32  # two prediction passes of 16 photos
    return {"pipeline": "A chip photos", "examples": 16, "prep_seconds": prep_s, "prep_s_per_photo": prep_s / 16,
            "train_pass_seconds": train_s, "predict_s_per_photo": pred_s, "reload_identical": same,
            "max_abs_diff": maxdiff, "model_file_mb": mb, "model": "resnet18.tv_in1k (pretrained) at 448x336"}


def pilot_nuclei(run):
    from g03_speed_test import UNet
    sp = pd.read_csv(SPLITS_ROOT / "bbbc039_splits.csv")
    stems = sp[sp.role == "practice"]["stem"].tolist()[:16]
    t0 = time.perf_counter()
    xs, ys = [], []
    for s in stems:
        im = B.read_image(s).astype(np.float32)
        lo, hi = np.percentile(im, [1, 99.8])
        xs.append(torch.from_numpy(np.clip((im - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32))[None])
        ys.append(torch.from_numpy((B.read_labels(s) > 0).astype(np.int64)))
    x = torch.stack(xs)[:, :, :512, :688].to(DEV)
    y = torch.stack(ys)[:, :512, :688].to(DEV)
    sync()
    prep_s = time.perf_counter() - t0

    def make():
        return UNet(out_ch=2)
    m = make().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    t1 = time.perf_counter()
    m.train()
    for i in range(0, 16, 4):
        opt.zero_grad()
        F.cross_entropy(m(x[i:i + 4]), y[i:i + 4]).backward()
        opt.step()
    sync()
    train_s = time.perf_counter() - t1
    t2 = time.perf_counter()
    same, maxdiff, mb = save_reload_same(m, make, x[:4])
    sync()
    pred_s = (time.perf_counter() - t2) / 8
    return {"pipeline": "B nuclei", "examples": 16, "prep_seconds": prep_s, "prep_s_per_image": prep_s / 16,
            "train_pass_seconds": train_s, "predict_s_per_image": pred_s, "reload_identical": same,
            "max_abs_diff": maxdiff, "model_file_mb": mb, "model": "small U-Net (from scratch) on 512x688 crops"}


def pilot_rl(run):
    from stable_baselines3 import PPO
    from ai4s.rl.droplet_env import DropletEnv
    out = []
    for dev in ["cpu"] + (["cuda"] if DEV == "cuda" else []):
        env = DropletEnv(obs_set="all", tau_range=(0.5, 5.0), mismatch_d=0.03, mismatch_rate=0.2, noise_d_um=2.0, noise_rate_rel=0.05)
        t0 = time.perf_counter()
        model = PPO("MlpPolicy", env, n_steps=2048, batch_size=64, device=dev, seed=0, verbose=0)
        model.learn(total_timesteps=4096)
        train_s = time.perf_counter() - t0
        obs = np.stack([env.reset(seed=i)[0] for i in range(16)])
        a1, _ = model.predict(obs, deterministic=True)
        buf = io.BytesIO()
        model.save(buf)
        mb = len(buf.getvalue()) / 1e6
        buf.seek(0)
        m2 = PPO.load(buf, device=dev)
        a2, _ = m2.predict(obs, deterministic=True)
        out.append({"pipeline": f"R droplet control ({dev})", "env_steps": 4096, "train_seconds": train_s,
                    "steps_per_s": 4096 / train_s, "reload_identical": bool(np.array_equal(a1, a2)),
                    "max_abs_diff": float(np.abs(a1 - a2).max()), "model_file_mb": mb})
    return out


def main():
    run = Run("G03", "real_data_pilot", config={"device": DEV, "examples": 16})
    results = []
    for fn in (pilot_chip, pilot_nuclei):
        try:
            r = fn(run)
        except Exception as e:  # noqa: BLE001
            r = {"pipeline": fn.__name__, "error": repr(e)[:400]}
        results.append(r)
        run.log(json.dumps(r, default=str))
    try:
        for r in pilot_rl(run):
            results.append(r)
            run.log(json.dumps(r, default=str))
    except Exception as e:  # noqa: BLE001
        results.append({"pipeline": "R", "error": repr(e)[:400]})
        run.log(results[-1])
    if DEV == "cuda":
        results.append({"peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                        "gpu": torch.cuda.get_device_name(0)})
    run.save_json("pilot_results.json", results)
    ok = all(r.get("reload_identical", True) and "error" not in r for r in results)
    run.finish("measured" if ok else "failed", notes="real-data pilot: save/reload identical" if ok else "pilot problem; see log")


if __name__ == "__main__":
    main()
