"""G03 speed test: the same small jobs on every machine (Mac CPU, Mac Apple GPU, Kaggle T4/L4, Modal, Vast).

Self-contained on purpose (one file can be pushed to Kaggle/Modal/Vast). Uses random tensors, so it measures
machine speed only; real data-loading speed is measured separately (G01 decode timing).

Jobs (each in try/except; failures are recorded, not hidden):
  1. image AI, frozen feature reading (forward only): resnet18, efficientnet_b0, convnext_tiny at 224/448;
     big modern model vit_large_patch14_dinov2 at 224/518
  2. image AI, full training step (forward+backward+update): resnet18, efficientnet_b0, convnext_tiny at 224/448;
     vit_large at 224
  3. small U-Net (nucleus outlines): training step on 256x256 crops; prediction on one 520x696 image
  4. reward learning: stable-baselines3 PPO on Pendulum-v1 for a fixed number of steps, on CPU and on the GPU
Batch size: largest from a list that fits in memory (recorded). Mixed precision used where supported (recorded).
"""
import json
import os
import platform
import shutil
import subprocess
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

WARMUP, TIMED = 3, 10
BATCHES = [64, 32, 16, 8, 4, 2, 1]


def ensure(pkg, pip_name=None):
    """On cloud notebooks, install a missing package (recorded in the output)."""
    try:
        __import__(pkg)
        return "present"
    except Exception:
        if os.path.isdir("/kaggle") or os.environ.get("G03_ALLOW_PIP"):
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name or pkg], check=False)
            return "installed"
        return "missing"


def device_info():
    info = {"python": sys.version.split()[0], "torch": torch.__version__, "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "machine": platform.machine()}
    try:
        import timm
        info["timm"] = timm.__version__
    except Exception:
        info["timm"] = None
    if torch.cuda.is_available():
        info["device"] = "cuda"
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info["gpu_mem_gb"] = [round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1) for i in range(torch.cuda.device_count())]
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        info["device"] = "mps"
    else:
        info["device"] = "cpu"
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    info["disk_free_gb"] = round(shutil.disk_usage(os.getcwd()).free / 1e9, 1)
    if info["device"] == "cuda":
        try:
            info["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                                 "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        except Exception:
            pass
    return info


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


def empty(dev):
    if dev == "cuda":
        torch.cuda.empty_cache()
    elif dev == "mps":
        torch.mps.empty_cache()


def autocast_ctx(dev):
    if dev == "cuda":
        return torch.autocast("cuda", dtype=torch.float16), "fp16"
    if dev == "mps":
        try:
            ctx = torch.autocast("mps", dtype=torch.float16)
            with ctx:
                pass
            return torch.autocast("mps", dtype=torch.float16), "fp16"
        except Exception:
            pass
    import contextlib
    return contextlib.nullcontext(), "fp32"


def is_oom(e):
    s = str(e).lower()
    return "out of memory" in s or "mps backend out of memory" in s or "invalid buffer size" in s


def bench(fn_make, dev, label):
    """Try batch sizes from large to small; return images (or items) per second for the first that fits."""
    for bs in BATCHES:
        try:
            step = fn_make(bs)
            for _ in range(WARMUP):
                step()
            sync(dev)
            t0 = time.perf_counter()
            for _ in range(TIMED):
                step()
            sync(dev)
            dt = time.perf_counter() - t0
            mem = None
            if dev == "cuda":
                mem = round(torch.cuda.max_memory_allocated() / 1e9, 2)
                torch.cuda.reset_peak_memory_stats()
            elif dev == "mps":
                mem = round(torch.mps.driver_allocated_memory() / 1e9, 2)
            del step
            empty(dev)
            return {"job": label, "batch": bs, "items_per_s": round(bs * TIMED / dt, 2),
                    "s_per_step": round(dt / TIMED, 4), "peak_mem_gb": mem}
        except Exception as e:  # noqa: BLE001
            empty(dev)
            if is_oom(e):
                continue
            return {"job": label, "error": repr(e)[:300]}
    return {"job": label, "error": "did not fit even at batch 1"}


def image_jobs(dev):
    import timm
    out = []
    amp, prec = autocast_ctx(dev)
    specs = [("resnet18", [224, 448]), ("efficientnet_b0", [224, 448]), ("convnext_tiny", [224, 448]),
             ("vit_large_patch14_dinov2.lvd142m", [224, 518])]
    for name, sizes in specs:
        for size in sizes:
            def make_frozen(bs, name=name, size=size):
                kw = {"img_size": size} if "vit" in name else {}
                m = timm.create_model(name, pretrained=False, num_classes=0, **kw).to(dev).eval()
                x = torch.randn(bs, 3, size, size, device=dev)

                def step():
                    with torch.no_grad(), autocast_ctx(dev)[0]:
                        m(x)
                return step
            r = bench(make_frozen, dev, f"frozen_features:{name}@{size}")
            r["precision"] = prec
            out.append(r)
            print(json.dumps(r), flush=True)
            if "vit_large" in name and size > 224:
                continue  # full training of the big model at large size is not part of the plan (frozen first)

            def make_train(bs, name=name, size=size):
                kw = {"img_size": size} if "vit" in name else {}
                m = timm.create_model(name, pretrained=False, num_classes=2, **kw).to(dev).train()
                opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
                scaler = torch.amp.GradScaler("cuda") if dev == "cuda" else None
                x = torch.randn(bs, 3, size, size, device=dev)
                y = torch.randint(0, 2, (bs,), device=dev)

                def step():
                    opt.zero_grad(set_to_none=True)
                    with autocast_ctx(dev)[0]:
                        loss = F.cross_entropy(m(x), y)
                    if scaler:
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                    else:
                        loss.backward()
                        opt.step()
                return step
            r = bench(make_train, dev, f"train_step:{name}@{size}")
            r["precision"] = prec
            out.append(r)
            print(json.dumps(r), flush=True)
    return out


class UNet(nn.Module):
    def __init__(self, c=32, depth=4, out_ch=3):
        super().__init__()
        chs = [c * 2 ** i for i in range(depth)]
        self.down = nn.ModuleList()
        prev = 1
        for ch in chs:
            self.down.append(nn.Sequential(nn.Conv2d(prev, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(True),
                                           nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(True)))
            prev = ch
        self.up = nn.ModuleList()
        self.upconv = nn.ModuleList()
        for ch in reversed(chs[:-1]):
            self.upconv.append(nn.ConvTranspose2d(prev, ch, 2, stride=2))
            self.up.append(nn.Sequential(nn.Conv2d(ch * 2, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(True),
                                         nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(True)))
            prev = ch
        self.head = nn.Conv2d(prev, out_ch, 1)

    def forward(self, x):
        skips = []
        for i, blk in enumerate(self.down):
            x = blk(x)
            if i < len(self.down) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for up, blk in zip(self.upconv, self.up):
            x = up(x)
            x = blk(torch.cat([x, skips.pop()], 1))
        return self.head(x)


def unet_jobs(dev):
    out = []
    amp, prec = autocast_ctx(dev)

    def make_train(bs):
        m = UNet().to(dev).train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda") if dev == "cuda" else None
        x = torch.randn(bs, 1, 256, 256, device=dev)
        y = torch.randint(0, 3, (bs, 256, 256), device=dev)

        def step():
            opt.zero_grad(set_to_none=True)
            with autocast_ctx(dev)[0]:
                loss = F.cross_entropy(m(x), y)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
        return step
    r = bench(make_train, dev, "unet_train_step@256")
    r["precision"] = prec
    out.append(r)
    print(json.dumps(r), flush=True)

    def make_pred(bs):
        m = UNet().to(dev).eval()
        x = torch.randn(bs, 1, 528, 704, device=dev)

        def step():
            with torch.no_grad(), autocast_ctx(dev)[0]:
                m(x)
        return step
    r = bench(make_pred, dev, "unet_predict@528x704")
    r["precision"] = prec
    out.append(r)
    print(json.dumps(r), flush=True)
    return out


def ppo_jobs(dev, steps=8192):
    out = []
    try:
        from stable_baselines3 import PPO
    except Exception as e:
        return [{"job": "ppo", "error": f"stable_baselines3 missing: {e!r}"}]
    for d in ["cpu", dev] if dev != "cpu" else ["cpu"]:
        try:
            model = PPO("MlpPolicy", "Pendulum-v1", n_steps=2048, batch_size=64, n_epochs=10, device=d, verbose=0, seed=0)
            t0 = time.perf_counter()
            model.learn(total_timesteps=steps)
            dt = time.perf_counter() - t0
            r = {"job": f"ppo_pendulum@{d}", "env_steps": steps, "seconds": round(dt, 2), "steps_per_s": round(steps / dt, 1)}
        except Exception as e:  # noqa: BLE001
            r = {"job": f"ppo_pendulum@{d}", "error": repr(e)[:300]}
        out.append(r)
        print(json.dumps(r), flush=True)
    return out


def main():
    t_all = time.time()
    installs = {p: ensure(p, n) for p, n in [("timm", "timm"), ("stable_baselines3", "stable-baselines3"),
                                            ("gymnasium", "gymnasium"), ("psutil", "psutil")]}
    force = os.environ.get("G03_DEVICE")
    info = device_info()
    info["package_installs"] = installs
    dev = force or info["device"]
    info["device_used"] = dev
    print(json.dumps(info), flush=True)
    torch.manual_seed(0)
    results = {"info": info, "jobs": []}
    for fn in (image_jobs, unet_jobs):
        try:
            results["jobs"] += fn(dev)
        except Exception as e:  # noqa: BLE001
            results["jobs"].append({"job": fn.__name__, "error": repr(e)[:300]})
    results["jobs"] += ppo_jobs(dev)
    results["total_seconds"] = round(time.time() - t_all, 1)
    out_dir = os.environ.get("G03_OUT", "/kaggle/working" if os.path.isdir("/kaggle/working") else ".")
    path = os.path.join(out_dir, f"g03_results_{dev}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print("WROTE", path, "total_seconds", results["total_seconds"], flush=True)


if __name__ == "__main__":
    main()
