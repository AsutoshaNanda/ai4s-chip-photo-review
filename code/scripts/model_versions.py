"""G01 (improvements.md): record exact software versions and exact model-file versions (SHA-256) used by the experiments.
Downloads the public model files once (DINOv2 ViT-L via timm; InstanSeg single_channel_nuclei) and hashes them, so the
records name the file bytes, not just a model name. No training."""
import hashlib
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai4s.runlog import Run


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    run = Run("G01", "model_and_package_versions", config={})
    out = {"packages": {}, "model_files": {}}
    for m in ["numpy", "pandas", "scipy", "sklearn", "skimage", "torch", "torchvision", "timm", "PIL", "stable_baselines3", "gymnasium", "instanseg"]:
        try:
            out["packages"][m] = getattr(importlib.import_module(m), "__version__", "no __version__")
        except Exception as e:                                            # noqa: BLE001
            out["packages"][m] = f"not installed ({type(e).__name__})"
    try:
        import timm
        m = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, dynamic_img_size=True)
        cfg = m.pretrained_cfg
        out["model_files"]["vit_large_patch14_dinov2.lvd142m"] = {"pretrained_cfg": {k: str(v) for k, v in cfg.items()}}
        from huggingface_hub import hf_hub_download, scan_cache_dir
        for repo in scan_cache_dir().repos:
            if "dinov2" in repo.repo_id.lower():
                for rev in repo.revisions:
                    for f in rev.files:
                        out["model_files"].setdefault("vit_large_patch14_dinov2.lvd142m", {}).setdefault("files", []).append(
                            {"repo": repo.repo_id, "commit": rev.commit_hash, "file": f.file_name, "sha256": sha(f.file_path)})
    except Exception as e:                                                # noqa: BLE001
        out["model_files"]["vit_large_patch14_dinov2.lvd142m"] = {"error": f"{type(e).__name__}: {e}"}
    try:
        from instanseg import InstanSeg
        InstanSeg("single_channel_nuclei", verbosity=0)
        import instanseg as pkg
        root = Path(pkg.__file__).parent / "bioimageglio_models"
        files = [p for p in Path(pkg.__file__).parent.rglob("*") if p.is_file() and "single_channel_nuclei" in str(p)]
        out["model_files"]["instanseg_single_channel_nuclei"] = [{"file": str(p.relative_to(Path(pkg.__file__).parent)), "sha256": sha(p)} for p in files]
    except Exception as e:                                                # noqa: BLE001
        out["model_files"]["instanseg_single_channel_nuclei"] = {"error": f"{type(e).__name__}: {e}"}
    run.log(json.dumps(out, indent=1)[:6000])
    run.save_json("versions.json", out)
    run.finish("measured", notes="package versions and model-file hashes")


if __name__ == "__main__":
    main()
