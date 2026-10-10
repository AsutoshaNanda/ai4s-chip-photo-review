"""Write one private Kaggle notebook folder per job from runner_template.py. Usage: python make_jobs.py"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOBS = {
    # name: (command, fetch, gpu)
    "ai4s-c02-fold1": ("scripts/c02_jump.py --fold 1", "", False),
    "ai4s-c02-fold2": ("scripts/c02_jump.py --fold 2", "", False),
    "ai4s-c02-fold3": ("scripts/c02_jump.py --fold 3", "", False),
    "ai4s-c02-fold4": ("scripts/c02_jump.py --fold 4", "", False),
    "ai4s-b03-repost": ("scripts/b_repost.py --exp B03", "bbbc039", False),
    "ai4s-b02-repost": ("scripts/b_repost.py --exp B02", "bbbc039", False),
    "ai4s-c03-v2": ("scripts/c03_fetch.py && scripts/c03_prep.py && scripts/c03_translate.py",
                    "pip:czifile==2026.8.16,imagecodecs==2026.8.16", True),
}
ONLY = __import__("os").environ.get("ONLY")

tpl = (HERE / "runner_template.py").read_text()
for name, (cmd, fetch, gpu) in JOBS.items():
    if ONLY and name != ONLY:
        continue
    d = HERE / "jobs" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.py").write_text(tpl.replace("__JOB__", cmd).replace("__FETCH__", fetch))
    meta = {"id": f"itsasup/{name}", "title": name, "code_file": "run.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": gpu, "enable_internet": True, "dataset_sources": ["itsasup/ai4s-code"],
            "competition_sources": [], "kernel_sources": [], "model_sources": []}
    if gpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print("wrote", d)
