"""Write Kaggle job folders (lanes_template.py). Usage: python kaggle/make_jobs2.py NAME [NAME...]  then  kaggle kernels push -p kaggle/jobs/NAME"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOISE = "0.1"      # = largest recorded B04 noise level (b_unet.STRESS); training sd ~ U(0, 0.1)
JOBS = {
    # name: (gpu?, fetch, lanes as [(gpu id or None, [commands])])
    "ai4s-nuc-gpu": (True, ["bbbc039", "bbbc038", "pip:czifile==2026.8.16,imagecodecs==2026.8.16"], [
        (0, [f"scripts/b_unet.py --target fg --seed 0 --train_noise_max {NOISE}",
             f"scripts/b_unet.py --target boundary --seed 0 --train_noise_max {NOISE}",
             "scripts/b_modern.py"]),
        (1, ["scripts/b04_refresh.py --exp B02 --seed 0", "scripts/b04_refresh.py --exp B03 --seed 0",
             "scripts/b07_outside.py --posts v2 --seed 0 --skip_b01", "scripts/b_tiny.py --seed 0",
             "scripts/c03_fetch.py", "scripts/c03_prep.py --depth_rule input_only",
             "scripts/c03_translate.py --slices c03_slices_inputonly.npz"])]),
    "ai4s-nuc-cpu": (False, ["bbbc039", "pip:stable-baselines3==2.9.0,gymnasium==1.4.0"], [
        (None, ["scripts/b_repost.py --exp B02 --seed 1"]),
        (None, ["scripts/b_repost.py --exp B03 --seed 1",
                "scripts/b_repost.py --exp B03 --seed 0 --thr_extra 0.96,0.97,0.98,0.99,0.995,0.999",
                "scripts/b_repost.py --exp B03 --seed 1 --thr_extra 0.96,0.97,0.98,0.99,0.995,0.999"]),
        (None, ["scripts/b_analysis.py --exp B05 --posts v2 --seed 0", "scripts/b_analysis.py --exp B06 --posts v2 --seed 0",
                "scripts/r02_confirm.py"])]),
}
SK = "pip:scikit-image==0.26.0"       # same version as the Mac lock / Modal image (Kaggle's older one lacks max_size)
JOBS["ai4s-nuc-cpu2"] = (False, ["bbbc039", "bbbc038", SK + ",stable-baselines3==2.9.0,gymnasium==1.4.0"], [
    (None, ["scripts/b_analysis.py --exp B05 --posts v2 --seed 0", "scripts/b_analysis.py --exp B06 --posts v2 --seed 0", "scripts/r02_confirm.py"]),
    (None, ["scripts/b_analysis.py --exp B05 --posts v2 --seed 1", "scripts/b_analysis.py --exp B06 --posts v2 --seed 1"]),
    (None, ["scripts/b04_refresh.py --exp B02 --seed 1", "scripts/b04_refresh.py --exp B03 --seed 1"]),
    (None, ["scripts/b07_outside.py --posts v2 --seed 1 --skip_b01"])])
JOBS["ai4s-nuc-cpu3"] = (False, ["bbbc039", SK], [
    (None, ["scripts/b_repost.py --exp B02 --variant _noisetrain --seed 0", "scripts/b04_refresh.py --exp B02 --variant _noisetrain --seed 0"]),
    (None, ["scripts/b_repost.py --exp B03 --variant _noisetrain --seed 0", "scripts/b04_refresh.py --exp B03 --variant _noisetrain --seed 0"]),
    (None, ["scripts/b_tiny_sweep.py --seed 0"]),
    (None, ["scripts/chip_final_checks.py"])])
JOBS["ai4s-c03-v3"] = (True, ["pip:czifile==2026.8.16,imagecodecs==2026.8.16"], [
    (0, ["scripts/c03_fetch.py", "scripts/c03_prep.py --depth_rule input_only", "scripts/c03_translate.py --slices c03_slices_inputonly.npz",
         "scripts/c03_prep.py --depth_rule target", "scripts/c03_translate.py --slices c03_slices.npz"])])
JOBS["ai4s-nuc-cpu4"] = (False, ["bbbc038", "pip:timm==1.0.30,instanseg-torch"], [
    (None, ["scripts/input_gate.py"]), (None, ["scripts/model_versions.py"])])
JOBS["ai4s-f01-nuc"] = (False, ["bbbc039", SK], [(None, ["scripts/f01_final.py --part nucleus"])])
JOBS["ai4s-f01-chip-eval"] = (False, [SK], [(None, ["scripts/f01_final.py --part chip_eval"])])
JOBS["ai4s-chip-cpu"] = (False, [SK], [(None, ["scripts/a05_confirm.py --part fit"])])
for _k in range(5):
    JOBS[f"ai4s-c02v2-fold{_k}"] = (False, [], [(None, [f"scripts/c02_jump.py --fold {_k}"])])
# F03: the retained seed-1 and extended-boundary clean-up searches again, in a fresh environment with every version declared in
# requirements.txt (the earlier f03b repeat changed scikit-image only). One kernel per repeat; same saved fold models, no training.
VP = ["bbbc039", "venv:requirements"]
JOBS["ai4s-f03c-b02s1"] = (False, VP, [(None, ["scripts/b_repost.py --exp B02 --seed 1"])])
JOBS["ai4s-f03c-b03s1"] = (False, VP, [(None, ["scripts/b_repost.py --exp B03 --seed 1"])])
JOBS["ai4s-f03c-b03x0"] = (False, VP, [(None, ["scripts/b_repost.py --exp B03 --seed 0 --thr_extra 0.96,0.97,0.98,0.99,0.995,0.999"])])
JOBS["ai4s-f03c-b03x1"] = (False, VP, [(None, ["scripts/b_repost.py --exp B03 --seed 1 --thr_extra 0.96,0.97,0.98,0.99,0.995,0.999"])])
tpl = (HERE / "lanes_template.py").read_text()
for name in sys.argv[1:]:
    gpu, fetch, lanes = JOBS[name]
    d = HERE / "jobs" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.py").write_text(tpl.replace("__LANES__", repr(lanes)).replace("__FETCH__", repr(fetch)))
    meta = {"id": f"itsasup/{name}", "title": name, "code_file": "run.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": gpu, "enable_internet": True, "dataset_sources": ["itsasup/ai4s-code"],
            "competition_sources": [], "kernel_sources": [], "model_sources": []}
    if gpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print("wrote", d)
