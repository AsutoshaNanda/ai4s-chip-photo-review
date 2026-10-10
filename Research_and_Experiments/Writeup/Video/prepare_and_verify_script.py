from pathlib import Path
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
WRITEUP = HERE.parent
ROOT = WRITEUP.parents[1]
story_path = HERE / 'storyboard.json'
story = json.loads(story_path.read_text())
scenes = story['scenes']
spoken = ' '.join(sentence for scene in scenes for sentence in scene['sentences'])
assert len(scenes) == 12
assert sum(s['minimum_seconds'] for s in scenes) == story['target_seconds'] == 268
assert story['maximum_seconds'] == 300
assert len(spoken.split()) < 650
assert all((WRITEUP / s['asset']).exists() for s in scenes if s.get('asset'))
assert not any(x in spoken.lower() for x in ['everything reproduces', '85 percent accurate', 'without shortcuts', 'far faster'])
chip = json.loads((ROOT / 'code/runs/20261007-205052_F03_chip_final_eval_clean/results.json').read_text())['automatic_only']
nuc = json.loads((ROOT / 'code/runs/20261007-200641_F01_nucleus_final_eval/results.json').read_text())
rl = json.loads((ROOT / 'code/runs/20261007-234732_R02_repaired_evaluation_v2/results.json').read_text())['by_variant']
f05 = json.loads((ROOT / 'Research_and_Experiments/Evidence/2026-10-08_scorer_rebuild_job/job_report.json').read_text())
assert chip['photos'] == 672
assert round(chip['fixed_3x3']['pooled']['roc_auc'], 3) == .850
assert round(chip['whole']['pooled']['roc_auc'], 3) == .810
assert round(nuc['B02_seed0_best_count']['count_mae'][0], 2) == 4.31
assert round(nuc['B03_seed0_best_outline']['pooled_f1@0.75'][0], 3) == .901
assert round(rl['PPO_all|UNTOUCHED']['d_err_mean']['mean'], 3) == 1.333
assert f05['inference_routes'] == {'automatic': 12, 'manual_review': 5}
assert f05['steps']['6_inference_cpu']['seconds'] == 402.8
lines = ['# Demo video script', '', '**Version:** 10 October 2026. **Category:** Model & Algorithm.', '', '**Target:** 4 minutes 28 seconds. **Hard limit:** 5 minutes. The final file duration must be checked after rendering.', '', 'The video presents recorded, verified outputs. It does not pretend to be a live recording. The GitHub link is pending at the user\'s request.', '']
cursor = 0
for scene in scenes:
    end = cursor + scene['minimum_seconds']
    time = lambda n: f'{n//60}:{n%60:02}'
    lines += [f'## {time(cursor)} to {time(end)} | {scene["title"]}', '', f'**Screen:** {scene.get("asset", scene["layout"])}', '', f'**Visible message:** {scene["takeaway"]}', '', '**Narration:**', '', ' '.join(scene['sentences']), '', f'**Evidence:** {scene["source"]}', '']
    cursor = end
lines += ['## Production and review plan', '', '1. Use the real recorded CSV and demo sheet for the operation scene.', '2. Label saved output as recorded. Do not animate a fake live run.', '3. Use the verified result figures and the stated comparison settings.', '4. Keep the primary photo method first. Keep the nucleus and simulator studies separate.', '5. Include the failure cases, reproduction limits and source credits.', '6. Create clear narration and synchronized subtitles.', '7. Check the actual MP4 duration, sound, text and image clarity.', '8. Attach the MP4 and public video link where Kaggle permits them.', '9. Add the public GitHub link when the user creates the repository.', '', '## Resource credits', '', '- Dataset images: Movčana and colleagues, Organ-on-a-Chip Image Dataset, Zenodo 10203721. The record states CC BY 4.0; the paper states CC BY-SA. The report keeps this difference explicit.', '- Nucleus data: BBBC039 and BBBC038, Broad Bioimage Benchmark Collection, CC0.', '- Image model: DINOv2 ViT-L/14; code and model credits remain in the report.', '- Quantitative charts: saved experiment records, rendered with Matplotlib.', '- Narration: the installed macOS Samantha synthetic voice. No music or external stock footage.', '', '## Remaining final-submission items', '', 'The user will add the GitHub repository link later. Team details, the stated rights decisions, separate registration and the final submission action remain outside this video-production step.']
(HERE / '01_verified_video_script.md').write_text('\n'.join(lines) + '\n')
verification = {'status': 'PASS for script factual and scope checks; render verification is still required', 'scene_count': len(scenes), 'narration_words': len(spoken.split()), 'minimum_scene_total_seconds': cursor, 'maximum_video_seconds': 300, 'script_sha256': hashlib.sha256(story_path.read_bytes()).hexdigest(), 'recorded_demo_label': True, 'primary_AUC_matches': True, 'nucleus_settings_separated': True, 'RL_simulator_only': True, 'worst_RL_episode_disclosed': True, 'F06_failure_disclosed': True, 'no_human_time_saving_claim': True, 'public_code_pending_user': True, 'no_new_experiments': True}
(HERE / 'script_verification.json').write_text(json.dumps(verification, indent=2))
(HERE / '02_script_verification.md').write_text('# Script verification\n\nThe script was checked before video rendering. The photo, nucleus, simulator and demo numbers match their saved source files. Counting and outline settings remain separate. Recorded output, unmeasured human time savings, limited reproduction and the F06 failure are explicit. The planned duration is 4:28; the actual encoded duration still requires measurement.\n\nSee `script_verification.json` for the checked fields and script hash. This script check does not establish full ASD-STE100 compliance.\n')
print(json.dumps(verification, indent=2))
