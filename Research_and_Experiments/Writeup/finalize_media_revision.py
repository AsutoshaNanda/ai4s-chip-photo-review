from pathlib import Path
import json
import re

HERE = Path(__file__).resolve().parent
base = HERE / 'before_workflow_redesign/kaggle_writeup.md'
text = base.read_text()
video = json.loads((HERE / 'Video/video_verification.json').read_text())
assert video['duration_seconds'] < 300

def change(old, new):
    global text
    if old not in text:
        raise ValueError(old)
    text = text.replace(old, new)

change('figures/01_workflow.png', 'figures/01_workflow_redesigned.png')
change('Workflow from format check through whole-photo and tile features to human review. Unsupported formats receive no score.', 'Redesigned photo-review process with six numbered stages and a separate manual-review branch.')
change('*Figure 1. Photo-review workflow. This is a method diagram. It does not represent a new experiment.*', '*Figure 1. Photo-review process. The amber branch sends small or unfamiliar formats to manual review without a score. The model names, dimensions and processing steps match the code. This diagram is not an experiment result.*')
change('**Pending:** Add the public video before submission. The local demonstration and a video script are available. The video must be accessible without login or payment. Keep the video within five minutes.', '**Video created:** The recorded demonstration is 4 minutes 34 seconds long. It has 1080p video, synthetic narration and English subtitles. The file is `Video/AI4S_Demo.mp4` and is approximately 6.5 MB.\n\n**Attachment and public link pending:** Browser approval blocked the Kaggle upload. The video must be attached or linked before submission and accessible without login or payment.')
change('**Pending:** Add the public code or Kaggle Notebook link before submission.', '**Pending:** The GitHub repository link will be added by the project owner before submission.')
change('#### 3.2 Why we retained 3×3 tiles', '#### 3.2 Extra training comparison\n\nThe frozen whole-photo model is the reference for these two training changes. Output-layer training did not show a clear AUC increase. Training the last model block decreased AUC in this comparison. These are development results.\n\n![Paired AUC differences for output-layer and last-block training against the frozen whole-photo model.](figures/16_extra_training_comparison.png)\n\n*Figure 16. Dots show paired AUC differences. Bars show the recorded 95% ranges on 2,397 development photos. The output-layer range includes zero. The last-block upper limit is about -0.000020, close to zero. These results do not show that all training methods fail.*\n\n#### 3.3 Why we retained 3×3 tiles')
change('#### 4.1 Cell types and image formats', '#### 4.1 Score reliability\n\nThe score is useful for review order. It is not a guarantee of the chance that one sample has poor quality. The saved score groups show differences between mean scores and observed bad-sample rates.\n\n![Score reliability for the whole-photo and tile models on the same 672 automatically scored reserved photos.](figures/15_score_reliability.png)\n\n*Figure 15. Each point contains 67 or 68 photos. The dotted line shows equal score and observed bad-sample rate. The weighted average gap is 0.045 for whole photos and 0.037 for tiles. These are descriptive results from the saved reserved-set predictions. No calibration setting was selected from this graph.*\n\n#### 4.2 Cell types and image formats')
change('#### 4.2 Working demonstration', '#### 4.3 Working demonstration')
change('We revised the text after a partial language review on 9 October 2026. Full compliance with the dictionary and grammar rules remains unverified.', 'The 10 October review used the full official standard. Sentence and paragraph checks passed within their stated scope. Dictionary checks found remaining word-use deviations. This report is STE-guided, not fully ASD-STE100 compliant.')
change('- Add the public demonstration video.', '- Upload or link the created demonstration video.')
change('The local photo tool and Figure 5 provide the current demonstration.', 'The local photo tool, Figure 5 and the created video give the current demonstration.')

parts = re.split(r'(```[\s\S]*?```|`[^`\n]+`|!?\[[^\]]*\]\([^)]+\))', text)
substitutions = {'retain': 'keep', 'retained': 'kept', 'whether': 'if', 'provide': 'give', 'provides': 'gives', 'provided': 'given'}
for i in range(0, len(parts), 2):
    for old, new in substitutions.items():
        parts[i] = re.sub(r'\b' + old + r'\b', lambda m: new.capitalize() if m.group()[0].isupper() else new, parts[i], flags=re.I)
text = ''.join(parts)
for old, new in {
    'The method achieved a ranking score': 'The method gave a ranking score',
    'The whole-photo reference achieved': 'The whole-photo reference gave',
    'Separate studies tested nucleus analysis and droplet control.': 'Separate studies examined nucleus analysis and droplet control.',
    'We tested one practical question:': 'We examined one practical question:',
    'We did not test a physical chip.': 'We did not do a test on a physical chip.',
    'We tested if learned models improved a simple image-processing reference.': 'We compared learned models with a simple image-processing reference.',
    'We tested control separately because': 'We examined control separately because',
    'C01 and C02 tested this risk through two separate datasets.': 'C01 and C02 examined this risk through two separate datasets.',
    'R03 tests if staged training helps.': 'R03 compares training methods.',
    'R04 tests which inputs help.': 'R04 compares input sets.',
    'It also tests missing inputs and increased measurement noise.': 'It also examines missing inputs and increased measurement noise.',
    'Test if': 'Find if',
    'Test which ordering finds': 'Find which ordering finds',
    'Test the exact public installation instructions.': 'Do a test of the exact public installation instructions.',
    'Test real noise before broad use.': 'Do a test with real noise before broad use.',
    'Test real camera noise before claiming camera robustness.': 'Do a test with real camera noise before you claim camera robustness.',
    'Test another dataset and image types.': 'Do tests on another dataset and different image types.',
    'Test blur, brightness loss and noise.': 'Do tests with blur, brightness loss and noise.',
    'Test flips, turns and brightness changes.': 'Do tests with flips, turns and brightness changes.',
    'Test reproduction in a fresh environment.': 'Do a reproduction test in a fresh environment.',
    'Test shared failures behind disagreement warnings.': 'Examine shared failures behind disagreement warnings.',
    'Test the responsible stages.': 'Do tests of the responsible stages.'
}.items():
    text = text.replace(old, new)

order = list(map(int, re.findall(r'^\*Figure (\d+)\.', text, flags=re.M)))
assert len(order) == len(set(order)) == 16
mapping = {old: i + 1 for i, old in enumerate(order)}
text = re.sub(r'\bFigure (1[0-6]|[1-9])\b', lambda m: f'Figure {mapping[int(m.group(1))]}', text)
old_ids = re.findall(r'^\| ([A-Z]\d\d) \|', base.read_text(), re.M)
assert re.findall(r'^\| ([A-Z]\d\d) \|', text, re.M) == old_ids
assert len(re.findall(r'!\[', text)) == 16
(HERE / 'kaggle_writeup.md').write_text(text)
(HERE / 'media_revision_manifest.json').write_text(json.dumps({'body_figures': 16, 'new_graphs': 2, 'replaced_workflow': True, 'thumbnail_created': True, 'video_duration_seconds': video['duration_seconds'], 'kaggle_upload_status': 'Blocked by browser automatic approval review; not uploaded in this revision', 'github_status': 'User will create the repository later', 'figure_number_mapping': mapping, 'canonical_ID_rows_preserved': len(old_ids)}, indent=2))
print(json.dumps({'figures': 16, 'ledger_rows': len(old_ids), 'words': len(text.split()), 'mapping': mapping}, indent=2))
