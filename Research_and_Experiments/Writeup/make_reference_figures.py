from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT / 'code/runs'
OUT = HERE / 'figures'
TABLES = HERE / 'tables'
BLUE, ORANGE, GRAY = '#136F96', '#C76528', '#627281'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.spines.top': False,
                     'axes.spines.right': False, 'text.color': '#243746', 'axes.labelcolor': '#243746',
                     'axes.titleweight': 'bold', 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
sources = {}

def read_json(path):
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())

final_dir = RUNS / '20261007-205052_F03_chip_final_eval_clean'
final = read_json(final_dir / 'results.json')['automatic_only']
pred_path = final_dir / 'final_predictions.csv'
sources[str(pred_path.relative_to(ROOT))] = hashlib.sha256(pred_path.read_bytes()).hexdigest()
pred = pd.read_csv(pred_path)
pred = pred[pred.route == 'automatic']
assert len(pred) == 672 and pred.label_bad.sum() == 284
rows = []
checks = {}
fig, ax = plt.subplots(figsize=(7.8, 7))
ax.plot([0, 1], [0, 100], color='#9BA7B0', ls=':', lw=1.8, label='Score equals observed share')
for key, column, label, color, style, marker in [('whole', 'p_whole', 'Whole photo', GRAY, '--', 's'), ('fixed_3x3', 'p_fixed_3x3', 'Whole + 3×3 tiles', BLUE, '-', 'o')]:
    record = final[key]['reliability']['10']
    bins = pd.DataFrame(record['bins'])
    assert bins.n.sum() == len(pred)
    assert abs(np.average(bins.mean_pred, weights=bins.n) - pred[column].mean()) < 1e-12
    assert abs(np.average(bins.share_bad, weights=bins.n) - pred.label_bad.mean()) < 1e-12
    gap = float(np.average(abs(bins.mean_pred - bins.share_bad), weights=bins.n))
    assert abs(gap - record['ece']) < 1e-12
    ax.plot(bins.mean_pred, 100 * bins.share_bad, color=color, ls=style, marker=marker, lw=2.2, ms=7, label=label)
    bins['method'] = key
    rows.append(bins)
    checks[key] = {'weighted_bin_gap': gap, 'bin_count': len(bins), 'photos': int(bins.n.sum()), 'smallest_bin': int(bins.n.min()), 'largest_bin': int(bins.n.max())}
ax.set(xlim=(0, 1), ylim=(0, 100), xlabel='Mean review score in each group', ylabel='Observed bad samples (%)', title='Score reliability on the final photos')
ax.legend(loc='upper left', fontsize=10)
ax.grid(alpha=.15)
fig.tight_layout(rect=(0, .12, 1, 1))
fig.text(.03, .04, '672 reserved photos. Ten score groups per model, each with 67 or 68 photos.\nDescriptive result only. The tool uses scores to order review.', fontsize=10, color=GRAY)
fig.savefig(OUT / '15_score_reliability.png', dpi=180, bbox_inches='tight')
fig.savefig(OUT / 'score_reliability.jpg', dpi=180, bbox_inches='tight', pil_kwargs={'quality': 95})
plt.close(fig)
pd.concat(rows, ignore_index=True).to_csv(TABLES / 'score_reliability_source.csv', index=False)

training = read_json(RUNS / '20261007-191149_A02_light_retrain_full/results.json')
variants = [('paired_head_only_minus_frozen_LR', 'vit_large_dinov2@518_head_only', 'Output layer only', BLUE), ('paired_last_block_minus_frozen_LR', 'vit_large_dinov2@518_last_block', 'Last model block', ORANGE)]
training_rows = []
fig, ax = plt.subplots(figsize=(8.5, 5.4))
ax.axvline(0, color=GRAY, ls=':', lw=1.8)
for y, (paired_key, result_key, label, color) in zip([1, 0], variants):
    d = training[paired_key]
    value = d['difference']
    low, high = d['range_95']
    score = training[result_key]['pooled']['roc_auc']
    assert training[result_key]['pooled']['n'] == 2397
    ax.errorbar(value, y, xerr=[[value - low], [high - value]], fmt='o', color=color, markersize=10, capsize=5, elinewidth=2)
    ax.text(value, y + .19, f'AUC {score:.3f}', ha='center', fontsize=12, color=color, fontweight='bold')
    training_rows.append({'method': label, 'auc': score, 'paired_difference': value, 'range_95_low': low, 'range_95_high': high, 'frozen_reference_auc': score - value})
assert abs(training_rows[0]['frozen_reference_auc'] - training_rows[1]['frozen_reference_auc']) < 1e-12
ax.set(yticks=[1, 0], yticklabels=['Output layer only', 'Last model block'], xlim=(-.064, .036), ylim=(-.5, 1.55), xlabel='AUC difference from frozen whole-photo model', title='Extra training did not improve whole-photo ranking')
ax.text(.98, .97, 'Right of zero: higher AUC', transform=ax.transAxes, ha='right', va='top', fontsize=10, color=GRAY)
fig.tight_layout(rect=(0, .2, 1, 1))
fig.text(.025, .055, '2,397 development photos. Dots show paired differences; bars show recorded 95% ranges.\nFrozen reference AUC: 0.843. Last-block upper limit: −0.000020, just below zero.', fontsize=10, color=GRAY)
fig.savefig(OUT / '16_extra_training_comparison.png', dpi=180, bbox_inches='tight')
fig.savefig(OUT / 'extra_training_comparison.jpg', dpi=180, bbox_inches='tight', pil_kwargs={'quality': 95})
plt.close(fig)
pd.DataFrame(training_rows).to_csv(TABLES / 'extra_training_source.csv', index=False)
checks['training'] = training_rows
(HERE / 'reference_figures_verification.json').write_text(json.dumps({'checks': checks, 'source_sha256': sources, 'new_training': False}, indent=2))
print(json.dumps(checks, indent=2))
