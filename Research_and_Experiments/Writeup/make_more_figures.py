from pathlib import Path
import hashlib
import json
import shutil
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
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'text.color': '#243746', 'axes.labelcolor': '#243746',
                     'axes.titleweight': 'bold', 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
sources = {}
checks = {}

def record(path):
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path

def js(name):
    return json.loads(record(RUNS / name).read_text())

def csv(name):
    return pd.read_csv(record(RUNS / name))

def save(fig, name, note):
    fig.text(0.025, 0.015, note, fontsize=9, color=GRAY)
    fig.savefig(OUT / name, dpi=170, bbox_inches='tight')
    if name == '13_supporting_route_comparisons.png':
        fig.savefig(OUT / 'supporting_studies.jpg', dpi=170, bbox_inches='tight', pil_kwargs={'quality': 95})
    plt.close(fig)

a03 = js('20261006-212404_A03_unseen_cell_types_groups_excluded/results.json')
formats = js('20261006-212925_X01_small_photos_and_formats/results.json')
fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
methods = [('whole', 'Whole photo'), ('whole_plus_patches_3x3', 'Whole + 3×3'), ('whole_plus_patches_5x5', 'Whole + 5×5')]
validation_rows = []
for y, (key, label) in enumerate(methods):
    old = a03['protocols']['shared_groups_original'][key]['pooled']['roc_auc']
    new = a03['protocols']['groups_excluded'][key]['pooled']['roc_auc']
    axes[0].plot([new, old], [y, y], color='#CBD3D9', lw=3)
    axes[0].scatter(old, y, color=GRAY, marker='s', s=65, label='Groups shared' if y == 0 else None)
    axes[0].scatter(new, y, color=BLUE, marker='o', s=65, label='Groups excluded' if y == 0 else None)
    axes[0].text(new, y + .18, f'{new:.3f}', ha='center', fontsize=10)
    axes[0].text(old, y - .2, f'{old:.3f}', ha='center', fontsize=10)
    validation_rows.append({'method': label, 'shared_groups_auc': old, 'excluded_groups_auc': new})
axes[0].set(yticks=np.arange(3), yticklabels=[m[1] for m in methods], xlim=(.68, .94), ylim=(-.7, 2.7), xlabel='AUC; higher is better')
axes[0].invert_yaxis(); axes[0].set_title('A03: excluding related groups\nreduces the apparent score', pad=16)
axes[0].legend(loc='lower left', fontsize=9)
formats_order = [('L/2056x1542', 'Gray 2056×1542\n1,597 photos'), ('RGB/2048x1536', 'RGB 2048×1536\n713 photos'), ('RGB/640x480', 'RGB 640×480\n87 photos; 6 bad')]
format_rows = []
for y, (key, label) in enumerate(formats_order):
    for method, offset, color, marker, name in [('whole', -.09, GRAY, 's', 'Whole photo'), ('fixed_3x3', .09, BLUE, 'o', 'Whole + 3×3')]:
        value = formats[method]['per_photo_format'][key]['roc_auc']
        axes[1].scatter(value, y + offset, color=color, marker=marker, s=65, label=name if y == 0 else None)
        axes[1].text(value + .008, y + offset, f'{value:.3f}', va='center', fontsize=9)
        format_rows.append({'format': key, 'method': method, 'auc': value})
axes[1].axvline(.5, color=GRAY, ls=':', lw=1)
axes[1].set(yticks=np.arange(3), yticklabels=[x[1] for x in formats_order], xlim=(.48, .97), ylim=(-.65, 2.65), xlabel='AUC; higher is better')
axes[1].invert_yaxis(); axes[1].set_title('Small photos remain a failure\nfor the tile method', pad=16)
axes[1].legend(loc='lower right', fontsize=9)
fig.tight_layout(rect=(0, .13, 1, 1))
save(fig, '09_validation_and_formats.png', 'Development data only. Left: 2,397 matched photos. Right: descriptive format results. No final-set or independent-lab claim.')
pd.DataFrame(validation_rows).to_csv(TABLES / 'validation_repair_source.csv', index=False)
pd.DataFrame(format_rows).to_csv(TABLES / 'development_formats_source.csv', index=False)
checks['A03'] = validation_rows

workload = csv('20261007-200641_F01_nucleus_final_eval/workload_curve_B02_count.csv')
shared = js('20261007-124801_B06_shared_failures/results.json')
fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
for key, label, color, style in [('error_left_disagreement_order', 'Model disagreement', BLUE, '-'), ('error_left_random_mean', 'Random order', GRAY, '--'), ('error_left_perfect_order', 'Perfect knowledge', ORANGE, ':')]:
    axes[0].plot(workload.reviewed, workload[key] * 100, color=color, ls=style, lw=2.4, label=label)
point = workload.loc[workload.reviewed == 10].iloc[0]
axes[0].scatter([10], [point.error_left_disagreement_order * 100], color=BLUE, s=65)
axes[0].annotate('10 reviews remove\n36.9% of the error', (10, point.error_left_disagreement_order * 100), (19, 72), arrowprops={'arrowstyle': '->', 'color': BLUE}, fontsize=10)
axes[0].set(xlim=(0, 50), ylim=(0, 105), xlabel='Final images reviewed (50 total)', ylabel='Count error remaining (%)', title='Review order helps under\nthe correction assumption')
axes[0].legend(loc='upper right', fontsize=9)
values = [shared[k]['share_of_b02_error_in_shared_failures'] * 100 for k in ['seed0', 'seed1']]
bars = axes[1].bar(['Seed 0\n83 of 150 images', 'Seed 1\n76 of 150 images'], values, color=[BLUE, ORANGE], width=.55)
axes[1].set(ylim=(0, 100), ylabel='B02 error in shared failures (%)', title='Shared errors limit\na disagreement warning')
for bar, value in zip(bars, values): axes[1].text(bar.get_x() + bar.get_width() / 2, value + 2, f'{value:.1f}%', ha='center')
fig.tight_layout(rect=(0, .12, 1, 1))
save(fig, '10_review_and_shared_failures.png', 'Left: saved final predictions, assuming reviewed counts are corrected. Right: development data. No human time-saving measurement.')
checks['F02_error_removed_at_10'] = 1 - float(point.error_left_disagreement_order)
checks['B06_shared'] = shared

outside = js('20261007-012406_B07_bbbc038_outside_posts_v2/results.json')
outside1 = js('20261007-013013_B07_bbbc038_outside_posts_v2_seed1/results.json')
dev = {'B02': js('20261006-161349_B02_unet_fg_repost_v2/results.json'), 'B03': js('20261006-161337_B03_unet_boundary_repost_v2/results.json')}
fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
positions = np.arange(3)
outside_rows = []
for i, model in enumerate(['B02', 'B03']):
    color = [BLUE, ORANGE][i]
    for ax, setting, metric in [(axes[0], 'best_count', 'count_mae'), (axes[1], 'best_outline', 'pooled_f1@0.75')]:
        vals = [dev[model][setting][metric], outside[f'{model}_{setting}|grey'][metric][0], outside[f'{model}_{setting}|colour'][metric][0]]
        xx = positions + (i - .5) * .32
        ax.bar(xx, vals, width=.3, color=color, label=model)
        for x, value in zip(xx, vals):
            label = '<0.001' if metric != 'count_mae' and value < .001 else f'{value:.2f}' if metric == 'count_mae' else f'{value:.3f}'
            ax.text(x, value + (.6 if metric == 'count_mae' else .02), label, ha='center', fontsize=9)
        for group, value in zip(['development_gray', 'outside_gray', 'outside_color'], vals): outside_rows.append({'model': model, 'setting': setting, 'metric': metric, 'group': group, 'value': value})
for ax in axes:
    ax.set(xticks=positions, xticklabels=['Development\n150 gray images', 'Outside\n519 gray images', 'Outside\n108 color images'])
    ax.legend(fontsize=10)
axes[0].set(ylim=(0, 47), ylabel='Mean count error (nuclei/image)', title='Count settings lose accuracy\non outside images')
axes[1].set(ylim=(0, 1.08), ylabel='Strict outline score (F1@0.75)', title='Color-image outline scores\nare below 0.001')
fig.tight_layout(rect=(0, .12, 1, 1))
save(fig, '11_outside_nucleus_test.png', 'Seed 0, retained v2 settings. Different datasets are separate tests. Outside values average five fold models; no paired causal claim.')
pd.DataFrame(outside_rows).to_csv(TABLES / 'outside_nucleus_source.csv', index=False)
checks['color_scores_both_seeds'] = {seed: {key: value['pooled_f1@0.75'][0] for key, value in dataset.items() if '|colour' in key} for seed, dataset in [('seed0', outside), ('seed1', outside1)]}
assert all(value < .001 for scores in checks['color_scores_both_seeds'].values() for value in scores.values())

gallery = record(RUNS / '20261007-013014_B05_crowding_tiny_nuclei_posts_v2/B05_gallery/B03_best_outline.png')
shutil.copy2(gallery, OUT / '12_b03_outline_examples.png')

paths = sorted(RUNS.glob('*C02_jump_batch_rescaling_v2_fold*/per_fold.csv'))
assert len(paths) == 5
c02 = pd.concat([pd.read_csv(record(path)) for path in paths], ignore_index=True)
raw = c02[c02.method == 'raw']
chosen = c02[c02.chosen_on_training]
assert len(raw) == 5 and len(chosen) == 5
c03 = js('20261007-142534_C03_dic_to_lamin_v3_inputonly/results.json')
fig, axes = plt.subplots(1, 3, figsize=(13, 5.3))
for ax, metric, title, ylabel, ylim in [(axes[0], 'replicate_mAP', 'C02: biological matching\nimproves', 'Matching score; higher is better', .08), (axes[1], 'batch_info', 'C02: batch identification\ngets worse', 'Batch accuracy; lower is better', 1.0)]:
    vals = [float(raw[metric].mean()), float(chosen[metric].mean())]
    ax.bar(['Raw', 'Selected\nrescaling'], vals, color=[GRAY, BLUE], width=.55)
    for x, value in enumerate(vals): ax.text(x, value + ylim * .025, f'{value:.3f}', ha='center')
    ax.set(ylim=(0, ylim), ylabel=ylabel, title=title)
    checks[f'C02_{metric}'] = {'raw': vals[0], 'selected': vals[1]}
vals = [c03['all_planes|B_direct']['count_mae'], c03['all_planes|A_predict_picture']['count_mae']]
axes[2].bar(['Direct\nmeasurement', 'Predict, then\nmeasure'], vals, color=[GRAY, ORANGE], width=.55)
for x, value in enumerate(vals): axes[2].text(x, value + .35, f'{value:.3f}', ha='center')
axes[2].set(ylim=(0, 15), ylabel='Mean count error (nuclei/image)', title='C03: predicted fluorescence\ndoes not help counting')
fig.tight_layout(rect=(0, .12, 1, 1))
save(fig, '13_supporting_route_comparisons.png', 'C02: unweighted means of five held-out folds. C03: input-only depth selection, 200 planes. Separate tasks; no combined score.')
checks['C03_exact_count_errors'] = {'direct': vals[0], 'predict_then_measure': vals[1]}
c02.to_csv(TABLES / 'c02_plot_source.csv', index=False)

episodes = csv('20261007-234732_R02_repaired_evaluation_v2/episodes.csv')
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={'width_ratios': [.8, 1.35]})
staged = episodes[(episodes.study == 'R03') & (episodes.condition == 'UNTOUCHED')]
staged_means = staged.groupby(['variant', 'scenario']).d_err_mean.mean().groupby('variant').mean()
names = ['ordinary', 'staged', 'nominal_only']
vals = [float(staged_means[n]) for n in names]
axes[0].bar(['Ordinary', 'Staged', 'Nominal\nonly'], vals, color=[BLUE, ORANGE, GRAY], width=.6)
for i, v in enumerate(vals): axes[0].text(i, v + .03, f'{v:.3f}', ha='center')
axes[0].set(ylim=(0, 1.9), ylabel='Mean diameter error (micrometers)', title='R03: ordinary training\nhas lower error')
r04 = episodes[episodes.study == 'R04'].groupby(['variant', 'condition', 'scenario']).d_err_mean.mean().groupby(['variant', 'condition']).mean()
variants = ['inputs_all', 'inputs_history', 'inputs_minimal', 'inputs_pumps', 'inputs_camera']
conditions = ['TRAIN', 'missing_0.1', 'missing_0.3', 'noise_x2', 'noise_x4']
matrix = np.array([[r04[v, c] for c in conditions] for v in variants])
im = axes[1].imshow(matrix, cmap='YlOrBr', vmin=0, vmax=2.5, aspect='auto')
axes[1].set(xticks=range(5), xticklabels=['Normal', '10%\nmissing', '30%\nmissing', 'Noise\n×2', 'Noise\n×4'], yticks=range(5), yticklabels=['All inputs', 'History', 'Minimal', 'Pump settings', 'Camera only'], title='R04: missing inputs and noise\nincrease error')
for y in range(5):
    for x in range(5): axes[1].text(x, y, f'{matrix[y, x]:.3f}', ha='center', va='center', color='white' if matrix[y, x] > 1.6 else '#243746', fontsize=10)
fig.colorbar(im, ax=axes[1], label='Error (micrometers)', fraction=.04, pad=.03)
fig.tight_layout(rect=(0, .12, 1, 1))
save(fig, '14_rl_training_and_inputs.png', 'Repaired simulator only. Left: harder ranges. Right: training ranges. Each value averages five saved seeds within 50 scenarios.')
checks['R03_harder_means'] = dict(zip(names, vals))
pd.DataFrame(matrix, index=variants, columns=conditions).to_csv(TABLES / 'r04_stress_source.csv')
checks['new_figures'] = ['09_validation_and_formats.png', '10_review_and_shared_failures.png', '11_outside_nucleus_test.png', '12_b03_outline_examples.png', '13_supporting_route_comparisons.png', '14_rl_training_and_inputs.png']
(HERE / 'more_figures_verification.json').write_text(json.dumps({'checks': checks, 'source_sha256': sources}, indent=2))
print(json.dumps(checks, indent=2))
