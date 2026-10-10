from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT / 'code/runs'
OUT = HERE / 'figures'
OUT.mkdir(exist_ok=True)
TABLES = HERE / 'tables'
TABLES.mkdir(exist_ok=True)
BLUE, ORANGE, GREY = '#136F96', '#C76528', '#627281'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.labelcolor': '#263849', 'text.color': '#263849',
                     'axes.titleweight': 'bold', 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
sources = {}

def read_json(path):
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())

def read_csv(path):
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path)

def save(fig, name, footer):
    fig.text(0.02, 0.01, footer, fontsize=9, color=GREY)
    fig.savefig(OUT / name, dpi=170, bbox_inches='tight')
    plt.close(fig)

chip_dir = RUNS / '20261007-205052_F03_chip_final_eval_clean'
chip = read_json(chip_dir / 'results.json')['automatic_only']
pred = read_csv(chip_dir / 'final_predictions.csv')
curves = read_csv(chip_dir / 'catch_curves.csv')
nuc_dir = RUNS / '20261007-200641_F01_nucleus_final_eval'
nuc = read_json(nuc_dir / 'results.json')
nuc_rows = read_csv(nuc_dir / 'final_per_image.csv')
rl_dir = RUNS / '20261007-234732_R02_repaired_evaluation_v2'
rl = read_csv(rl_dir / 'episodes.csv')
paired = read_csv(rl_dir / 'paired_differences.csv')

fig, ax = plt.subplots(figsize=(12, 4.9))
ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis('off')
boxes = [(0.1, 2.4, 2.1, 1.2, 'Photo input\nCheck format'),
         (3, 2.4, 2.7, 1.2, 'Whole photo + 9 tiles\nFrozen DINOv2'),
         (6.5, 2.4, 2.1, 1.2, 'Combine features\nScore sample quality'),
         (9.4, 2.4, 2.4, 1.2, 'Rank for review\nHuman checks result'),
         (0.1, 0.25, 2.7, 1.15, 'Small or unfamiliar format\nManual review; no score')]
for x,y,w,h,t in boxes:
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.07',facecolor='#EFF5F8',edgecolor=BLUE,linewidth=1.4))
    ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=11)
for a,b in [((2.25,3),(2.9,3)),((5.77,3),(6.4,3)),((8.67,3),(9.3,3)),((1.2,2.3),(1.2,1.5))]:
    ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':GREY,'lw':1.8})
ax.text(6.7,1.1,'Prepared image: grey, long side 1036 pixels.\nWhole-photo view: long side 518 pixels.\nTiles come from the prepared image.',ha='center',va='center',fontsize=12)
ax.set_title('A review aid with a clear route for unsupported photos',loc='left',pad=18,fontsize=17)
save(fig,'01_workflow.png','Method diagram, not a measured result. Source: code/inference.py and scorer metadata.')

fig, axes = plt.subplots(1,2,figsize=(12,5.2),gridspec_kw={'width_ratios':[0.9,1.25]})
for y,key,label,col in [(1,'whole','Whole photo',GREY),(0,'fixed_3x3','Whole + 3×3 tiles',BLUE)]:
    v=chip[key]['pooled']['roc_auc']; lo,hi=chip[key]['auc_95']
    axes[0].errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt='o',color=col,capsize=5,markersize=9)
    axes[0].text(v,y+0.17,f'{v:.3f}',ha='center',fontweight='bold')
axes[0].set(yticks=[0,1],yticklabels=['Whole + 3×3 tiles','Whole photo'],xlim=(0.5,1),ylim=(-0.45,1.5),xlabel='Ranking score (AUC); higher is better')
axes[0].set_title('Final ranking improves')
axes[0].text(0.04,0.48,'Paired gain: +0.040\n95% range: +0.015 to +0.067',fontsize=11,transform=axes[0].transAxes)
for key,label,col,style in [('whole','Whole photo',GREY,'--'),('fixed_3x3','Whole + 3×3 tiles',BLUE,'-'),('random_mean','Random order',ORANGE,':')]:
    d=curves[(curves.scope=='automatic_only')&(curves.subset=='ALL')&(curves.order==key)]
    axes[1].plot(d.fraction_reviewed*100,d.catch_rate*100,label=label,color=col,ls=style,lw=2.3)
axes[1].set(xlim=(0,100),ylim=(0,100),xlabel='Photos reviewed (%)',ylabel='Poor-quality samples found (%)')
axes[1].set_title('Review high scores first'); axes[1].legend(loc='lower right',fontsize=10)
fig.suptitle('672 final photos, including 284 poor-quality samples',fontsize=16,y=1.01)
fig.tight_layout(rect=(0,0.13,1,1))
save(fig,'02_chip_final.png','Clean rebuild. AUC bars show 95% ranges from filename-group resampling. Curves simulate review, not minutes saved.')

sub=[]
for group in ['A549','CACO','HPMEC','HSAEC','HUVEC','NHBE']:
    a=chip['whole']['per_cell_type'][group]; b=chip['fixed_3x3']['per_cell_type'][group]
    sub.append({'cell_type':group,'photos':a['n'],'bad':a['n_bad'],'good':a['n_good'],'whole_auc':a['roc_auc'],'x01_auc':b['roc_auc']})
sub=pd.DataFrame(sub);sub.to_csv(TABLES/'chip_final_by_cell_type.csv',index=False)
fig,ax=plt.subplots(figsize=(10,5.4))
y=np.arange(len(sub))
ax.scatter(sub.whole_auc,y+0.09,label='Whole photo',color=GREY,marker='s',s=55)
ax.scatter(sub.x01_auc,y-0.09,label='Whole + 3×3 tiles',color=BLUE,s=60)
ax.set(yticks=y,yticklabels=[f'{r.cell_type}  (n={r.photos}; bad={r.bad})' for r in sub.itertuples()],xlim=(0.5,1.03),xlabel='Ranking score (AUC); higher is better')
ax.invert_yaxis(); ax.legend(loc='lower left'); ax.set_title('Scores differ between cell types on the same final set',loc='left',pad=15)
fig.tight_layout(rect=(0,0.09,1,1))
save(fig,'03_chip_subgroups.png','Descriptive subgroup results; no subgroup confidence bars. HUVEC: 12 good photos. NHBE: 6 bad photos. Small groups limit certainty.')

demo=ROOT/'Research_and_Experiments/Evidence/2026-10-08_scorer_rebuild_job/sheet_fresh.png'
shutil.copy2(demo,OUT/'04_real_demo.png'); sources[str(demo.relative_to(ROOT))]=hashlib.sha256(demo.read_bytes()).hexdigest()

fig,axes=plt.subplots(1,2,figsize=(12,5.4))
for ax,setting,metric,title,xlab in [(axes[0],'best_count','count_mae','Settings chosen for counting','Mean count error (nuclei/image); lower is better'),(axes[1],'best_outline','pooled_f1@0.75','Settings chosen for outlines','Strict outline score (F1@0.75); higher is better')]:
    vals=[nuc[f'B01_{setting}'][metric][0],nuc[f'B02_seed0_{setting}'][metric][0],nuc[f'B03_seed0_{setting}'][metric][0]]
    ax.barh(['B01: simple reference','B02: learned foreground','B03: learned edges'],vals,color=[GREY,BLUE,ORANGE],height=0.5)
    ax.invert_yaxis();ax.set_xlim(0,6 if metric=='count_mae' else 1.04);ax.set_xlabel(xlab);ax.set_title(title,pad=14)
    for i,v in enumerate(vals):ax.text(v+(.07 if metric=='count_mae' else .008),i,f'{v:.2f}' if metric=='count_mae' else f'{v:.3f}',va='center',fontsize=11)
fig.suptitle('50 final nucleus images: counting and outlines are separate choices',fontsize=16)
fig.tight_layout(rect=(0,0.08,1,.95))
save(fig,'05_nucleus_final.png','Saved averages across five fold models. The images share plates with development. No independent-lab confidence interval is shown.')

fig,axes=plt.subplots(1,2,figsize=(12,5.6))
stresses=[]
for model,folder in [('B02 retained','20261007-011404_B04_stress_refresh_B02_seed0'),('B03 retained','20261007-012125_B04_stress_refresh_B03_seed0'),('B02 noise-trained','20261007-142119_B04_stress_refresh_B02_noisetrain_seed0'),('B03 noise-trained','20261007-122654_B04_stress_refresh_B03_noisetrain_seed0')]:
    s=read_csv(RUNS/folder/'stress_summary.csv');s['model']=model;stresses.append(s)
stress=pd.concat(stresses);stress.to_csv(TABLES/'nucleus_noise_source.csv',index=False)
noise=stress[(stress.setting=='best_outline') & stress.stress.str.contains('noise')]
for (name,col,style) in [('B02 retained',BLUE,'--'),('B03 retained',ORANGE,'--'),('B02 noise-trained',BLUE,'-'),('B03 noise-trained',ORANGE,'-')]:
    s=noise[noise.model==name].sort_values('level')
    axes[0].plot(s.level,s.f1_075,label=name,color=col,ls=style,lw=2,marker='o')
axes[0].set(xlabel='Added noise level (standard deviation)',ylabel='Strict outline score (F1@0.75)',ylim=(0,1));axes[0].set_title('Artificial noise reveals a failure'); axes[0].legend(fontsize=9)
tiny=read_csv(RUNS/'20261007-013014_B05_crowding_tiny_nuclei_posts_v2/B05_found_by_size_decile.csv')
tiny.to_csv(TABLES/'tiny_nuclei_source.csv',index=False)
for key,label,col,mark in [('B01_best_count','B01',GREY,'s'),('B02_best_count','B02',BLUE,'o'),('B03_best_count','B03',ORANGE,'^')]:
    s=tiny[tiny.method==key]
    axes[1].plot(s.size_decile+1,s['found@0.50']*100,label=label,color=col,marker=mark,lw=2)
axes[1].set(xlabel='Nucleus size group (1 = smallest tenth)',ylabel='Nuclei found at overlap 0.50 (%)',ylim=(0,105),xticks=[1,2,4,6,8,10]);axes[1].set_title('Small nuclei remain difficult');axes[1].legend(loc='lower right')
fig.tight_layout(rect=(0,0.12,1,1))
save(fig,'06_nucleus_limits.png','Development results. Left: outline settings. Right: count settings. Noise training helps on matched artificial noise, with a clean-data cost.')

gallery=RUNS/'20261007-013014_B05_crowding_tiny_nuclei_posts_v2/B05_gallery/B02_best_count.png'
shutil.copy2(gallery,OUT/'07_real_nucleus_failures.png');sources[str(gallery.relative_to(ROOT))]=hashlib.sha256(gallery.read_bytes()).hexdigest()

fig,axes=plt.subplots(1,2,figsize=(12,5.6),gridspec_kw={'width_ratios':[1.1,1]})
r=rl[rl.study=='R02'];avg=r.groupby(['variant','condition','scenario'],as_index=False).d_err_mean.mean()
stats=avg.groupby(['variant','condition']).d_err_mean.mean()
x=np.arange(2);w=.25
for i,(var,label,col) in enumerate([('PI','Simple rule (PI)',GREY),('FF+PI','Model + rule (FF+PI)',ORANGE),('PPO_all','Learned control (PPO)',BLUE)]):
    vals=[stats[var,c] for c in ['TRAIN','UNTOUCHED']]
    axes[0].bar(x+(i-1)*w,vals,w,label=label,color=col)
    for xx,v in zip(x+(i-1)*w,vals):axes[0].text(xx,v+.035,f'{v:.3f}',ha='center',fontsize=9)
axes[0].set(xticks=x,xticklabels=['Training ranges','Harder ranges'],ylim=(0,2.6),ylabel='Mean diameter error (micrometres)');axes[0].legend(loc='upper left',fontsize=9);axes[0].set_title('Learned control reduces mean error')
p=paired[(paired.comparison=='R02')&(paired.A=='PPO_all')&(paired.condition=='UNTOUCHED')&(paired.metric=='d_err_mean')]
for y,var in enumerate(['PI','FF+PI']):
    row=p[p.B==var].iloc[0];v=row.difference_A_minus_B
    axes[1].errorbar(v,y,xerr=[[v-row.range_95_low],[row.range_95_high-v]],fmt='o',color=BLUE,markersize=9,capsize=5)
    axes[1].text(v,y+.18,f'{v:+.3f}',ha='center',fontweight='bold')
axes[1].axvline(0,color=GREY,lw=1);axes[1].set(yticks=[0,1],yticklabels=['PPO minus PI','PPO minus FF+PI'],xlim=(-.95,.1),ylim=(-.5,1.5),xlabel='Error difference (micrometres); negative is better');axes[1].set_title('Paired gain on harder ranges')
fig.suptitle('Repaired simulator comparison: 50 scenarios and five saved training seeds',fontsize=15)
fig.tight_layout(rect=(0,0.1,1,.94))
save(fig,'08_rl_simulator.png','Simulator only. Right: 95% ranges resample scenarios after averaging seeds. No physical chip was tested.')

checks={}
automatic=pred[pred.route=='automatic']
assert len(automatic)==672 and automatic.label_bad.sum()==284
def auc(y,s):
    ranks=pd.Series(s).rank().to_numpy();n1=int(np.sum(y));n0=len(y)-n1
    return float((ranks[np.asarray(y)==1].sum()-n1*(n1+1)/2)/(n1*n0))
for key,col in [('whole','p_whole'),('fixed_3x3','p_fixed_3x3')]:
    v=auc(automatic.label_bad.to_numpy(),automatic[col].to_numpy())
    assert abs(v-chip[key]['pooled']['roc_auc'])<1e-12
    d=curves[(curves.scope=='automatic_only')&(curves.subset=='ALL')&(curves.order==key)].sort_values('fraction_reviewed')
    area=float(d.catch_rate.iloc[1:].mean())
    assert abs(area-chip[key]['review_area'])<1e-12
    checks[key]={'auc_recomputed':v,'review_area_recomputed':area}
for name in ['B02_seed0_best_count','B03_seed0_best_outline']:
    rows=nuc_rows[nuc_rows.method==name]
    assert rows.stem.nunique()==50
    mean=float(rows.groupby('fold_model').abs_count_error.mean().mean())
    assert abs(mean-nuc[name]['count_mae'][0])<1e-12
    checks[name]={'count_mae_recomputed':mean,'distinct_images':50}
checks['R02_harder_means']={v:float(stats[v,'UNTOUCHED']) for v in ['PPO_all','PI','FF+PI']}
checks['R02_worst_episode']={v:{'diameter_error':float(r[(r.variant==v)&(r.condition=='UNTOUCHED')].d_err_mean.max()),'outside_range_share':float(r[(r.variant==v)&(r.condition=='UNTOUCHED')].violation_share.max())} for v in ['PPO_all','PI','FF+PI']}
checks['HUVEC']=sub[sub.cell_type=='HUVEC'].to_dict('records')[0]
checks['figures']=sorted(p.name for p in OUT.glob('*.png'))
(HERE/'figure_verification.json').write_text(json.dumps({'checks':checks,'source_sha256':sources},indent=2))
print(json.dumps(checks,indent=2))
