"""Generate a reproducible report from saved scores; never train or select models."""
import csv
import json
import os
from pathlib import Path
import numpy as np
from .run import ROOT, ART, CFG
os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPORT=ROOT/'experiments/e0_rotation/reports'


def errors(name,split,schedule,metric='foreground',horizon=-1):
    with np.load(ART/'evaluation'/name/split/schedule/'errors.npz') as f:
        values=f[metric][:,horizon]; ids=f['object_ids']
        return np.array([values[ids==i].mean() for i in np.unique(ids)])


def primary(name,split='test',metric='foreground',horizon=-1):
    return np.mean([errors(name,split,s,metric,horizon) for s in ['positive','negative']],0)


def one_step(name):
    return np.mean([errors(name,'test',f'one_{d:+d}') for d in [-15,-5,0,5,15]],0)


def main():
    assert (ART/'evaluation_complete.json').exists(), 'Wait for complete frozen evaluation'
    REPORT.mkdir(exist_ok=True)
    cfg=json.loads(CFG.read_text());runs=json.loads((ART/'campaign.json').read_text())
    selection=json.loads((ART/'selection.json').read_text())
    methods={m:sorted([r for r in runs if r['method']==m],key=lambda r:r['seed']) for m in ['S','U']}
    arrays={m:np.stack([primary(r['run']) for r in rs]) for m,rs in methods.items()}
    rng=np.random.default_rng(cfg['bootstrap_seed']); draws=[]
    for _ in range(cfg['bootstrap_draws']):
        si=rng.integers(0,5,5);oi=rng.integers(0,16,16)
        sm=arrays['S'][si[:,None],oi].mean();um=arrays['U'][si[:,None],oi].mean()
        draws.append((um-sm)/um)
    gain=(arrays['U'].mean()-arrays['S'].mean())/arrays['U'].mean()
    ci=np.percentile(draws,[2.5,97.5]);wins=int(np.sum(arrays['S'].mean(1)<arrays['U'].mean(1)))
    np.savez(REPORT/'paired_primary.npz',S=arrays['S'],U=arrays['U'],bootstrap_gains=draws)
    rows=[]
    for m,names in [('S',[r['run'] for r in methods['S']]),('U',[r['run'] for r in methods['U']]),('P',['P']),('W',['W'])]:
        rows.append({'method':m,'test_h8_foreground':float(np.mean([primary(n).mean() for n in names])),
            'test_h8_full':float(np.mean([primary(n,metric='full').mean() for n in names])),
            'test_one_step_foreground':float(np.mean([one_step(n).mean() for n in names])),
            'seen_h8_foreground':float(np.mean([primary(n,'seen').mean() for n in names])),
            'symmetry2_h8_foreground':float(np.mean([primary(n,'symmetry2').mean() for n in names])),
            'symmetry4_h8_foreground':float(np.mean([primary(n,'symmetry4').mean() for n in names]))})
    with open(REPORT/'metrics.csv','w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    row={r['method']:r for r in rows}
    gates={'gain_at_least_10_percent':bool(gain>=.1),'paired_interval_above_zero':bool(ci[0]>0),
        'positive_gain_at_least_four_seeds':wins>=4,
        'beats_persistence':row['S']['test_h8_foreground']<row['P']['test_h8_foreground'],
        'one_step_noninferiority':row['S']['test_one_step_foreground']<=1.05*row['U']['test_one_step_foreground']+.0001,
        'full_image_not_worse':row['S']['test_h8_full']<=row['U']['test_h8_full']}
    summary={'relative_gain':float(gain),'paired_95_ci':ci.tolist(),'seed_wins':wins,'gates':gates,
             'scientific_gate_pass':all(gates.values()),'metrics':rows}
    (REPORT/'summary.json').write_text(json.dumps(summary,indent=2))
    # Preserve all pilot outcomes, including black-output fits.
    pilot_diagnostics=[]
    for r in selection['pilots']:
        p=ART/'runs'/r['run']/'validation'/f"epoch{r['best_epoch']:03d}"/'positive/predictions.npy'
        a=np.load(p,mmap_mode='r')
        pilot_diagnostics.append({'run':r['run'],'best_epoch':r['best_epoch'],'validation_mse':r['best_validation'],
            'prediction_mean':float(a.mean()),'prediction_max':float(a.max()),'fraction_above_0_01':float((a>.01).mean())})
    (REPORT/'pilot_diagnostics.json').write_text(json.dumps(pilot_diagnostics,indent=2))
    # Compact diagnostics inventory.
    diagnostic=[]
    for r in runs:
        d=json.loads((ART/'runs'/r['run']/'diagnostics/diagnostics.json').read_text())
        diagnostic.append({'method':r['method'],'seed':r['seed'],**d})
    with open(REPORT/'diagnostics.csv','w') as f:
        writer=csv.DictWriter(f,fieldnames=list(diagnostic[0]));writer.writeheader();writer.writerows(diagnostic)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for m,color in [('S','#2463ac'),('U','#da7333')]:
        axes[0].plot(range(5),arrays[m].mean(1),'-o',label=m,color=color)
        curves=[]
        for r in methods[m]:
            curves.append([primary(r['run'],horizon=h-1).mean() for h in [1,2,4,8]])
        curves=np.array(curves)
        axes[1].plot([1,2,4,8],curves.mean(0),'-o',label=m,color=color)
        axes[1].fill_between([1,2,4,8],curves.min(0),curves.max(0),alpha=.15,color=color)
    axes[0].set(xlabel='Model seed',ylabel='Horizon-8 foreground-union MSE',title='Paired unseen-object results')
    axes[1].set(xlabel='Rollout horizon',ylabel='Foreground-union MSE',title='Mean and seed range')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(REPORT/'prediction_metrics.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,m in zip(axes,['S','U']):
        for r in selection['pilots']:
            if r['method']!=m:continue
            history=[json.loads(l) for l in (ART/'runs'/r['run']/'history.jsonl').read_text().splitlines()]
            ax.plot([h['epoch'] for h in history],[h['validation_primary'] for h in history],label=f'lr={r["learning_rate"]}')
        ax.set(title=f'{m}: seed-0 pilot',xlabel='Epoch',ylabel='Validation horizon-8 MSE');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(REPORT/'pilot_curves.png',dpi=180);plt.close(fig)
    # Median and worst S errors for seed 0, selected by declared rule.
    sname=methods['S'][0]['run'];uname=methods['U'][0]['run']
    folder=ART/'evaluation'/sname/'test/positive'
    with np.load(folder/'errors.npz') as f:ranking=np.argsort(f['foreground'][:,-1])
    chosen=[int(ranking[len(ranking)//2]),int(ranking[-1])]
    with np.load(ART/'data/test/positive_inputs.npz') as f:x0=f['x0']
    with np.load(ART/'data/test/positive_targets.npz') as f:target=f['targets']
    preds={name:np.load(ART/'evaluation'/name/'test/positive/predictions.npy',mmap_mode='r') for name in [sname,uname,'W']}
    fig,axes=plt.subplots(2,6,figsize=(12,4))
    for rownum,i in enumerate(chosen):
        panels=[x0[i,0],target[i,-1,0],preds[sname][i,-1,0],preds[uname][i,-1,0],preds['W'][i,-1,0],abs(preds[sname][i,-1,0]-target[i,-1,0])]
        for ax,im,title in zip(axes[rownum],panels,['Initial','Target +120°','Structured S','Unrestricted U','Pixel warp W','S absolute error']):
            ax.imshow(im,cmap='gray',vmin=0,vmax=1);ax.set_title(title,fontsize=9);ax.axis('off')
    fig.suptitle('Seed 0: median (top) and worst (bottom) S foreground error',fontsize=11)
    fig.tight_layout();fig.savefig(REPORT/'examples.png',dpi=180);plt.close(fig)
    # Fixed example 0 across every seed exposes unstable/action-insensitive fits.
    fig,axes=plt.subplots(5,5,figsize=(10,10))
    for seed in range(5):
        sn=methods['S'][seed]['run'];un=methods['U'][seed]['run']
        sp=np.load(ART/'evaluation'/sn/'test/positive/predictions.npy',mmap_mode='r')
        up=np.load(ART/'evaluation'/un/'test/positive/predictions.npy',mmap_mode='r')
        flip=np.load(ART/'runs'/un/'diagnostics/sign_flip/predictions.npy',mmap_mode='r')
        panels=[x0[0,0],target[0,-1,0],sp[0,-1,0],up[0,-1,0],flip[0,-1,0]]
        for ax,im,title in zip(axes[seed],panels,['Initial','Target +120°','Structured S','Unrestricted U','U with −120° action']):
            ax.imshow(im,cmap='gray',vmin=0,vmax=1);ax.set_xticks([]);ax.set_yticks([])
            if seed==0:ax.set_title(title,fontsize=9)
        axes[seed,0].set_ylabel(f'Seed {seed}')
    fig.suptitle('Fixed test example 0 across all five seeds',fontsize=12)
    fig.tight_layout();fig.savefig(REPORT/'all_seed_examples.png',dpi=160);plt.close(fig)
    # Flatten schedule scores for downstream independent analysis.
    schedule_rows=[]
    for m,names in [('S',[r['run'] for r in methods['S']]),('U',[r['run'] for r in methods['U']]),('P',['P']),('W',['W'])]:
        for split in ['test','seen','symmetry2','symmetry4']:
            for schedule in json.loads((ART/'data/manifest.json').read_text())['schedules'][split]:
                schedule_rows.append({'method':m,'split':split,'schedule':schedule,
                    'final_foreground_mse':float(np.mean([errors(n,split,schedule).mean() for n in names])),
                    'final_full_mse':float(np.mean([errors(n,split,schedule,'full').mean() for n in names]))})
    with open(REPORT/'schedules.csv','w') as f:
        writer=csv.DictWriter(f,fieldnames=list(schedule_rows[0]));writer.writeheader();writer.writerows(schedule_rows)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
