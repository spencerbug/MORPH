import argparse
import copy
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import torch
from .data import prepare, digest
from .model import Predictor, warp
from .evidence import commit, score

ROOT=Path(__file__).resolve().parents[4]
CFG=ROOT/'experiments/e0_rotation/configs/e0.json'
ART=ROOT/'artifacts/e0_rotation'


def setup(device):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    return torch.device(device if device!='auto' else ('mps' if torch.backends.mps.is_available() else 'cpu'))


def provenance(folder, cfg, device):
    folder.mkdir(parents=True,exist_ok=True)
    tracked=[*ROOT.glob('*.md'),*ROOT.glob('*.toml'),*ROOT.glob('requirements*.txt'),
             *ROOT.glob('experiments/e0_rotation/src/**/*.py'),
             *ROOT.glob('experiments/e0_rotation/tests/*.py'),CFG]
    sources={str(p.relative_to(ROOT)):p.read_text() for p in tracked}
    (folder/'sources.json').write_text(json.dumps(sources,indent=2))
    record={'command':sys.argv,'cwd':str(Path.cwd()),'config':cfg,'config_sha256':digest(CFG),
        'sources_sha256':digest(folder/'sources.json'),'python':sys.version,'torch':torch.__version__,
        'numpy':np.__version__,'platform':platform.platform(),'device':str(device),
        'threads':torch.get_num_threads(),'deterministic_algorithms':True,
        'git_revision':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),
        'git_status':subprocess.run(['git','status','--short'],capture_output=True,text=True).stdout,
        'dataset_manifest_sha256':digest(ART/'data/manifest.json') if (ART/'data/manifest.json').exists() else None}
    (folder/'provenance.json').write_text(json.dumps(record,indent=2))
    (folder/'git.diff').write_text(subprocess.run(['git','diff','HEAD'],capture_output=True,text=True).stdout)


@torch.no_grad()
def predict(model, inputs, device, last_only=False):
    batches=[]
    for start in range(0,len(inputs['x0']),128):
        x=torch.from_numpy(inputs['x0'][start:start+128]).to(device)
        a=torch.from_numpy(inputs['actions'][start:start+128]).to(device)
        if model=='P': out=x[:,None].expand(-1,a.shape[1],-1,-1,-1)
        elif model=='W': out=warp(x,a)
        else: out=model.rollout(x,a)
        if last_only: out=out[:,-1:]
        batches.append(out.cpu().numpy())
    return np.concatenate(batches)


def evaluate_schedule(model, name, schedule, folder, device, model_hash, last_only=False):
    data=ART/'data'/name
    input_path=data/f'{schedule}_inputs.npz'
    # Target file is opened only after prediction and commitment.
    with np.load(input_path) as f: inputs={k:f[k] for k in f.files}
    pred=predict(model,inputs,device,last_only)
    ticket=commit(folder,pred,{'model_sha256':model_hash,'input_sha256':digest(input_path),
        'input_events':f'{name}/{schedule}/x0','actions':inputs['actions'].tolist(),
        'last_only':last_only,'split':name,'schedule':schedule})
    target_path=data/f'{schedule}_targets.npz'
    with np.load(target_path) as targets: result=score(ticket,inputs,targets,target_path)
    return result


def train(cfg,method,seed,lr,device,smoke=False):
    name=f'{method}_seed{seed}_lr{lr:g}'+('_smoke' if smoke else '')
    folder=ART/'runs'/name
    if (folder/'result.json').exists():
        return json.loads((folder/'result.json').read_text())
    folder.mkdir(parents=True,exist_ok=True)
    provenance(folder,cfg,device)
    torch.manual_seed(seed)
    model=Predictor(method).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=lr,betas=(.9,.999))
    with np.load(ART/'data/train.npz') as data:
        images=torch.from_numpy(data['images']).to(device)
        actions=torch.from_numpy(data['actions']).to(device)
    if smoke: images,actions=images[:256],actions[:256]
    epochs=1 if smoke else cfg['epochs']
    start_epoch,best,best_epoch,elapsed=0,float('inf'),0,0
    rng=torch.Generator().manual_seed(seed+10000)
    if (folder/'latest.pt').exists():
        checkpoint=torch.load(folder/'latest.pt',map_location=device,weights_only=False)
        model.load_state_dict(checkpoint['model']); opt.load_state_dict(checkpoint['optimizer'])
        rng.set_state(checkpoint['shuffle_rng'].cpu())
        start_epoch=checkpoint['epoch']; best=checkpoint['best']; best_epoch=checkpoint['best_epoch']; elapsed=checkpoint['elapsed']
    run_start=time.perf_counter()
    for epoch in range(start_epoch,epochs):
        epoch_start=time.perf_counter(); model.train()
        order=torch.randperm(len(images),generator=rng).to(device)
        aggregate=torch.zeros(4,device=device)
        for idx in order.split(cfg['batch_size']):
            opt.zero_grad(set_to_none=True)
            loss,parts=model.loss(images[idx],actions[idx])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            opt.step(); aggregate+=parts*len(idx)
        model.eval()
        # Snapshot used for committed validation predictions before targets are read.
        snapshot=folder/'validation_model.pt'
        torch.save(model.state_dict(),snapshot); model_hash=digest(snapshot)
        vals=[]
        for sched in ['positive','negative']:
            target_folder=folder/'validation'/f'epoch{epoch+1:03d}'/sched
            if target_folder.exists():
                # Interrupted epoch: preserve old ticket rather than overwrite.
                target_folder=target_folder.with_name(sched+f'_retry{time.time_ns()}')
            vals.append(evaluate_schedule(model,'validation',sched,target_folder,device,model_hash,True))
        val=float(np.mean([v['foreground_mse'] for v in vals]))
        if not np.isfinite(val): raise RuntimeError('Non-finite validation')
        if val<best:
            best,best_epoch=val,epoch+1
            torch.save(model.state_dict(),folder/'best.pt')
        history={'epoch':epoch+1,'losses':(aggregate/len(images)).cpu().tolist(),
            'validation_primary':val,'seconds':time.perf_counter()-epoch_start}
        with open(folder/'history.jsonl','a') as f:f.write(json.dumps(history)+'\n')
        elapsed_total=elapsed+time.perf_counter()-run_start
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'shuffle_rng':rng.get_state(),
            'epoch':epoch+1,'best':best,'best_epoch':best_epoch,'elapsed':elapsed_total},folder/'latest.tmp')
        os.replace(folder/'latest.tmp',folder/'latest.pt')
        if epoch==0 or (epoch+1)%10==0 or smoke:
            print(f'{name} epoch {epoch+1}/{epochs}: val={val:.6f}, best={best:.6f}, epoch_s={history["seconds"]:.2f}',flush=True)
    result={'run':name,'method':method,'seed':seed,'learning_rate':lr,'best_validation':best,
        'best_epoch':best_epoch,'epochs':epochs,'optimizer_steps':epochs*((len(images)+63)//64),
        'parameters':sum(p.numel() for p in model.parameters()),'seconds':elapsed+time.perf_counter()-run_start,
        'checkpoint_sha256':digest(folder/'best.pt'),'status':'completed'}
    (folder/'result.json').write_text(json.dumps(result,indent=2))
    return result


def campaign(cfg,device):
    pilots=[]
    for method in ['S','U']:
        for lr in cfg['learning_rates']:pilots.append(train(cfg,method,0,lr,device))
    selection={m:min([p for p in pilots if p['method']==m],key=lambda p:p['best_validation'])['learning_rate'] for m in ['S','U']}
    selection_path=ART/'selection.json'
    if selection_path.exists():assert json.loads(selection_path.read_text())['learning_rates']==selection
    else:selection_path.write_text(json.dumps({'learning_rates':selection,'pilots':pilots,'frozen_at_ns':time.time_ns()},indent=2))
    print('Validation-only selection frozen:',selection,flush=True)
    results=[]
    for seed in cfg['model_seeds']:
        for method in ['S','U']:results.append(train(cfg,method,seed,selection[method],device))
    (ART/'campaign.json').write_text(json.dumps(results,indent=2))


@torch.no_grad()
def diagnostics(model, device, folder):
    with np.load(ART/'data/test/positive_inputs.npz') as f:
        x=torch.from_numpy(f['x0']).to(device);a=torch.from_numpy(f['actions']).to(device)
    u,p=model.encode(x)
    u1,p1=model.step(u,p,a[:,0]);_,p2=model.step(u1,p1,a[:,1]);_,psum=model.step(u,p,a[:,0]+a[:,1]);_,pinv=model.step(u1,p1,-a[:,0])
    pred=model.rollout(x,a); flipped=model.rollout(x,-a)
    rec=model.decode(u,p)
    # Commit the intervention before opening targets.
    ticket=commit(folder/'sign_flip',flipped.cpu().numpy(),{'model_sha256':digest(folder.parent/'best.pt'),
        'split':'test','schedule':'positive_sign_flipped','last_only':False})
    target_path=ART/'data/test/positive_targets.npz'
    with np.load(target_path) as f:
        inputs={'x0':x.cpu().numpy()}
        flip_score=score(ticket,inputs,f,target_path)
        future=torch.from_numpy(f['targets'][:,-1]).to(device)
    uf,pf=model.encode(future)
    result={'reconstruction_mse':float((rec-x).square().mean()),
        'u_within_episode_mse':float((u-uf).square().mean()),'p_variance':float(p.var(0).mean()),
        'u_variance':float(u.var(0).mean()),'action_sign_sensitivity':float((pred-flipped).square().mean()),
        'sign_flipped_foreground_mse':flip_score['foreground_mse'],
        'composition_max_abs':float((p2-psum).abs().max()),'inverse_max_abs':float((pinv-p).abs().max())}
    # A separate action-shuffle intervention on mixed schedules.
    mixed_path=ART/'data/test/mixed_inputs.npz'
    with np.load(mixed_path) as f:
        mixed={k:f[k] for k in f.files}
    shuffled={k:v.copy() for k,v in mixed.items()}
    shuffle_rng=np.random.default_rng(701)
    shuffled['actions']=shuffled['actions'][shuffle_rng.permutation(len(shuffled['actions']))]
    shuffled_pred=predict(model,shuffled,device)
    shuffle_ticket=commit(folder/'action_shuffle',shuffled_pred,{
        'model_sha256':digest(folder.parent/'best.pt'),'input_sha256':digest(mixed_path),
        'intervention':'permute action sequences between episodes; seed 701','last_only':False})
    mixed_targets=ART/'data/test/mixed_targets.npz'
    with np.load(mixed_targets) as f:
        shuffle_score=score(shuffle_ticket,mixed,f,mixed_targets)
    result['shuffled_action_foreground_mse']=shuffle_score['foreground_mse']
    (folder/'diagnostics.json').write_text(json.dumps(result,indent=2))


def evaluate_all(cfg,device):
    campaign_results=json.loads((ART/'campaign.json').read_text())
    manifest=json.loads((ART/'data/manifest.json').read_text())
    models=[('P','P',None),('W','W',None)]
    for result in campaign_results:
        model=Predictor(result['method']).to(device)
        path=ART/'runs'/result['run']/'best.pt'
        assert digest(path)==result['checkpoint_sha256']
        model.load_state_dict(torch.load(path,map_location=device,weights_only=True));model.eval()
        models.append((result['run'],model,path))
    for name,model,path in models:
        out=ART/'evaluation'/name
        provenance(out,cfg,device)
        for split,schedules in manifest['schedules'].items():
            if split=='validation':continue
            for schedule in schedules:
                folder=out/split/schedule
                if (folder/'score.json').exists():continue
                evaluate_schedule(model,split,schedule,folder,device,digest(path) if path else name)
            print('Evaluated',name,split,flush=True)
        if path:
            # Diagnostic function obtains checkpoint via this explicit copy-free reference.
            diag_folder=path.parent/'diagnostics'
            if not (diag_folder/'diagnostics.json').exists():diagnostics(model,device,diag_folder)
    (ART/'evaluation_complete.json').write_text(json.dumps({'completed_at_ns':time.time_ns()}))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','smoke','campaign','evaluate']);ap.add_argument('--device',default='auto')
    args=ap.parse_args(); cfg=json.loads(CFG.read_text());device=setup(args.device)
    print('Device:',device,flush=True)
    if args.command=='prepare':prepare(cfg,ART/'data')
    elif args.command=='smoke':
        for m in ['S','U']:train(cfg,m,0,.001,device,True)
    elif args.command=='campaign':campaign(cfg,device)
    elif args.command=='evaluate':evaluate_all(cfg,device)

if __name__=='__main__':main()
