"""Read-only integrity review of frozen E0 artifacts."""
import json
from pathlib import Path
import numpy as np
from .run import ART, ROOT
from .data import digest, render


def main():
    manifest=json.loads((ART/'data/manifest.json').read_text())
    checks={}
    for rel,expected in manifest['files'].items():
        assert digest(ART/'data'/rel)==expected,rel
    checks['dataset_files_verified']=len(manifest['files'])
    with np.load(ART/'data/train.npz') as data:
        assert set(data.files)=={'images','actions'}
        assert data['images'].shape==(8192,3,1,32,32)
        assert data['actions'].shape==(8192,2,1)
    for p in (ART/'data').glob('*/*_inputs.npz'):
        with np.load(p) as data:assert set(data.files)=={'x0','actions'}
    checks['learner_input_schema']='pixels and actions only'
    geo=json.loads((ART/'data/geometry_evaluator.json').read_text())
    assert len(set(geo['geometry_hashes']))==96
    checks['distinct_geometry_hashes']=96
    checks['renderer_symmetry_max_mse']={}
    for n in [2,4]:
        values=[]
        for shape in geo[f'symmetry{n}']:
            im=render(shape,[.31,.31+2*np.pi/n]);values.append(float(np.mean((im[0]-im[1])**2)))
        checks['renderer_symmetry_max_mse'][str(n)]=max(values)
    runs=json.loads((ART/'campaign.json').read_text());selection=json.loads((ART/'selection.json').read_text())
    assert len(runs)==10
    for r in runs:
        assert r['epochs']==50 and r['optimizer_steps']==6400
        assert r['learning_rate']==selection['learning_rates'][r['method']]
        assert digest(ART/'runs'/r['run']/'best.pt')==r['checkpoint_sha256']
    checks['selected_runs_verified']=10
    ticket_count=0
    for score_path in ART.rglob('score.json'):
        folder=score_path.parent
        score=json.loads(score_path.read_text());ticket=json.loads((folder/'ticket.json').read_text())
        assert digest(folder/'ticket.json')==score['ticket_sha256']
        assert digest(folder/'predictions.npy')==ticket['prediction_sha256']
        assert score['scored_at_ns']>ticket['committed_at_ns']
        if 'evaluation' in folder.parts:
            assert ticket['committed_at_ns']>selection['frozen_at_ns']
        ticket_count+=1
    checks['verified_ticket_score_pairs']=ticket_count
    checks['test_predictions_after_selection']=True
    checks['status']='passed'
    path=ROOT/'experiments/e0_rotation/reports/integrity.json'
    path.write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
