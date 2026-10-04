"""Append-only, hash-checked prediction and score artifacts, without belief credit."""
import json
import time
from pathlib import Path
import numpy as np
from .data import digest


def commit(folder, predictions, metadata):
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=False)
    np.save(folder/'predictions.npy',predictions)
    ticket={**metadata,'prediction_sha256':digest(folder/'predictions.npy'),
            'committed_at_ns':time.time_ns(),'scoring_rule':'foreground_union_and_full_mse_v1'}
    (folder/'ticket.json').write_text(json.dumps(ticket,indent=2))
    (folder/'ticket.sha256').write_text(digest(folder/'ticket.json'))
    return folder


def score(folder, inputs, targets, target_path):
    folder=Path(folder)
    if digest(folder/'ticket.json') != (folder/'ticket.sha256').read_text():
        raise ValueError('Committed ticket was modified')
    ticket=json.loads((folder/'ticket.json').read_text())
    if digest(folder/'predictions.npy') != ticket['prediction_sha256']:
        raise ValueError('Committed prediction was modified')
    target_hash=digest(target_path)
    if (folder/'score.json').exists():
        old=json.loads((folder/'score.json').read_text())
        if old['target_sha256'] != target_hash:
            raise ValueError('Attempt to rescore with different evidence')
        return old
    pred=np.load(folder/'predictions.npy')
    truth=targets['targets']
    if ticket.get('last_only',False): truth=truth[:,-1:]
    assert pred.shape==truth.shape
    squared=(pred-truth)**2
    union=(inputs['x0'][:,None]>0)|(truth>0)
    foreground=(squared*union).sum((2,3,4))/union.sum((2,3,4)).clip(1)
    full=squared.mean((2,3,4))
    np.savez(folder/'errors.npz',foreground=foreground,full=full,object_ids=targets['object_ids'])
    result={'foreground_mse':float(foreground.mean()),'full_mse':float(full.mean()),
            'target_sha256':target_hash,'ticket_sha256':digest(folder/'ticket.json'),
            'scored_at_ns':time.time_ns()}
    assert result['scored_at_ns'] > ticket['committed_at_ns']
    (folder/'score.json').write_text(json.dumps(result,indent=2))
    return result
