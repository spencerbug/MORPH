import json
import numpy as np
import pytest
import torch
from e0_rotation.model import Predictor, warp
from e0_rotation.data import render, geometries
from e0_rotation.evidence import commit, score


def test_rotation_direction_and_direct_render_inverse():
    shape=[[.4,0,.08,.08,0,1]]
    frames=render(shape,[0,np.pi/2,2*np.pi])
    row,col=np.indices((32,32))
    assert (col*frames[0,0]).sum()/frames[0].sum()>20
    assert (row*frames[1,0]).sum()/frames[1].sum()<11
    np.testing.assert_allclose(frames[0],frames[2],atol=1e-6)
    w=warp(torch.from_numpy(frames[:1]),torch.tensor([[[np.pi/2]]])).numpy()[0,0]
    np.testing.assert_allclose(w,frames[1],atol=1e-6)


def test_group_and_shared_initialization():
    torch.manual_seed(0);s=Predictor('S')
    torch.manual_seed(0);u=Predictor('U')
    for a,b in zip(s.encoder.parameters(),u.encoder.parameters()):torch.testing.assert_close(a,b)
    for a,b in zip(s.decoder.parameters(),u.decoder.parameters()):torch.testing.assert_close(a,b)
    ident=torch.randn(20,32);p=torch.randn(20,16);a=torch.randn(20,1);b=torch.randn(20,1)
    i,q=s.step(ident,p,a);_,r=s.step(i,q,b);_,total=s.step(ident,p,a+b);_,inverse=s.step(i,q,-a)
    torch.testing.assert_close(r,total,atol=1e-5,rtol=0)
    torch.testing.assert_close(inverse,p,atol=1e-5,rtol=0)
    torch.testing.assert_close(i,ident,atol=0,rtol=0)
    for model in [s,u]:
        assert model.rollout(torch.rand(2,1,32,32),torch.rand(2,8,1)).shape==(2,8,1,32,32)


def test_future_is_absent_from_predictor_interface():
    model=Predictor('S').eval();x=torch.rand(2,1,32,32);a=torch.rand(2,2,1)
    target=torch.rand(2,2,1,32,32)
    before=model.rollout(x,a).detach().clone()
    target.zero_()
    torch.testing.assert_close(model.rollout(x,a),before,atol=0,rtol=0)


def test_tickets_immutable_and_idempotent(tmp_path):
    inputs={'x0':np.ones((2,1,32,32),np.float32)}
    targets={'targets':np.ones((2,1,1,32,32),np.float32),'object_ids':np.array([0,1])}
    target_path=tmp_path/'targets.npz';np.savez(target_path,**targets)
    folder=commit(tmp_path/'ticket',targets['targets'].copy(),{})
    first=score(folder,inputs,targets,target_path)
    assert first==score(folder,inputs,targets,target_path)
    assert first['foreground_mse']==first['full_mse']==0
    with pytest.raises(FileExistsError):commit(folder,targets['targets'],{})
    np.save(folder/'predictions.npy',targets['targets']*0)
    with pytest.raises(ValueError):score(folder,inputs,targets,target_path)


def test_shape_hashes_and_symmetry():
    geo=geometries({'geometry_seed':100,'symmetry_seeds':[500,600]})
    assert len(set(geo['geometry_hashes']))==96
    for n in [2,4]:
        for shape in geo[f'symmetry{n}']:
            frames=render(shape,[.31,.31+2*np.pi/n])
            np.testing.assert_allclose(frames[0],frames[1],atol=1e-6)


def test_ticket_metadata_mutation_rejected(tmp_path):
    inputs={'x0':np.ones((1,1,32,32),np.float32)}
    targets={'targets':np.ones((1,1,1,32,32),np.float32),'object_ids':np.array([0])}
    target_path=tmp_path/'targets.npz';np.savez(target_path,**targets)
    folder=commit(tmp_path/'ticket',targets['targets'],{'model_sha256':'original'})
    ticket=json.loads((folder/'ticket.json').read_text());ticket['model_sha256']='modified'
    (folder/'ticket.json').write_text(json.dumps(ticket))
    with pytest.raises(ValueError):score(folder,inputs,targets,target_path)
