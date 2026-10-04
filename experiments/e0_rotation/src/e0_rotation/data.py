"""Simulator/evaluator only; learner loaders expose pixels and actions."""
import hashlib
import json
from pathlib import Path
import numpy as np

ACTIONS = np.deg2rad([-15, -5, 0, 5, 15]).astype(np.float32)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def render(shape, angles):
    """Direct canonical-shape render. Shape rows: cx,cy,rx,ry,phi,intensity."""
    angles = np.atleast_1d(angles).astype(np.float64)
    q = (np.arange(128) + .5 - 64) / 64
    xx, yy = np.meshgrid(q, -q)
    co, si = np.cos(angles)[:, None, None], np.sin(angles)[:, None, None]
    # Inverse rotate each sensor point into canonical coordinates.
    x, y = co * xx + si * yy, -si * xx + co * yy
    out = np.zeros_like(x)
    for cx, cy, rx, ry, phi, value in shape:
        dx, dy = x - cx, y - cy
        ex = np.cos(phi) * dx + np.sin(phi) * dy
        ey = -np.sin(phi) * dx + np.cos(phi) * dy
        inside = (ex / rx)**2 + (ey / ry)**2 <= 1
        out = np.maximum(out, inside * value)
    return out.reshape(-1, 32, 4, 32, 4).mean((2, 4)).astype(np.float32)[:, None]


def sample_shape(rng):
    return np.column_stack([rng.uniform(-.25, .25, (3, 2)),
                            rng.uniform(.08, .22, (3, 2)),
                            rng.uniform(0, 2*np.pi, 3), rng.uniform(.4, 1, 3)])


def geometries(cfg):
    rng = np.random.default_rng(cfg['geometry_seed'])
    shapes, rejects = [], 0
    while len(shapes) < 96:
        shape = sample_shape(rng)
        images = render(shape, [0, np.pi/4])
        if np.mean((images[0]-images[1])**2) <= .002:
            rejects += 1
        else:
            shapes.append(shape.tolist())
    symmetry = {}
    for n, seed in zip([2, 4], cfg['symmetry_seeds']):
        rng = np.random.default_rng(seed)
        objects = []
        for _ in range(8):
            e = sample_shape(rng)[0]
            copies = []
            for phi in np.arange(n)*2*np.pi/n:
                cx, cy = e[:2]
                copies.append([np.cos(phi)*cx-np.sin(phi)*cy,
                               np.sin(phi)*cx+np.cos(phi)*cy,
                               *e[2:4], e[4]+phi, e[5]])
            objects.append(copies)
        symmetry[f'symmetry{n}'] = objects
    hashes = [hashlib.sha256(json.dumps(s).encode()).hexdigest() for s in shapes]
    assert len(set(hashes)) == 96
    return {'train': shapes[:64], 'validation': shapes[64:80], 'test': shapes[80:],
            **symmetry, 'rejections': rejects, 'geometry_hashes': hashes}


def make_training(shapes, cfg, path):
    rng = np.random.default_rng(cfg['train_seed'])
    images, actions, angles = [], [], []
    for shape in shapes:
        theta = rng.uniform(0, 2*np.pi, cfg['train_episodes_per_object'])
        a = rng.choice(ACTIONS, (len(theta), 2))
        theta_all = np.column_stack([theta, theta[:, None]+np.cumsum(a, axis=1)])
        images.append(render(shape, theta_all.reshape(-1)).reshape(-1, 3, 1, 32, 32))
        actions.append(a[..., None]); angles.append(theta_all)
    # Only images/actions are in the learner file. Metadata is evaluator-side.
    np.savez(path/'train.npz', images=np.concatenate(images), actions=np.concatenate(actions))
    np.save(path/'train_angles_evaluator.npy', np.concatenate(angles))


def make_evaluation(shapes, seed, cfg, path, name):
    rng = np.random.default_rng(seed)
    entries = []
    initial_angles = rng.uniform(0, 2*np.pi, (len(shapes), cfg['eval_angles_per_object']))
    schedules = [(f'one_{d:+d}', [d]) for d in [-15, -5, 0, 5, 15, -10, 10, -30, 30]]
    schedules += [('positive', [15]*8), ('negative', [-15]*8), ('mixed', None),
                  ('inverse', [15, 5, -5, -15])]
    for schedule, degrees in schedules:
        x0, target, actions, object_ids, theta_records = [], [], [], [], []
        for oid, shape in enumerate(shapes):
            theta = initial_angles[oid]
            a = (rng.choice(ACTIONS, (len(theta), 8)) if degrees is None else
                 np.broadcast_to(np.deg2rad(degrees).astype(np.float32), (len(theta), len(degrees))).copy())
            future_angles = theta[:, None] + np.cumsum(a, axis=1)
            x0.append(render(shape, theta))
            target.append(render(shape, future_angles.reshape(-1)).reshape(len(theta), a.shape[1], 1, 32, 32))
            actions.append(a[..., None]); object_ids.extend([oid]*len(theta)); theta_records.append(theta)
        folder = path/name
        folder.mkdir(exist_ok=True)
        # Prediction inputs and withheld targets deliberately live in different files.
        np.savez(folder/f'{schedule}_inputs.npz', x0=np.concatenate(x0), actions=np.concatenate(actions))
        np.savez(folder/f'{schedule}_targets.npz', targets=np.concatenate(target), object_ids=object_ids,
                 theta0=np.concatenate(theta_records))
        entries.append(schedule)
    return entries


def prepare(cfg, path):
    path.mkdir(parents=True, exist_ok=True)
    if (path/'manifest.json').exists():
        raise FileExistsError('Dataset exists; reuse it, do not silently regenerate.')
    geo = geometries(cfg)
    (path/'geometry_evaluator.json').write_text(json.dumps(geo, indent=2))
    make_training(geo['train'], cfg, path)
    sets = [('validation', geo['validation'], cfg['validation_seed']),
            ('test', geo['test'], cfg['test_seed']), ('seen', geo['train'], cfg['seen_seed']),
            ('symmetry2', geo['symmetry2'], 501), ('symmetry4', geo['symmetry4'], 601)]
    schedules = {}
    for name, shapes, seed in sets:
        print('Rendering', name, flush=True)
        schedules[name] = make_evaluation(shapes, seed, cfg, path, name)
    hashes = {str(p.relative_to(path)): digest(p) for p in sorted(path.rglob('*')) if p.is_file()}
    manifest = {'config': cfg, 'files': hashes, 'schedules': schedules, 'rejections': geo['rejections']}
    (path/'manifest.json').write_text(json.dumps(manifest, indent=2))
