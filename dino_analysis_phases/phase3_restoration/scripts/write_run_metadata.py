"""Write `run_metadata.json` for one run. NEVER overwrites an existing file.

Called by the training SLURM scripts immediately before basicsr/train.py, so
each job records the exact identity it ran under. A second job for the same
experiment (an auto-resume) writes `run_metadata_resume_<n>.json` instead of
touching the original -- history is appended to, never mutated.

The config HASH is recorded so a later edit to a frozen YAML is detectable.
"""

import argparse
import datetime
import hashlib
import json
import os
import socket
import subprocess
import sys

import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE3 = os.path.dirname(_HERE)
_REPO = os.path.dirname(os.path.dirname(_PHASE3))


def git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       cwd=_REPO).decode().strip()
    except Exception:                                       # noqa: BLE001
        return 'unknown'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--run-id', default=os.environ.get('SLURM_JOB_ID', 'local'))
    args = ap.parse_args()

    with open(args.config, 'rb') as f:
        raw = f.read()
    cfg = yaml.safe_load(raw.decode())
    name = cfg['name']

    exp_dir = os.path.join(_REPO, 'experiments', name)
    states = os.path.join(exp_dir, 'training_states')
    prior = sorted(int(f.split('.')[0]) for f in os.listdir(states)
                   if f.endswith('.state')) if os.path.isdir(states) else []
    resume = bool(prior)

    out_dir = os.path.join(_PHASE3, 'results', name, 'metadata')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'run_metadata.json')
    if os.path.isfile(out):
        n = 1
        while os.path.isfile(os.path.join(out_dir, f'run_metadata_resume_{n}.json')):
            n += 1
        out = os.path.join(out_dir, f'run_metadata_resume_{n}.json')

    net = cfg['network_g']
    ds = cfg['datasets']['train']
    rec = {
        'experiment_name': name,
        'run_id': args.run_id,
        'config_path': os.path.abspath(args.config),
        'config_sha256': hashlib.sha256(raw).hexdigest(),
        'git_commit': git_commit(),
        'start_time': datetime.datetime.now().astimezone().isoformat(),
        'end_time': None,
        'hostname': socket.gethostname(),
        'gpu': (torch.cuda.get_device_name(0) if torch.cuda.is_available()
                else 'none'),
        'cuda_device': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'torch': torch.__version__,
        'seed': cfg['manual_seed'],
        'splits': {'train': ds['dataroot_lq'],
                   'train_gt': ds['dataroot_gt'],
                   'val': cfg['datasets']['val']['dataroot_lq'],
                   'val_gt': cfg['datasets']['val']['dataroot_gt']},
        'crop_protocol': {'gt_size': ds['gt_size'], 'gt_sizes': ds['gt_sizes'],
                          'mini_batch_sizes': ds['mini_batch_sizes'],
                          'iters': ds['iters'],
                          'description': 'fixed 128x128 crop, constant batch 8'},
        'total_iter': cfg['train']['total_iter'],
        'batch_size_per_gpu': ds['batch_size_per_gpu'],
        'optimizer': cfg['train']['optim_g'],
        'scheduler': cfg['train']['scheduler'],
        'loss': cfg['train']['pixel_opt'],
        'checkpoint_selection_metric': 'validation PSNR (8-bit path); the test '
                                       'split never drives selection',
        'fresh_or_resume': 'AUTO-RESUME' if resume else 'START FRESH',
        'resume_from_iter': prior[-1] if prior else None,
        'checkpoint_used': (os.path.join(exp_dir, 'models',
                                         f'net_g_{prior[-1]}.pth')
                            if prior else None),
        'arch': net['type'],
    }
    if net['type'] == 'RestormerDinoSpatial':
        rec['dino'] = {
            'model': net['dino_model'],
            'block_1indexed': net['dino_block'],
            'block_0indexed': net['dino_block'] - 1,
            'frozen': net['dino_frozen'],
            'source': net['dino_source'],
            'injection': net['dino_injection'],
            'projection': net['dino_projection'],
            'init': net['dino_init'],
            'mean_train128': net['dino_mean_train128'],
            'mean_eval256': net['dino_mean_eval256'],
            'default_mode': net['dino_default_mode'],
        }
        rec['stability_gate'] = cfg.get('dino_stability')

    with open(out, 'w') as f:
        json.dump(rec, f, indent=2)
    print(f'{rec["fresh_or_resume"]}  {name}  run {args.run_id}')
    print(f'wrote {out}')
    if resume:
        print(f'  resuming from iter {prior[-1]} '
              f'({len(prior)} training states on disk)')


if __name__ == '__main__':
    main()
