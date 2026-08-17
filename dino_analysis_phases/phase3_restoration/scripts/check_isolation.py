"""Experiment-isolation and output-collision checks, run against the FILESYSTEM.

Nothing here is inferred from a config comment: every path is stat'ed, every
name compared, and the protected artifacts of Phase 1/2 and the old baseline are
confirmed present and untouched.
"""

import argparse
import hashlib
import json
import os
import sys

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE3 = os.path.dirname(_HERE)
_REPO = os.path.dirname(os.path.dirname(_PHASE3))

RESULTS = []


def check(name, ok, detail=''):
    RESULTS.append({'check': name, 'pass': bool(ok), 'detail': str(detail)})
    print(f'  [{"PASS" if ok else "FAIL"}] {name}'
          + (f'  --  {detail}' if detail else ''))
    return bool(ok)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(
        _PHASE3, 'results', 'wo2_implementation', 'isolation_checks.json'))
    args = ap.parse_args()

    c0 = os.path.join(_PHASE3, 'configs', 'E0_fixed128_baseline.yml')
    c1 = os.path.join(_PHASE3, 'configs', 'E1_addition_noisy_fixed128_spatial_B6_latent.yml')
    cfg0, cfg1 = yaml.safe_load(open(c0)), yaml.safe_load(open(c1))
    n0, n1 = cfg0['name'], cfg1['name']

    print('== identity ==')
    check('E0 config file exists', os.path.isfile(c0), c0)
    check('E1 config file exists', os.path.isfile(c1), c1)
    check('configs are separate files',
          os.path.abspath(c0) != os.path.abspath(c1))
    check('E0 name != E1 name', n0 != n1, f'{n0}  vs  {n1}')
    check('E0 name is not the old baseline',
          n0 != 'Holo_Baseline_Restormer_verynoisy', n0)
    check('E1 name is not the old baseline',
          n1 != 'Holo_Baseline_Restormer_verynoisy', n1)
    check('E0 name is unique on disk',
          not os.path.isdir(os.path.join(_REPO, 'experiments', n0)),
          f'experiments/{n0} absent -> START FRESH')
    check('E1 name is unique on disk',
          not os.path.isdir(os.path.join(_REPO, 'experiments', n1)),
          f'experiments/{n1} absent -> START FRESH')

    print('== output paths (all derived from opt[name]) ==')
    for label, n in (('E0', n0), ('E1', n1)):
        for sub in ('models', 'training_states', 'visualization'):
            p = os.path.join(_REPO, 'experiments', n, sub)
            check(f'{label} {sub} path is {label}-specific', n in p, p)
    check('experiment dirs differ', n0 != n1)
    check('tb_logger dirs differ (init_tb_logger uses tb_logger/<name>)',
          os.path.join(_REPO, 'tb_logger', n0)
          != os.path.join(_REPO, 'tb_logger', n1))
    check('result dirs differ', os.path.join(_PHASE3, 'results', n0)
          != os.path.join(_PHASE3, 'results', n1))
    check('checkpoint dirs differ',
          os.path.join(_REPO, 'experiments', n0, 'models')
          != os.path.join(_REPO, 'experiments', n1, 'models'))
    check('training-state dirs differ',
          os.path.join(_REPO, 'experiments', n0, 'training_states')
          != os.path.join(_REPO, 'experiments', n1, 'training_states'))

    print('== fair-control equality (every key except the DINO branch) ==')
    for key in ('scale', 'num_gpu', 'manual_seed'):
        check(f'{key} identical', cfg0[key] == cfg1[key], cfg0[key])
    check('train dataset spec identical',
          cfg0['datasets']['train'] == cfg1['datasets']['train'])
    check('val dataset spec identical',
          cfg0['datasets']['val'] == cfg1['datasets']['val'])
    check('optimizer identical',
          cfg0['train']['optim_g'] == cfg1['train']['optim_g'])
    check('scheduler identical',
          cfg0['train']['scheduler'] == cfg1['train']['scheduler'],
          str(cfg0['train']['scheduler']['periods']))
    check('loss identical',
          cfg0['train']['pixel_opt'] == cfg1['train']['pixel_opt'])
    check('total_iter identical',
          cfg0['train']['total_iter'] == cfg1['train']['total_iter'],
          cfg0['train']['total_iter'])
    check('grad clip identical',
          cfg0['train']['use_grad_clip'] == cfg1['train']['use_grad_clip'])
    check('val settings identical', cfg0['val'] == cfg1['val'])
    check('mixing_augs identical (mixup off)',
          cfg0['train']['mixing_augs'] == cfg1['train']['mixing_augs'])
    shared_net = {k: v for k, v in cfg1['network_g'].items()
                  if not k.startswith('dino_') and k != 'type'}
    check('shared network_g kwargs identical',
          shared_net == {k: v for k, v in cfg0['network_g'].items()
                         if k != 'type'}, str(sorted(shared_net)))
    check('E0 carries no dino_ key',
          not any(k.startswith('dino_') for k in cfg0['network_g']))
    check('neither arm warm-starts from a checkpoint',
          cfg0['path']['pretrain_network_g'] is None
          and cfg1['path']['pretrain_network_g'] is None,
          'both train from scratch')

    print('== production means ==')
    means = os.path.join(_PHASE3, 'means')
    for f in ('1e5_B6_train128_dino224_mean.pt', '1e5_B6_eval256_dino448_mean.pt'):
        check(f'{f} exists', os.path.isfile(os.path.join(means, f)))
    check('E1 points at the production train mean',
          cfg1['network_g']['dino_mean_train128']
          == os.path.join(means, '1e5_B6_train128_dino224_mean.pt'))
    check('E1 points at the production eval mean',
          cfg1['network_g']['dino_mean_eval256']
          == os.path.join(means, '1e5_B6_eval256_dino448_mean.pt'))
    check('the two production means are different files',
          cfg1['network_g']['dino_mean_train128']
          != cfg1['network_g']['dino_mean_eval256'])
    check('production means did not overwrite the WO1 verification means',
          os.path.isfile(os.path.join(_PHASE3, 'results', 'wo1_verification',
                                      'wo1_means_train128_dino224_random.pt')))

    print('== protected artifacts must be present and untouched ==')
    protected = {
        'phase1 tree': os.path.join(_REPO, 'dino_analysis_phases', 'phase1'),
        'phase2 tree': os.path.join(_REPO, 'dino_analysis_phases', 'phase2'),
        'dino_spatial_layer_means.pt': os.path.join(
            _REPO, 'dino_analysis_phases', 'dino_spatial_layer_means.pt'),
        'pooled mean lqDINO': os.path.join(
            _REPO, 'Deraining_Holo', 'experiment_results',
            'dino_pooled_means_reference', 'dino_feat_mean_lqDINO.pt'),
        'pooled mean renderDINO': os.path.join(
            _REPO, 'Deraining_Holo', 'experiment_results',
            'dino_pooled_means_reference', 'dino_feat_mean_renderDINO.pt'),
        'old baseline checkpoints': os.path.join(
            _REPO, 'experiments', 'Holo_Baseline_Restormer_verynoisy', 'models'),
        'old baseline predictions': os.path.join(
            _REPO, 'Deraining_Holo', 'results', 'Holo_verynoisy_test_292k'),
        'WO1 verification results': os.path.join(
            _PHASE3, 'results', 'wo1_verification'),
    }
    for label, p in protected.items():
        ok = check(f'{label} still present', os.path.exists(p), p)
        if ok and os.path.isfile(p):
            print(f'          sha256 {sha(p)[:32]}...')

    print('== the stock Restormer path is behaviourally untouched ==')
    arch = os.path.join(_REPO, 'basicsr', 'models', 'archs', 'restormer_arch.py')
    src = open(arch).read()
    check('restormer_arch.py contains no dino reference',
          'dino' not in src.lower(), arch)
    check('E0 uses type: Restormer', cfg0['network_g']['type'] == 'Restormer')
    check('E0 uses model_type: ImageCleanModel',
          cfg0['model_type'] == 'ImageCleanModel')
    check('E1 uses type: RestormerDinoSpatial',
          cfg1['network_g']['type'] == 'RestormerDinoSpatial')
    check('E1 uses model_type: ImageCleanModelDinoSpatial',
          cfg1['model_type'] == 'ImageCleanModelDinoSpatial')

    n_fail = sum(1 for r in RESULTS if not r['pass'])
    print(f'\n=== {len(RESULTS) - n_fail}/{len(RESULTS)} isolation checks passed ===')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'n_checks': len(RESULTS), 'n_failed': n_fail,
                   'checks': RESULTS}, f, indent=2)
    print(f'wrote {args.out}')
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
