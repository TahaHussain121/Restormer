"""Smoke tests for the two FINAL multi-level arms, on REAL data.

Isolated: nothing else imports this, and it touches no shared implementation.
Driven by --config; the arm family (addition / ACA) is read from the config.

WHAT IS VERIFIED, for either arm:

  identity   the class the config names; layout 'post_latent+dec3+dec2'; B6;
             render source; the render B6 means in both regimes; projection
             shapes 768->384 / 768->192 / 768->96, every one zero-initialised.
  capacity   added trainable parameters EXACTLY the pre-computed figure, total
             trainable = 26,124,052 + added, frozen DINO reported separately.
  RNG        every trunk tensor byte-identical to E0's for seed 100.
  step 0     output identical to E0's on a real batch.
  one DINO   exactly ONE call into the ViT per forward, in both regimes.
  shapes     train128: pl 384x16x16, d3 192x32x32, d2 96x64x64
             eval256 : pl 384x32x32, d3 192x64x64, d2 96x128x128
             DINO input 224 / 448; eval mean buffer under eval256.
  mapping    each decoder-site prior is the stage's native projection expanded
             by NEAREST-NEIGHBOUR replication (2x / 4x) — checked exactly, after
             training steps have made the projections non-zero.
  frozen     DINO parameters require no grad, stay in eval(), get no gradient.
  learning   a REAL multi-step backward with the recipe's optimiser (AdamW,
             lr 3e-4, wd 1e-4) on real crops:
               addition: every P_s has non-zero gradient on the FIRST backward;
               ACA:      the three-step staircase at EACH stage separately —
                         step 1 only project_out; step 2 P_s and the feature
                         path; step 3 every group including the cross path and
                         alpha. A step-0 identity check alone cannot show this.
  monitoring the gate's three keys, plus per-site norms/ratios/finiteness for
             ALL THREE sites, present and finite.

Inference / few-step only. No experiment identity, no checkpoint.
"""

import argparse
import copy
import datetime
import json
import os
import random
import sys

import cv2
import numpy as np
import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE3 = os.path.dirname(_HERE)
_REPO = os.path.dirname(os.path.dirname(_PHASE3))
for p in (_REPO, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from basicsr.models.archs import define_network              # noqa: E402

DATASET = '/home/woody/iwnt/iwnt174h/thesis_dino/holographic_image_dataset'
E0_TRUNK = 26124052
EXPECTED_ADDED = {'RestormerMultiLevelAdditionRender': 516768,
                  'RestormerMultiLevelAcaRender': 1910535}
LAYOUT = 'post_latent+dec3+dec2'
P_ATTR = {'pl': 'P', 'd3': 'P_dec3', 'd2': 'P_dec2'}
ACA_ATTR = {'pl': 'aca', 'd3': 'aca_dec3', 'd2': 'aca_dec2'}
WIDTH = {'pl': 384, 'd3': 192, 'd2': 96}
RESULTS = []


def check(name, ok, detail=''):
    RESULTS.append({'check': name, 'pass': bool(ok), 'detail': str(detail)})
    print(f'  [{"PASS" if ok else "FAIL"}] {name}'
          + (f'  --  {detail}' if detail else ''), flush=True)
    return bool(ok)


def build(cfg_path, seed, device):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    torch.manual_seed(seed)
    return cfg, define_network(copy.deepcopy(cfg['network_g'])).to(device)


def trainable(net):
    return sum(p.numel() for n, p in net.named_parameters()
               if p.requires_grad and not n.startswith('dino_ext.'))


def real_batch(split, n, crop, seed):
    """Real radar/render/target triplets, radar+render cropped identically."""
    gt_dir = os.path.join(DATASET, f'{split}_clean')
    ids = sorted(f for f in os.listdir(gt_dir) if f.endswith('.png'))
    rng = random.Random(seed)
    radars, renders, gts = [], [], []
    for f in rng.sample(ids, n):
        lq = cv2.imread(os.path.join(DATASET, f'{split}_verynoisy', f),
                        cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.
        gt = cv2.imread(os.path.join(gt_dir, f),
                        cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.
        rd = cv2.imread(os.path.join(DATASET, f'{split}_renders_blackbg', f),
                        cv2.IMREAD_COLOR)[:, :, 0].astype(np.float32) / 255.
        if crop and crop < lq.shape[0]:
            y = rng.randint(0, lq.shape[0] - crop)
            x = rng.randint(0, lq.shape[1] - crop)
            lq, gt, rd = (a[y:y + crop, x:x + crop] for a in (lq, gt, rd))
        radars.append(lq); renders.append(rd); gts.append(gt)
    t = lambda a: torch.from_numpy(np.stack(a))[:, None]
    radar, render, gt = t(radars), t(renders), t(gts)
    return torch.cat([radar, render], 1), radar, gt


def count_dino_calls(net):
    """Wrap the ViT's feature entry point with a call counter."""
    vit = net.dino_ext.dino
    orig = vit.get_intermediate_layers
    box = {'n': 0}

    def counted(*a, **k):
        box['n'] += 1
        return orig(*a, **k)
    vit.get_intermediate_layers = counted
    return box


def gmax(p):
    return 0.0 if p.grad is None else float(p.grad.abs().max())


def aca_groups(net, site):
    """The parameter groups of one ACA stage plus its projection."""
    m = getattr(net, ACA_ATTR[site])
    return {'project_out': m.project_out.weight,
            'to_q': m.to_q[0].weight, 'to_k': m.to_k[0].weight,
            'to_v': m.to_v[0].weight,
            'to_q_cross': m.to_q_cross[0].weight,
            'to_k_cross': m.to_k_cross[0].weight,
            'to_v_cross': m.to_v_cross[0].weight,
            'alpha_logit': m.alpha_logit,
            'P': getattr(net, P_ATTR[site]).weight}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--e0-config', default=os.path.join(
        _PHASE3, 'configs', 'E0_fixed128_baseline.yml'))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=100)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    dev = args.device

    cfg, net = build(args.config, args.seed, dev)
    arch = cfg['network_g']['type']
    is_aca = arch == 'RestormerMultiLevelAcaRender'
    print(f'{cfg["name"]}   [{arch}]   ({args.config})')

    # ------------------------------------------------------------ identity
    print('\n=== identity ===')
    check('built class matches the config', type(net).__name__ == arch,
          type(net).__name__)
    check(f'layout is {LAYOUT}', net.dino_injection == LAYOUT, net.dino_injection)
    check('B6, render source', net.dino_block_1indexed == 6
          and net.dino_source == 'render')
    check('render B6 means in both regimes',
          all('render_B6_' in os.path.basename(net.dino_mean_paths[k])
              for k in ('train128', 'eval256')))
    for s in ('pl', 'd3', 'd2'):
        P = getattr(net, P_ATTR[s])
        check(f'{s}: {P_ATTR[s]} is Conv2d(768 -> {WIDTH[s]}, 1x1) with bias, '
              f'zero-initialised',
              tuple(P.weight.shape) == (WIDTH[s], 768, 1, 1) and P.bias is not None
              and float(P.weight.abs().max()) == 0.0
              and float(P.bias.abs().max()) == 0.0)
        if is_aca:
            a = getattr(net, ACA_ATTR[s])
            check(f'{s}: DinoAca at width {WIDTH[s]}, 6 heads, alpha logit -2, '
                  f'project_out zero',
                  a.dim == WIDTH[s] and a.heads == 6
                  and abs(float(a.alpha_logit) + 2.0) < 1e-6
                  and float(a.project_out.weight.abs().max()) == 0.0)

    # ------------------------------------------------------------ capacity
    print('\n=== capacity ===')
    _, e0 = build(args.e0_config, args.seed, 'cpu')
    n_e0, n_net = trainable(e0), trainable(net)
    added = n_net - n_e0
    n_dino = sum(p.numel() for p in net.dino_ext.parameters())
    check(f'E0 trunk has {E0_TRUNK:,} trainable parameters', n_e0 == E0_TRUNK,
          f'{n_e0:,}')
    check(f'ADDED trainable parameters == {EXPECTED_ADDED[arch]:,}',
          added == EXPECTED_ADDED[arch], f'+{added:,}')
    check(f'TOTAL trainable parameters == {E0_TRUNK + EXPECTED_ADDED[arch]:,}',
          n_net == E0_TRUNK + EXPECTED_ADDED[arch], f'{n_net:,}')
    check('frozen DINO parameters counted separately and all frozen',
          n_dino > 0 and all(not p.requires_grad
                             for p in net.dino_ext.parameters()),
          f'{n_dino:,} frozen')

    # --------------------------------------------------------------- RNG
    e0_sd = e0.state_dict()
    sd = {k: v.cpu() for k, v in net.state_dict().items()}
    shared = [k for k in e0_sd if k in sd]
    same = [k for k in shared if torch.equal(e0_sd[k], sd[k])]
    check('every trunk tensor byte-identical to E0 (seed 100, RNG fence held)',
          len(same) == len(shared) and len(shared) > 0,
          f'{len(same)}/{len(shared)}')

    # ------------------------------------------------ train128, real data
    print('\n=== train128 regime, real crops ===')
    stacked, radar, gt = real_batch('train', 2, 128, seed=0)
    stacked, radar, gt = stacked.to(dev), radar.to(dev), gt.to(dev)
    net.train()
    net.set_dino_mode('train128')
    net._capture_dino_io = True
    calls = count_dino_calls(net)
    out = net(stacked)
    check('exactly ONE DINO extraction per forward (train128)', calls['n'] == 1,
          f'{calls["n"]} call(s)')
    e0 = e0.to(dev).train()
    with torch.no_grad():
        out_e0 = e0(radar)
    dev0 = float((out - out_e0).abs().max())
    check('STEP-0 OUTPUT EQUALS E0 on real data', torch.allclose(
        out, out_e0, atol=1e-6, rtol=0), f'max dev {dev0:.3e}')
    cap = net._dino_capture
    want = {'pl': [2, 384, 16, 16], 'd3': [2, 192, 32, 32], 'd2': [2, 96, 64, 64]}
    for s, shp in want.items():
        check(f'{s}: feature and prior are {shp}',
              list(cap[f'site_{s}_feat'].shape) == shp
              and list(cap[f'site_{s}_prior'].shape) == shp,
              f'{list(cap[f"site_{s}_feat"].shape)} / '
              f'{list(cap[f"site_{s}_prior"].shape)}')
    check('DINO input 224 under train128',
          list(cap['preprocessed'].shape[-2:]) == [224, 224])
    check('DINO source is the render channel',
          torch.equal(cap['source'], stacked[:, 1:2]))
    st = dict(net.last_dino_stats)
    gate_keys = ('latent_norm', 'projected_norm', 'injection_ratio')
    site_keys = [f'site_{s}_{k}' for s in ('pl', 'd3', 'd2')
                 for k in ('feat_norm', 'update_norm', 'ratio', 'finite')]
    check('the gate\'s three keys are present and finite',
          all(k in st and np.isfinite(st[k]) for k in gate_keys))
    check('ALL THREE sites recorded separately: norms, ratio, finiteness',
          all(k in st and np.isfinite(st[k]) for k in site_keys)
          and all(st[f'site_{s}_finite'] == 1.0 for s in ('pl', 'd3', 'd2')))
    check('site observations published to the wrapper hook',
          all(k in net.last_attn_stats for k in site_keys))

    # ----------------------------------------- learning: real multi-step backward
    print('\n=== learning: a real multi-step backward, recipe optimiser ===')
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                            lr=3e-4, weight_decay=1e-4, betas=(0.9, 0.999))
    history = []
    for step in (1, 2, 3):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.l1_loss(net(stacked), gt)
        loss.backward()
        if is_aca:
            history.append({s: {k: gmax(p) for k, p in aca_groups(net, s).items()}
                            for s in ('pl', 'd3', 'd2')})
        else:
            history.append({s: gmax(getattr(net, P_ATTR[s]).weight)
                            for s in ('pl', 'd3', 'd2')})
        opt.step()
        check(f'step {step}: loss finite', torch.isfinite(loss).item(),
              f'{float(loss):.5f}')

    if not is_aca:
        for s in ('pl', 'd3', 'd2'):
            check(f'{s}: {P_ATTR[s]} has NON-ZERO gradient on the FIRST backward',
                  history[0][s] > 0, f'{history[0][s]:.3e}')
        for s in ('pl', 'd3', 'd2'):
            check(f'{s}: {P_ATTR[s]} is non-zero after training steps',
                  float(getattr(net, P_ATTR[s]).weight.abs().max()) > 0)
    else:
        for s in ('pl', 'd3', 'd2'):
            h1, h2, h3 = history[0][s], history[1][s], history[2][s]
            check(f'{s} step 1: ONLY project_out is live '
                  f'(P, cross path and alpha exactly 0)',
                  h1['project_out'] > 0 and h1['P'] == 0.0
                  and h1['to_k_cross'] == 0.0 and h1['to_v_cross'] == 0.0
                  and h1['alpha_logit'] == 0.0,
                  f"project_out {h1['project_out']:.2e}  P {h1['P']:.1e}  "
                  f"alpha {h1['alpha_logit']:.1e}")
            check(f'{s} step 2: P and the feature path are live',
                  h2['P'] > 0 and h2['to_q'] > 0 and h2['to_v'] > 0,
                  f"P {h2['P']:.2e}  to_q {h2['to_q']:.2e}")
            dead = [k for k, v in h3.items() if not v > 0]
            check(f'{s} step 3: EVERY group has non-zero gradient '
                  f'(cross path and alpha included) — no dead branch',
                  not dead, 'dead: ' + (', '.join(dead) if dead else 'none'))

    # ------------------------------------------------ nearest-neighbour mapping
    print('\n=== the decoder-site mapping is exact nearest-neighbour ===')
    with torch.no_grad():
        net(stacked)
    cap = net._dino_capture
    for s, k in (('d3', 2), ('d2', 4)):
        nat = cap[f'{s}_native']
        ref = nat.repeat_interleave(k, 2).repeat_interleave(k, 3)
        check(f'{s}: prior == native projection replicated {k}x{k} '
              f'(and the projection is non-zero, so this is not trivially 0 == 0)',
              torch.equal(cap[f'site_{s}_prior'], ref)
              and float(nat.abs().max()) > 0,
              f'max |native| {float(nat.abs().max()):.3e}')
    check('DINO parameters received no gradient',
          all(p.grad is None for p in net.dino_ext.parameters()))
    check('DINO stays in eval() under parent .train()',
          not net.dino_ext.dino.training and net.training)

    # ------------------------------------------------ eval256, real full frame
    print('\n=== eval256 regime, a real full frame ===')
    big, _, _ = real_batch('val', 1, 0, seed=1)
    big = big.to(dev)
    net.eval()
    net.set_dino_mode('eval256')
    calls['n'] = 0
    with torch.no_grad():
        out256 = net(big)
    cap = net._dino_capture
    check('exactly ONE DINO extraction per forward (eval256)', calls['n'] == 1,
          f'{calls["n"]} call(s)')
    want = {'pl': [1, 384, 32, 32], 'd3': [1, 192, 64, 64], 'd2': [1, 96, 128, 128]}
    for s, shp in want.items():
        check(f'eval256 {s}: feature and prior are {shp}',
              list(cap[f'site_{s}_feat'].shape) == shp
              and list(cap[f'site_{s}_prior'].shape) == shp)
    check('eval256: DINO input 448, native grid 32x32, output 256x256',
          list(cap['preprocessed'].shape[-2:]) == [448, 448]
          and list(cap['grid'].shape[-2:]) == [32, 32]
          and list(out256.shape) == [1, 1, 256, 256])
    check('eval256: the eval mean buffer is in use',
          net.last_mean_key == 'mu_eval256', net.last_mean_key)
    check('eval256: output finite', bool(torch.isfinite(out256).all()))
    net._capture_dino_io = False

    n_fail = sum(1 for r in RESULTS if not r['pass'])
    print(f'\n=== {len(RESULTS) - n_fail}/{len(RESULTS)} checks passed ===')
    out_path = args.out or os.path.join(
        _PHASE3, 'results', 'wo2_implementation',
        f'smoke_results_{cfg["name"]}.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'created': datetime.datetime.now().astimezone().isoformat(),
                   'config': os.path.abspath(args.config), 'arch': arch,
                   'device': dev, 'hostname': os.uname().nodename,
                   'added_trainable': added, 'total_trainable': n_net,
                   'frozen_dino': n_dino, 'step0_max_dev': dev0,
                   'gradient_history': history,
                   'n_checks': len(RESULTS), 'n_failed': n_fail,
                   'checks': RESULTS}, f, indent=2)
    print(f'wrote {out_path}')
    if n_fail:
        print('SMOKE TESTS FAILED -- diagnose the isolated implementation; '
              'do not alter the scientific design.')
        sys.exit(1)


if __name__ == '__main__':
    main()
