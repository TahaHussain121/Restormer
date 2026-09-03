"""Smoke tests for ANY affm-render layer subset (the additive depth curve).

`smoke_tests_affm_render.py` hard-codes {3,6,9,12}. The middle-point arms
({3,6}, {3,6,9}) reuse the same class with a shorter `dino_layers`, so this file
re-runs the checks that depend on the layer COUNT, reading the set from the
config instead of assuming it:

  * the arm is RestormerDinoAffmRender, source render, fusion affm, and the
    configured set contains B6 and is exactly what the network reports;
  * one scoring conv per layer, all zero-init -> softmax uniform at 1/L;
  * one mean per (layer, regime), all distinct, B6 byte-identical to
    addition-render's;
  * parameter delta over E0 == 295,296 + 769 * L exactly;
  * trunk weights byte-identical to E0 (the RNG fence held);
  * step-0 output equals E0 (P is zero);
  * the one-step gradient staircase: scoring convs get zero gradient on the
    first backward, non-zero on the second, P non-zero on the first;
  * the eval256 regime: 448 input, 32x32 grid, no interpolation.

Inference / one-step only. No training, no experiment identity, no checkpoint.
"""

import argparse
import copy
import datetime
import json
import os
import sys

import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE3 = os.path.dirname(_HERE)
_REPO = os.path.dirname(os.path.dirname(_PHASE3))
for p in (_REPO, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from basicsr.models.archs import define_network              # noqa: E402
import dino_shared                                           # noqa: E402

RESULTS = []
ADDITION_DELTA = 295296
PER_LAYER = 769                      # Conv2d(768 -> 1, 1x1) with bias


def check(name, ok, detail=''):
    RESULTS.append({'check': name, 'pass': bool(ok), 'detail': str(detail)})
    print(f'  [{"PASS" if ok else "FAIL"}] {name}'
          + (f'  --  {detail}' if detail else ''), flush=True)
    return bool(ok)


def build(cfg_path, seed, device):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    torch.manual_seed(seed)
    net = define_network(copy.deepcopy(cfg['network_g'])).to(device)
    return cfg, net


def trainable_count(net):
    return sum(p.numel() for n, p in net.named_parameters()
               if p.requires_grad and not n.startswith('dino_ext.'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--e0-config', default=os.path.join(
        _PHASE3, 'configs', 'E0_fixed128_baseline.yml'))
    ap.add_argument('--addition-config', default=os.path.join(
        _PHASE3, 'configs', 'E1_addition_render_fixed128_spatial_B6_latent.yml'))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=100)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    cfg, net = build(args.config, args.seed, args.device)
    layers = [int(b) for b in cfg['network_g']['dino_layers']]
    L = len(layers)
    print(f'affm subset {cfg["name"]}   layers {layers}   ({args.config})')

    print('\n=== identity ===')
    check('arch is RestormerDinoAffmRender',
          type(net).__name__ == 'RestormerDinoAffmRender', type(net).__name__)
    check('dino_fusion affm / source render',
          net.dino_fusion == 'affm' and net.dino_source == 'render')
    check(f'network reads exactly the configured set {layers}',
          net.dino_layers_1indexed == layers, net.dino_layers_1indexed)
    check('0-indexed set is layers - 1',
          net.dino_layers_0indexed == [b - 1 for b in layers],
          net.dino_layers_0indexed)
    check('set contains B6 and B6 is the reference block',
          6 in layers and net.dino_block_1indexed == 6)
    check('P is Conv2d(768 -> 384, 1x1), zero-init',
          tuple(net.P.weight.shape) == (384, 768, 1, 1)
          and float(net.P.weight.abs().max()) == 0.0
          and float(net.P.bias.abs().max()) == 0.0)
    check(f'{L} scoring convs, each Conv2d(768 -> 1) with bias, zero-init',
          len(net.affm.score) == L
          and all(tuple(c.weight.shape) == (1, 768, 1, 1) and c.bias is not None
                  and float(c.weight.abs().max()) == 0.0
                  and float(c.bias.abs().max()) == 0.0 for c in net.affm.score))

    print('\n=== means ===')
    for regime in ('train128', 'eval256'):
        paths = net.dino_mean_paths_per_layer[regime]
        check(f'{regime}: exactly {L} mean files, all distinct',
              sorted(paths) == sorted(layers) and len(set(paths.values())) == L,
              ', '.join(os.path.basename(p) for p in paths.values()))
        check(f'{regime}: B6 mean byte-identical to addition-render\'s buffer',
              torch.equal(getattr(net, f'mu_b6_{regime}'),
                          getattr(net, f'mu_{regime}')))
        check(f'{regime}: every mean is [768] and finite',
              all(tuple(getattr(net, f'mu_b{b}_{regime}').shape) == (768,)
                  and torch.isfinite(getattr(net, f'mu_b{b}_{regime}')).all()
                  for b in layers))

    print('\n=== parameters and RNG fence ===')
    _, e0 = build(args.e0_config, args.seed, 'cpu')
    _, add = build(args.addition_config, args.seed, 'cpu')
    n_e0, n_net, n_add = trainable_count(e0), trainable_count(net), trainable_count(add)
    delta = n_net - n_e0
    expected = ADDITION_DELTA + PER_LAYER * L
    check(f'parameter delta over E0 == {expected} (295,296 + 769 x {L})',
          delta == expected, f'E0 {n_e0} -> {n_net} = +{delta}')
    check('addition-render delta is 295,296 (context)',
          n_add - n_e0 == ADDITION_DELTA, f'+{n_add - n_e0}')
    e0_sd = e0.state_dict()
    sd = {k: v.cpu() for k, v in net.state_dict().items()}
    shared = [k for k in e0_sd if k in sd]
    same = [k for k in shared if torch.equal(e0_sd[k], sd[k])]
    check('every trunk tensor byte-identical to E0 (fence held)',
          len(same) == len(shared) and shared,
          f'{len(same)}/{len(shared)}')
    buffers = {'mu_train128', 'mu_eval256'} | {
        f'mu_b{b}_{r}' for b in layers for r in ('train128', 'eval256')}
    extra = sorted(set(sd) - set(e0_sd) - buffers)
    want = sorted(['P.bias', 'P.weight'] + [
        f'affm.score.{i}.{w}' for i in range(L) for w in ('bias', 'weight')])
    check('the only extra parameters are P and the scoring convs',
          extra == want, extra)

    print('\n=== step 0, uniform softmax, staircase ===')
    net.train()
    net.set_dino_mode('train128')
    net._capture_dino_io = True
    torch.manual_seed(0)
    radar = torch.rand(2, 1, 128, 128, device=args.device)
    render = torch.rand(2, 1, 128, 128, device=args.device)
    stacked = torch.cat([radar, render], dim=1)
    gt = torch.rand(2, 1, 128, 128, device=args.device)
    out = net(stacked)
    e0 = e0.to(args.device).train()
    with torch.no_grad():
        out_e0 = e0(radar)
    max_dev = float((out - out_e0).abs().max())
    check('STEP-0 OUTPUT EQUALS E0',
          torch.allclose(out, out_e0, atol=1e-6, rtol=0), f'max dev {max_dev:.3e}')
    cap = net._dino_capture
    w = cap['affm_weights']
    check(f'weight map is [2, {L}, 16, 16]', list(w.shape) == [2, L, 16, 16],
          list(w.shape))
    check('weights sum to 1 at every position',
          float((w.sum(dim=1) - 1.0).abs().max()) < 1e-5)
    check(f'softmax exactly uniform at init (1/{L} per layer)',
          float((w - 1.0 / L).abs().max()) < 1e-6,
          f'max |w - 1/{L}| = {float((w - 1.0 / L).abs().max()):.3e}')
    grids = cap['layer_grids']
    check(f'{L} distinct layer grids, each [2, 768, 16, 16]',
          len(grids) == L
          and all(list(g.shape) == [2, 768, 16, 16] for g in grids)
          and min(float((grids[i] - grids[j]).abs().max())
                  for i in range(L) for j in range(i + 1, L)) > 1e-3)
    check('fused output is 768 channels and equals the unweighted mean at init',
          list(cap['grid'].shape) == [2, 768, 16, 16]
          and torch.allclose(cap['grid'], sum(grids) / L, atol=1e-4))
    check('DINO source is the render channel',
          torch.equal(cap['source'], stacked[:, 1:2]))
    check('DINO frozen and in eval()',
          all(not p.requires_grad for p in net.dino_ext.parameters())
          and not net.dino_ext.dino.training)

    st = dict(net.last_dino_stats)
    obs = dict(net.last_attn_stats)
    check('at init projected_norm == 0', st['projected_norm'] == 0.0)
    check(f'published weights are 1/{L} each and sum to 1',
          all(abs(obs[f'affm_w_b{b}'] - 1.0 / L) < 1e-6 for b in layers)
          and abs(obs['affm_w_sum'] - 1.0) < 1e-6, obs)

    loss = torch.nn.functional.l1_loss(out, gt)
    loss.backward()
    gP = net.P.weight.grad
    check('P has non-zero gradient on the first backward',
          gP is not None and float(gP.abs().max()) > 0,
          f'{float(gP.abs().max()):.3e}')
    gs = [c.weight.grad for c in net.affm.score]
    check('scoring convs get ZERO gradient on the first backward (P is zero)',
          all(g is not None and float(g.abs().max()) == 0.0 for g in gs))
    opt = torch.optim.AdamW([p_ for p_ in net.parameters() if p_.requires_grad],
                            lr=3e-4)
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.nn.functional.l1_loss(net(stacked), gt).backward()
    gs2 = [c.weight.grad for c in net.affm.score]
    check('every scoring conv has non-zero gradient on the second backward',
          all(g is not None and torch.isfinite(g).all().item()
              and float(g.abs().max()) > 0 for g in gs2),
          '  '.join(f'B{b} {float(g.abs().max()):.3e}'
                    for b, g in zip(layers, gs2)))
    check('DINO gradients are None',
          all(p.grad is None for p in net.dino_ext.parameters()))

    print('\n=== eval256 regime ===')
    net.set_dino_mode('eval256')
    big = torch.rand(1, 2, 256, 256, device=args.device)
    with torch.no_grad():
        out256 = net(big)
    cap = net._dino_capture
    check('eval256: DINO input 448, weights [1, L, 32, 32], grid 32x32, out 256',
          list(cap['preprocessed'].shape) == [1, 3, 448, 448]
          and list(cap['affm_weights'].shape) == [1, L, 32, 32]
          and list(cap['grid'].shape) == [1, 768, 32, 32]
          and list(out256.shape) == [1, 1, 256, 256])
    check('eval256: eval mean buffers in use', net.last_mean_key == 'mu_eval256')
    check('eval256: weights still sum to 1',
          float((cap['affm_weights'].sum(dim=1) - 1.0).abs().max()) < 1e-5)
    net._capture_dino_io = False

    n_fail = sum(1 for r in RESULTS if not r['pass'])
    print(f'\n=== {len(RESULTS) - n_fail}/{len(RESULTS)} checks passed ===')
    out_path = args.out or os.path.join(
        _PHASE3, 'results', 'wo2_implementation',
        f'smoke_results_{cfg["name"]}.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'created': datetime.datetime.now().astimezone().isoformat(),
                   'config': os.path.abspath(args.config), 'layers': layers,
                   'device': args.device, 'hostname': os.uname().nodename,
                   'param_delta_over_E0': delta, 'expected_delta': expected,
                   'step0_max_deviation_from_E0': max_dev,
                   'n_checks': len(RESULTS), 'n_failed': n_fail,
                   'checks': RESULTS}, f, indent=2)
    print(f'wrote {out_path}')
    if n_fail:
        print('SMOKE TESTS FAILED -- stop and report; do not redesign.')
        sys.exit(1)


if __name__ == '__main__':
    main()
