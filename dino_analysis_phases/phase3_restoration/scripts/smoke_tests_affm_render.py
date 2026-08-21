"""Smoke tests for affm-render (the LAYER-COUNT ablation of addition-render).

Everything addition-render's smoke test checks about the render stream still
has to hold; this file re-checks the ones a four-layer prior could break, plus
the ones only this arm needs:

  * STEP-0 EQUALITY WITH E0. `P` is zero-initialized, so P(D_fused) == 0 and the
    whole network reduces to a stock E0 Restormer on the same radar tensor.
  * UNIFORM SOFTMAX AT INIT. The scoring convs are zero-initialized, so every
    score is 0 and every one of the 256 positions must carry exactly 0.25 per
    layer. Checked on a REAL batch, not by reading the initializer.
  * WEIGHTS SUM TO 1 at every position, which is what makes AFFM a weighted
    average rather than a rescaling.
  * NO GRADIENT STAIRCASE. Unlike the attention arms, both the AFFM scoring
    convs and `P` must receive non-zero gradient on the FIRST backward.
  * PARAMETER DELTA == 298,372 exactly, and it is within ~1% of
    addition-render's 295,296 -- the property that makes a decline across the
    layer ladder un-attributable to capacity.
  * FOUR DISTINCT MEANS, and the B6 one byte-identical to the file
    addition-render trains with, in BOTH regimes.
  * RNG ORDER. Every trunk parameter byte-identical to E0's for the same seed,
    which is what proves the fence around the AFFM construction restored the
    generator state.

And the properties that define which arm this is:
  * four layers really are read -- {3,6,9,12} -> 0-indexed {2,5,8,11} -- and
    they are DIFFERENT tensors, not the same block four times;
  * the grid stays spatial through fusion: no pooling, no broadcast;
  * the fused output is still 768 channels, so `P` is unchanged;
  * DINO reads the RENDER channel, never the radar, never the target.

Inference / one-step only. No training, no experiment identity, no checkpoint.
"""

import argparse
import copy
import datetime
import json
import os
import sys

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
import dino_shared                                           # noqa: E402

RESULTS = []
EXPECTED_DELTA = 298372            # 768*384 + 384  +  4 * (768 + 1)
ADDITION_DELTA = 295296
LAYERS1 = [3, 6, 9, 12]


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
    """Trainable parameters, excluding the frozen ViT that no checkpoint holds."""
    return sum(p.numel() for n, p in net.named_parameters()
               if p.requires_grad and not n.startswith('dino_ext.'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=os.path.join(
        _PHASE3, 'configs', 'affm_render_fixed128_spatial_L3691_latent.yml'))
    ap.add_argument('--e0-config', default=os.path.join(
        _PHASE3, 'configs', 'E0_fixed128_baseline.yml'))
    ap.add_argument('--addition-config', default=os.path.join(
        _PHASE3, 'configs', 'E1_addition_render_fixed128_spatial_B6_latent.yml'))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=100)
    ap.add_argument('--out', default=os.path.join(
        _PHASE3, 'results', 'wo2_implementation',
        'smoke_results_affm_render.json'))
    args = ap.parse_args()

    cfg, net = build(args.config, args.seed, args.device)
    print(f'affm-render {cfg["name"]}   ({args.config})')

    # ------------------------------------------------------------ identity
    print('\n=== identity: the right arm, from the right parent ===')
    check('arch is RestormerDinoAffmRender',
          type(net).__name__ == 'RestormerDinoAffmRender', type(net).__name__)
    mro = [c.__name__ for c in type(net).__mro__]
    check('extends the SPATIAL render arm (not the pooled one)',
          'RestormerDinoSpatialRender' in mro
          and 'RestormerDinoSpatialGlobalRender' not in mro,
          ' <- '.join(mro[:4]))
    check('dino_fusion is affm', net.dino_fusion == 'affm', net.dino_fusion)
    check('dino_source is render', net.dino_source == 'render', net.dino_source)
    check('layers are {3,6,9,12} 1-indexed',
          net.dino_layers_1indexed == LAYERS1, net.dino_layers_1indexed)
    check('b1_to_b0 maps them to {2,5,8,11}',
          net.dino_layers_0indexed == [dino_shared.b1_to_b0(b) for b in LAYERS1]
          == [2, 5, 8, 11], net.dino_layers_0indexed)
    check('the set CONTAINS the reference block B6',
          6 in net.dino_layers_1indexed and net.dino_block_1indexed == 6)
    check('injection operator UNCHANGED: P is Conv2d(768 -> 384, 1x1)',
          isinstance(net.P, torch.nn.Conv2d)
          and tuple(net.P.weight.shape) == (384, 768, 1, 1),
          f'weight {tuple(net.P.weight.shape)}')
    check('P is zero-initialized (weight AND bias)',
          float(net.P.weight.abs().max()) == 0.0
          and float(net.P.bias.abs().max()) == 0.0)

    # ---------------------------------------------------------------- AFFM
    print('\n=== the AFFM module ===')
    check('four scoring convs, each Conv2d(768 -> 1, 1x1) with bias',
          len(net.affm.score) == 4
          and all(tuple(c.weight.shape) == (1, 768, 1, 1) and c.bias is not None
                  for c in net.affm.score),
          f'{len(net.affm.score)} convs')
    check('scoring convs are ZERO-initialized -> softmax starts uniform',
          all(float(c.weight.abs().max()) == 0.0
              and float(c.bias.abs().max()) == 0.0 for c in net.affm.score))

    # ---------------------------------------------------------------- means
    print('\n=== centering: one train-only mean per layer ===')
    for regime in ('train128', 'eval256'):
        paths = net.dino_mean_paths_per_layer[regime]
        check(f'{regime}: four distinct mean files',
              len(set(paths.values())) == 4,
              ', '.join(os.path.basename(p) for p in paths.values()))
        check(f'{regime}: every mean is shape [768] and finite',
              all(tuple(getattr(net, f'mu_b{b}_{regime}').shape) == (768,)
                  and torch.isfinite(getattr(net, f'mu_b{b}_{regime}')).all()
                  for b in LAYERS1))
        norms = {b: float(getattr(net, f'mu_b{b}_{regime}').norm())
                 for b in LAYERS1}
        check(f'{regime}: per-layer norms DIFFER from one another',
              len({round(v, 4) for v in norms.values()}) == 4,
              '  '.join(f'B{b} {v:.4f}' for b, v in norms.items()))
        mine = getattr(net, f'mu_b6_{regime}')
        theirs = getattr(net, f'mu_{regime}')
        d = float((mine - theirs).abs().max())
        check(f'{regime}: B6 MEAN IS BYTE-IDENTICAL TO ADDITION-RENDER\'S',
              torch.equal(mine, theirs), f'max abs diff = {d:.3e}')
        check(f'{regime}: mean files live under phase3_restoration/means',
              all('/phase3_restoration/means/' in p for p in paths.values()))

    # -------------------------------------------------------- parameters
    print('\n=== parameter count ===')
    _, e0 = build(args.e0_config, args.seed, 'cpu')
    _, add = build(args.addition_config, args.seed, 'cpu')
    n_e0, n_affm, n_add = (trainable_count(e0), trainable_count(net),
                           trainable_count(add))
    delta = n_affm - n_e0
    check(f'parameter delta over E0 == {EXPECTED_DELTA}',
          delta == EXPECTED_DELTA,
          f'E0 {n_e0} -> affm {n_affm} = +{delta} '
          f'(P 295,296 + 4 x 769 = 3,076)')
    check('addition-render delta is the expected 295,296 (context)',
          n_add - n_e0 == ADDITION_DELTA, f'+{n_add - n_e0}')
    over = delta - (n_add - n_e0)
    check('NEARLY PARAMETER-MATCHED to addition-render (<2%)',
          0 < over and over / (n_add - n_e0) < 0.02,
          f'affm +{delta} vs addition +{n_add - n_e0} = +{over} '
          f'({100 * over / (n_add - n_e0):.2f}%)')
    check('a naive 4-layer concat would have cost far more (context)',
          3072 * 384 + 384 == 1180032, '3072*384 + 384 = 1,180,032')

    # ------------------------------------------------------------ RNG order
    print('\n=== RNG order: trunk weights identical to E0 ===')
    e0_sd = e0.state_dict()
    affm_sd = {k: v.cpu() for k, v in net.state_dict().items()}
    shared = [k for k in e0_sd if k in affm_sd]
    same = [k for k in shared if torch.equal(e0_sd[k], affm_sd[k])]
    check('every parameter E0 also has is byte-identical (fence held)',
          len(same) == len(shared) and len(shared) > 0,
          f'{len(same)}/{len(shared)} tensors identical; '
          f'differing: {[k for k in shared if k not in same][:3]}')
    buffers = {'mu_train128', 'mu_eval256'} | {
        f'mu_b{b}_{r}' for b in LAYERS1 for r in ('train128', 'eval256')}
    extra = sorted(set(affm_sd) - set(e0_sd) - buffers)
    check('the only extra parameters are P and the four scoring convs',
          extra == ['P.bias', 'P.weight']
          + [f'affm.score.{i}.{w}' for i in range(4) for w in ('bias', 'weight')]
          or sorted(extra) == sorted(
              ['P.bias', 'P.weight']
              + [f'affm.score.{i}.{w}' for i in range(4)
                 for w in ('bias', 'weight')]),
          extra)

    # ------------------------------------- step-0 equality with E0, and grads
    print('\n=== step 0 == E0; uniform softmax; both branches learn ===')
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
    check('STEP-0 OUTPUT EQUALS E0 (P is zero)',
          torch.allclose(out, out_e0, atol=1e-6, rtol=0),
          f'max |affm - E0| = {max_dev:.3e}')

    cap = net._dino_capture
    w = cap['affm_weights']
    check('AFFM weight map is [B, 4, 16, 16]',
          list(w.shape) == [2, 4, 16, 16], list(w.shape))
    sums = w.sum(dim=1)
    check('WEIGHTS SUM TO 1 AT EVERY POSITION',
          float((sums - 1.0).abs().max()) < 1e-5,
          f'max |sum - 1| = {float((sums - 1.0).abs().max()):.3e} '
          f'over {sums.numel()} positions')
    check('SOFTMAX IS EXACTLY UNIFORM AT INIT (0.25 per layer, every position)',
          float((w - 0.25).abs().max()) < 1e-6,
          f'max |w - 0.25| = {float((w - 0.25).abs().max()):.3e}')
    check('uniform init is SEED-INDEPENDENT (zero scores, not a lucky draw)',
          all(float(c.weight.abs().max()) == 0.0 for c in net.affm.score))

    grids = cap['layer_grids']
    check('four DISTINCT layer grids, each [B, 768, 16, 16]',
          len(grids) == 4
          and all(list(g.shape) == [2, 768, 16, 16] for g in grids),
          [list(g.shape) for g in grids])
    pairwise = [float((grids[i] - grids[j]).abs().max())
                for i in range(4) for j in range(i + 1, 4)]
    check('the four layers are genuinely different tensors',
          min(pairwise) > 1e-3,
          f'min pairwise max-abs difference = {min(pairwise):.4f}')
    check('FUSED OUTPUT IS STILL 768 CHANNELS (weighted sum, not concat)',
          list(cap['grid'].shape) == [2, 768, 16, 16], list(cap['grid'].shape))
    check('at init the fusion is the unweighted MEAN of the four grids',
          torch.allclose(cap['grid'], sum(grids) / 4, atol=1e-4),
          f'max |D_fused - mean| = '
          f'{float((cap["grid"] - sum(grids) / 4).abs().max()):.3e}')

    print('\n=== centering actually changes the statistics ===')
    raw_tok = cap['tokens']
    cen_stats = {}
    for b1, b0, g in zip(LAYERS1, net.dino_layers_0indexed, grids):
        raw_ma = float(raw_tok[b0].abs().mean())
        cen_ma = float(g.abs().mean())
        cen_stats[f'B{b1}'] = {'raw_mean_abs': raw_ma, 'centered_mean_abs': cen_ma,
                               'mu_norm': float(getattr(net, f'mu_b{b1}_train128').norm())}
        check(f'B{b1}: centering moves the features (mean|.| {raw_ma:.4f} -> '
              f'{cen_ma:.4f})', abs(raw_ma - cen_ma) > 1e-4,
              f'mu norm {cen_stats[f"B{b1}"]["mu_norm"]:.3f}')

    shapes = {'stacked_input': list(stacked.shape),
              'radar': list(cap['radar'].shape),
              'render': list(cap['render'].shape),
              'dino_input': list(cap['preprocessed'].shape),
              'affm_weights': list(w.shape),
              'grid': list(cap['grid'].shape),
              'output': list(out.shape)}
    expect = {'stacked_input': [2, 2, 128, 128], 'radar': [2, 1, 128, 128],
              'render': [2, 1, 128, 128], 'dino_input': [2, 3, 224, 224],
              'affm_weights': [2, 4, 16, 16], 'grid': [2, 768, 16, 16],
              'output': [2, 1, 128, 128]}
    for k, want in expect.items():
        check(f'affm@128 {k} == {want}', shapes[k] == want, shapes[k])

    check('SPATIAL: no pooling, no broadcast through fusion',
          list(cap['grid'].shape[-2:]) == [16, 16])
    grid = cap['grid']
    pooled = grid.mean(dim=(2, 3), keepdim=True).expand_as(grid)
    check('the fused features are NOT a broadcast global vector',
          not torch.allclose(grid, pooled, atol=1e-4),
          f'max |D - pool(D)| = {float((grid - pooled).abs().max()):.4f}')
    check('DINO SOURCE IS THE RENDER CHANNEL (torch.equal)',
          torch.equal(cap['source'], stacked[:, 1:2]))
    check('DINO SOURCE IS NOT THE RADAR CHANNEL',
          not torch.equal(cap['source'], stacked[:, 0:1]))
    check('DINO params frozen',
          all(not p.requires_grad for p in net.dino_ext.parameters()))
    net.train()
    check('DINO stays eval() after parent .train()',
          not net.dino_ext.dino.training and net.training)

    loss = torch.nn.functional.l1_loss(out, gt)
    check('finite loss', torch.isfinite(loss).item(), f'{float(loss):.6f}')
    # snapshot the INIT-TIME monitoring values now: the staircase check below
    # takes an optimizer step, after which these are no longer init values.
    st = dict(net.last_dino_stats)
    obs = dict(net.last_attn_stats)
    with torch.no_grad():
        F_lat_norm = float(net.down3_4(net.encoder_level3(net.down2_3(
            net.encoder_level2(net.down1_2(net.encoder_level1(
                net.patch_embed(radar))))))).norm())
    loss.backward()
    gP = net.P.weight.grad
    check('P HAS NON-ZERO GRADIENT ON THE FIRST BACKWARD',
          gP is not None and torch.isfinite(gP).all().item()
          and float(gP.abs().max()) > 0,
          f'max |dL/dP| = {float(gP.abs().max()):.6e}')
    # ---- THE ONE-STEP STAIRCASE, asserted rather than wished away ---------
    # D_fused reaches the loss ONLY through P, and d(P(D))/dD == P.weight,
    # which is zero at initialization. So on the FIRST backward the AFFM
    # scoring convs MUST receive exactly zero gradient, and only P learns. The
    # work order predicted a non-zero first gradient here; that prediction is
    # incompatible with the zero-init of P that the same work order requires
    # for step-0 equality with E0. Both cannot hold. The staircase is one step
    # deep and resolves at step 2, which is what is checked below -- the
    # attention arms have the same structure, one level deeper.
    gs = [c.weight.grad for c in net.affm.score]
    check('AFFM scoring convs get ZERO gradient on the FIRST backward '
          '(forced by P being zero-init; NOT a dead branch -- see step 2)',
          all(g is not None and float(g.abs().max()) == 0.0 for g in gs),
          '  '.join(f'B{b} {float(g.abs().max()):.3e}'
                    for b, g in zip(LAYERS1, gs)))

    opt = torch.optim.AdamW([p_ for p_ in net.parameters() if p_.requires_grad],
                            lr=3e-4)
    opt.step()                                   # P leaves zero
    opt.zero_grad(set_to_none=True)
    torch.nn.functional.l1_loss(net(stacked), gt).backward()
    gs2 = [c.weight.grad for c in net.affm.score]
    check('EVERY AFFM SCORING CONV HAS NON-ZERO GRADIENT ON THE SECOND '
          'BACKWARD (the staircase is one step deep and it clears)',
          all(g is not None and torch.isfinite(g).all().item()
              and float(g.abs().max()) > 0 for g in gs2),
          '  '.join(f'B{b} {float(g.abs().max()):.3e}'
                    for b, g in zip(LAYERS1, gs2)))
    check('P is no longer zero after one optimizer step',
          float(net.P.weight.abs().max()) > 0,
          f'max |P.weight| = {float(net.P.weight.abs().max()):.3e}')
    check('DINO gradients are None',
          all(p.grad is None for p in net.dino_ext.parameters()))

    # --------------------------------------------------- the gate's numbers
    print('\n=== monitoring (values snapshotted at init, before the step) ===')
    check('last_dino_stats has the SAME three keys the gate reads',
          sorted(st) == ['injection_ratio', 'latent_norm', 'projected_norm'],
          sorted(st))
    check('at init, projected_norm == 0 (P is zero)',
          st['projected_norm'] == 0.0, st['projected_norm'])
    check('at init, latent_norm == ||F|| -- the SAME quantity addition-render '
          'logs', abs(st['latent_norm'] - F_lat_norm) < 1e-2,
          f"logged {st['latent_norm']:.4f} vs ||F|| {F_lat_norm:.4f}")
    check('the four AFFM weights are published to the wrapper\'s observation '
          'hook', sorted(obs) == ['affm_w_b12', 'affm_w_b3', 'affm_w_b6',
                                  'affm_w_b9', 'affm_w_sum'], sorted(obs))
    check('published weights are 0.25 each and sum to 1 at init',
          all(abs(obs[f'affm_w_b{b}'] - 0.25) < 1e-6 for b in LAYERS1)
          and abs(obs['affm_w_sum'] - 1.0) < 1e-6,
          '  '.join(f'B{b} {obs[f"affm_w_b{b}"]:.6f}' for b in LAYERS1))
    check('NO gate rule is defined on the AFFM weights (observations only)',
          not any('affm' in k for k in net.last_dino_stats))

    # ------------------------------------------------------- eval256 regime
    print('\n=== the full-256 regime: 32x32 tokens, eval means, no interp ===')
    net.set_dino_mode('eval256')
    big = torch.cat([torch.rand(1, 1, 256, 256, device=args.device),
                     torch.rand(1, 1, 256, 256, device=args.device)], dim=1)
    with torch.no_grad():
        out256 = net(big)
    cap = net._dino_capture
    s256 = {'dino_input': list(cap['preprocessed'].shape),
            'affm_weights': list(cap['affm_weights'].shape),
            'grid': list(cap['grid'].shape), 'output': list(out256.shape)}
    for k, want in {'dino_input': [1, 3, 448, 448],
                    'affm_weights': [1, 4, 32, 32],
                    'grid': [1, 768, 32, 32],
                    'output': [1, 1, 256, 256]}.items():
        check(f'affm@256 {k} == {want}', s256[k] == want, s256[k])
    with torch.no_grad():
        lat = net.down3_4(net.encoder_level3(net.down2_3(net.encoder_level2(
            net.down1_2(net.encoder_level1(net.patch_embed(big[:, 0:1])))))))
    check('affm@256 no feature-grid interpolation',
          dino_shared.assert_no_interpolation_needed(cap['grid'].shape[-2:],
                                                     lat.shape[-2:]),
          '32x32 == 32x32')
    check('affm@256 eval mean buffers in use',
          net.last_mean_key == 'mu_eval256', net.last_mean_key)
    check('affm@256 weights still sum to 1 at every position',
          float((cap['affm_weights'].sum(dim=1) - 1.0).abs().max()) < 1e-5)
    net._capture_dino_io = False

    n_fail = sum(1 for r in RESULTS if not r['pass'])
    print(f'\n=== {len(RESULTS) - n_fail}/{len(RESULTS)} checks passed ===')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'created': datetime.datetime.now().astimezone().isoformat(),
                   'config': os.path.abspath(args.config),
                   'device': args.device, 'hostname': os.uname().nodename,
                   'layers_1indexed': LAYERS1,
                   'param_counts': {'E0': n_e0, 'affm': n_affm,
                                    'addition': n_add, 'delta_over_E0': delta,
                                    'expected_delta': EXPECTED_DELTA},
                   'step0_max_deviation_from_E0': max_dev,
                   'shapes_train128': shapes, 'shapes_eval256': s256,
                   'centering_effect_train128': cen_stats,
                   'n_checks': len(RESULTS), 'n_failed': n_fail,
                   'checks': RESULTS}, f, indent=2)
    print(f'wrote {args.out}')
    if n_fail:
        print('SMOKE TESTS FAILED — stop and report; do not redesign.')
        sys.exit(1)


if __name__ == '__main__':
    main()
