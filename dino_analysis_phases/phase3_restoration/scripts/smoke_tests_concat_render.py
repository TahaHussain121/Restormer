"""Smoke tests for concat-render (the FUSION ablation of addition-render).

Everything addition-render's smoke test checks about the render stream still
has to hold -- this file re-checks the ones the fusion change could break --
plus the four that only this arm needs:

  * STEP-0 EQUALITY WITH E0. The arm's output at initialization must equal a
    stock E0 Restormer's output on the same radar tensor, built from E0's own
    config with the same seed. Identity radar half + zero DINO half + zero bias
    means guided == F exactly, so the whole network reduces to E0.
  * NON-ZERO FIRST GRADIENT ON THE DINO HALF. W[:, 384:] starts at zero; that
    must not mean it stays there. Checked on the first backward, together with
    the radar half, so a dead branch cannot pass silently.
  * PARAMETER DELTA == 442,752 exactly, and no `P.*` parameter survives -- the
    residual projection must be GONE, not kept alongside the concatenation.
  * RNG ORDER. Every trunk parameter must be byte-identical to E0's for the
    same seed, which is what proves the fence around `fuse`'s construction
    actually restored the generator state.

And the two properties that define which arm this is:
  * the token grid is preserved through fusion -- NO pooling, NO broadcast;
    the [B,768,g,g] grid enters the concatenation position by position;
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
EXPECTED_DELTA = 442752            # 1152 * 384 + 384


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
        _PHASE3, 'configs', 'concat_render_fixed128_spatial_B6_latent.yml'))
    ap.add_argument('--e0-config', default=os.path.join(
        _PHASE3, 'configs', 'E0_fixed128_baseline.yml'))
    ap.add_argument('--addition-config', default=os.path.join(
        _PHASE3, 'configs', 'E1_addition_render_fixed128_spatial_B6_latent.yml'))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=100)
    ap.add_argument('--out', default=os.path.join(
        _PHASE3, 'results', 'wo2_implementation',
        'smoke_results_concat_render.json'))
    args = ap.parse_args()

    cfg, net = build(args.config, args.seed, args.device)
    print(f'concat-render {cfg["name"]}   ({args.config})')

    # ------------------------------------------------------------ identity
    print('\n=== identity: the right arm, from the right parent ===')
    check('arch is RestormerDinoConcatRender',
          type(net).__name__ == 'RestormerDinoConcatRender', type(net).__name__)
    check('extends the SPATIAL render arm (not the pooled one)',
          'RestormerDinoSpatialRender' in [c.__name__ for c in type(net).__mro__]
          and 'RestormerDinoSpatialGlobalRender' not in
          [c.__name__ for c in type(net).__mro__],
          ' <- '.join(c.__name__ for c in type(net).__mro__[:4]))
    check('dino_fusion is concat', net.dino_fusion == 'concat', net.dino_fusion)
    check('dino_source is render', net.dino_source == 'render', net.dino_source)
    check('block B6 / index 5',
          net.dino_block_1indexed == 6 and net.dino_block_0indexed == 5)
    check('train mean is the RENDER train128 mean (no new mean file)',
          os.path.basename(net.dino_mean_paths['train128'])
          == 'render_B6_train128_dino224_mean.pt',
          os.path.basename(net.dino_mean_paths['train128']))
    check('eval mean is the RENDER eval256 mean (no new mean file)',
          os.path.basename(net.dino_mean_paths['eval256'])
          == 'render_B6_eval256_dino448_mean.pt',
          os.path.basename(net.dino_mean_paths['eval256']))

    # ------------------------------------------------------------- fusion
    print('\n=== fusion block: concat-then-project, addition removed ===')
    check('self.fuse exists and is 1x1 Conv2d(1152 -> 384)',
          isinstance(net.fuse, torch.nn.Conv2d)
          and tuple(net.fuse.weight.shape) == (384, 1152, 1, 1)
          and net.fuse.bias is not None,
          f'weight {tuple(net.fuse.weight.shape)}')
    check('THE RESIDUAL PROJECTION P IS GONE',
          not hasattr(net, 'P')
          and not any(n.startswith('P.') for n, _ in net.named_parameters()),
          'no P module and no P.* parameter')
    eye = torch.eye(384, device=net.fuse.weight.device).reshape(384, 384, 1, 1)
    check('radar half W[:, :384] is IDENTITY at init',
          torch.equal(net.W_F.detach(), eye),
          f'max |W_F - I| = {float((net.W_F.detach() - eye).abs().max()):.3e}')
    check('DINO half W[:, 384:] is ZERO at init',
          float(net.W_D.detach().abs().max()) == 0.0)
    check('bias is ZERO at init',
          float(net.fuse.bias.detach().abs().max()) == 0.0)

    # -------------------------------------------------------- parameters
    print('\n=== parameter count ===')
    _, e0 = build(args.e0_config, args.seed, 'cpu')
    _, add = build(args.addition_config, args.seed, 'cpu')
    n_e0, n_cat, n_add = (trainable_count(e0), trainable_count(net),
                          trainable_count(add))
    delta = n_cat - n_e0
    check(f'parameter delta over E0 == {EXPECTED_DELTA}',
          delta == EXPECTED_DELTA,
          f'E0 {n_e0} -> concat {n_cat} = +{delta} (1152*384 + 384)')
    check('addition-render delta is the expected 295,296 (context, not a target)',
          n_add - n_e0 == 295296, f'+{n_add - n_e0}')
    check('NOT parameter-matched to addition -- recorded, not corrected',
          delta != n_add - n_e0,
          f'concat +{delta} vs addition +{n_add - n_e0}, '
          f'difference {delta - (n_add - n_e0)}')

    # ------------------------------------------------------------ RNG order
    print('\n=== RNG order: trunk weights identical to E0 ===')
    e0_sd = e0.state_dict()
    cat_sd = {k: v.cpu() for k, v in net.state_dict().items()}
    shared = [k for k in e0_sd if k in cat_sd]
    same = [k for k in shared if torch.equal(e0_sd[k], cat_sd[k])]
    check('every parameter E0 also has is byte-identical (fence held)',
          len(same) == len(shared) and len(shared) > 0,
          f'{len(same)}/{len(shared)} tensors identical; '
          f'differing: {[k for k in shared if k not in same][:3]}')
    check('the only extra keys are the fuse conv',
          sorted(set(cat_sd) - set(e0_sd) - {'mu_train128', 'mu_eval256'})
          == ['fuse.bias', 'fuse.weight'],
          sorted(set(cat_sd) - set(e0_sd) - {'mu_train128', 'mu_eval256'}))

    # ------------------------------------- step-0 equality with E0, and grads
    print('\n=== step 0 == E0, and the DINO half still learns ===')
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
    check('STEP-0 OUTPUT EQUALS E0 (identity radar half, zero DINO half)',
          torch.allclose(out, out_e0, atol=1e-6, rtol=0),
          f'max |concat - E0| = {max_dev:.3e}')

    cap = net._dino_capture
    b = stacked.shape[0]
    shapes = {'stacked_input': list(stacked.shape),
              'radar': list(cap['radar'].shape),
              'render': list(cap['render'].shape),
              'dino_input': list(cap['preprocessed'].shape),
              'tokens': list(cap['tokens'].shape),
              'grid': list(cap['grid'].shape),
              'concat': list(cap['concat'].shape),
              'guided': list(cap['guided'].shape),
              'output': list(out.shape)}
    expect = {'stacked_input': [b, 2, 128, 128], 'radar': [b, 1, 128, 128],
              'render': [b, 1, 128, 128], 'dino_input': [b, 3, 224, 224],
              'tokens': [b, 256, 768], 'grid': [b, 768, 16, 16],
              'concat': [b, 1152, 16, 16], 'guided': [b, 384, 16, 16],
              'output': [b, 1, 128, 128]}
    for k, want in expect.items():
        check(f'concat@128 {k} == {want}', shapes[k] == want, shapes[k])

    check('SPATIAL: the token grid survives fusion, position by position',
          list(cap['grid'].shape[-2:]) == list(cap['guided'].shape[-2:]) == [16, 16],
          'no pooling and no broadcast: 16x16 tokens -> 16x16 concat -> 16x16 guided')
    grid = cap['grid']
    pooled = grid.mean(dim=(2, 3), keepdim=True).expand_as(grid)
    check('the fused features are NOT a broadcast global vector',
          not torch.allclose(grid, pooled, atol=1e-4),
          f'max |D - pool(D)| = {float((grid - pooled).abs().max()):.4f}')
    check('DINO SOURCE IS THE RENDER CHANNEL (torch.equal)',
          torch.equal(cap['source'], stacked[:, 1:2]))
    check('DINO SOURCE IS NOT THE RADAR CHANNEL',
          not torch.equal(cap['source'], stacked[:, 0:1]))
    check('DINO params frozen', all(not p.requires_grad
                                    for p in net.dino_ext.parameters()))
    net.train()
    check('DINO stays eval() after parent .train()',
          not net.dino_ext.dino.training and net.training)
    check('centered render features are non-zero',
          float(cap['grid'].abs().max()) > 0,
          f'max |D_centered| = {float(cap["grid"].abs().max()):.4f}')

    loss = torch.nn.functional.l1_loss(out, gt)
    check('finite loss', torch.isfinite(loss).item(), f'{float(loss):.6f}')
    loss.backward()
    g = net.fuse.weight.grad
    gD = g[:, 384:]
    gF = g[:, :384]
    check('DINO HALF W[:, 384:] HAS NON-ZERO GRADIENT ON THE FIRST BACKWARD',
          g is not None and torch.isfinite(gD).all().item()
          and float(gD.abs().max()) > 0,
          f'max |dL/dW_D| = {float(gD.abs().max()):.6e}')
    check('radar half W[:, :384] also has a finite non-zero gradient',
          torch.isfinite(gF).all().item() and float(gF.abs().max()) > 0,
          f'max |dL/dW_F| = {float(gF.abs().max()):.6e}')
    check('DINO gradients are None',
          all(p.grad is None for p in net.dino_ext.parameters()))

    # --------------------------------------------------- the gate's numbers
    print('\n=== monitoring: the addition arms\' three numbers ===')
    st = net.last_dino_stats
    check('last_dino_stats has the SAME three keys the gate reads',
          sorted(st) == ['injection_ratio', 'latent_norm', 'projected_norm'],
          sorted(st))
    check('injection_ratio finite',
          np.isfinite(st['injection_ratio']),
          f"latent {st['latent_norm']:.3f} projected {st['projected_norm']:.3f} "
          f"ratio {st['injection_ratio']:.6e}")
    check('at init, projected_norm == 0 (W_D is zero)',
          st['projected_norm'] == 0.0, st['projected_norm'])
    with torch.no_grad():
        F_lat = net.down3_4(net.encoder_level3(net.down2_3(net.encoder_level2(
            net.down1_2(net.encoder_level1(net.patch_embed(radar)))))))
    check('at init, latent_norm == ||F|| (identity radar half) -- the SAME '
          'quantity addition-render logs',
          abs(st['latent_norm'] - float(F_lat.norm())) < 1e-2,
          f"logged {st['latent_norm']:.4f} vs ||F|| {float(F_lat.norm()):.4f}")
    check('gate thresholds unchanged from addition-render',
          cfg['dino_stability']['injection_ratio_max'] == 10.0
          and cfg['dino_stability']['ratio_rules_start_iter'] == 5000
          and cfg['dino_stability']['growth_factor_max'] == 10.0,
          json.dumps({k: cfg['dino_stability'][k] for k in
                      ('injection_ratio_max', 'ratio_rules_start_iter',
                       'growth_factor_max')}))

    # ------------------------------------------------------------ eval 256
    print('\n=== full-256 eval mode ===')
    net.eval()
    net.set_dino_mode('eval256')
    x = torch.rand(1, 2, 256, 256, device=args.device)
    with torch.no_grad():
        out256 = net(x)
    cap = net._dino_capture
    with torch.no_grad():
        lat = net.down3_4(net.encoder_level3(net.down2_3(net.encoder_level2(
            net.down1_2(net.encoder_level1(net.patch_embed(x[:, 0:1])))))))
    s256 = {'input': list(x.shape), 'dino_input': list(cap['preprocessed'].shape),
            'tokens': list(cap['tokens'].shape), 'grid': list(cap['grid'].shape),
            'concat': list(cap['concat'].shape), 'latent': list(lat.shape),
            'output': list(out256.shape)}
    exp256 = {'input': [1, 2, 256, 256], 'dino_input': [1, 3, 448, 448],
              'tokens': [1, 1024, 768], 'grid': [1, 768, 32, 32],
              'concat': [1, 1152, 32, 32], 'latent': [1, 384, 32, 32],
              'output': [1, 1, 256, 256]}
    for k, want in exp256.items():
        check(f'concat@256 {k} == {want}', s256[k] == want, s256[k])
    check('concat@256 no feature-grid interpolation',
          dino_shared.assert_no_interpolation_needed(cap['grid'].shape[-2:],
                                                     lat.shape[-2:]),
          '32x32 == 32x32')
    check('concat@256 eval mean buffer in use',
          net.last_mean_key == 'mu_eval256', net.last_mean_key)
    net._capture_dino_io = False

    n_fail = sum(1 for r in RESULTS if not r['pass'])
    print(f'\n=== {len(RESULTS) - n_fail}/{len(RESULTS)} checks passed ===')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'created': datetime.datetime.now().astimezone().isoformat(),
                   'config': os.path.abspath(args.config),
                   'device': args.device, 'hostname': os.uname().nodename,
                   'param_counts': {'E0': n_e0, 'concat': n_cat,
                                    'addition': n_add, 'delta_over_E0': delta,
                                    'expected_delta': EXPECTED_DELTA},
                   'step0_max_deviation_from_E0': max_dev,
                   'shapes_train128': shapes, 'shapes_eval256': s256,
                   'n_checks': len(RESULTS), 'n_failed': n_fail,
                   'checks': RESULTS}, f, indent=2)
    print(f'wrote {args.out}')
    if n_fail:
        print('SMOKE TESTS FAILED — stop and report; do not redesign.')
        sys.exit(1)


if __name__ == '__main__':
    main()
