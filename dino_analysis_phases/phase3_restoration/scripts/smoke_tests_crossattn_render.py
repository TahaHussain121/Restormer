"""Smoke tests for crossattn-render (the cross-attention fusion arm).

THE GRADIENT RULE HERE IS DIFFERENT FROM EVERY OTHER ARM, ON PURPOSE.

W_o is zero-initialised at the OUTPUT of the fusion, so on the FIRST backward
the gradient into W_q, W_k and W_v is exactly zero -- it has to travel through
W_o, which is still zero. The step-1 assertion used for the concat arm WOULD
FAIL HERE AND SHOULD. What is asserted instead is the staircase:

    step 0   output bit-identical to E0
    step 1   W_o has a non-zero gradient; q/k/v are expected to be zero
    step 2   W_q, W_k, W_v all have non-zero gradients
    step 10  if q/k/v are still dead, something IS broken -> FAIL

Also checked: parameter delta 888,576 exactly; the RNG fence (every trunk weight
byte-identical to E0); radar is the QUERY and DINO the key/value; the token grid
is preserved with no pooling, no broadcast and no CLS; the gate's three tags are
still produced; and the attention observations (entropy, diagonal mass) are
finite and correctly bounded.

Inference / a few optimizer steps on random tensors only. No experiment
identity, no dataset write, no checkpoint.
"""

import argparse, copy, datetime, json, os, sys
import numpy as np, torch, yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE3 = os.path.dirname(_HERE)
_REPO = os.path.dirname(os.path.dirname(_PHASE3))
for p in (_REPO, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
from basicsr.models.archs import define_network              # noqa: E402
import dino_shared                                           # noqa: E402

RESULTS = []
EXPECTED_DELTA = 888576


def check(name, ok, detail=''):
    RESULTS.append({'check': name, 'pass': bool(ok), 'detail': str(detail)})
    print(f'  [{"PASS" if ok else "FAIL"}] {name}'
          + (f'  --  {detail}' if detail else ''), flush=True)
    return bool(ok)


def build(cfg_path, seed, device):
    cfg = yaml.safe_load(open(cfg_path))
    torch.manual_seed(seed)
    return cfg, define_network(copy.deepcopy(cfg['network_g'])).to(device)


def trainable(net):
    return sum(p.numel() for n, p in net.named_parameters()
               if p.requires_grad and not n.startswith('dino_ext.'))


def peak_vram(cfg_path, seed, device, steps=3):
    """Peak allocated VRAM for one training-shaped step at batch 8, crop 128."""
    if device != 'cuda':
        return float('nan')
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    _, net = build(cfg_path, seed, device)
    net.train(); net.set_dino_mode('train128')
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                            lr=3e-4)
    torch.manual_seed(0)
    x = torch.rand(8, 2, 128, 128, device=device)
    gt = torch.rand(8, 1, 128, 128, device=device)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        torch.nn.functional.l1_loss(net(x), gt).backward()
        opt.step()
    peak = torch.cuda.max_memory_allocated() / 2**20
    del net, opt, x, gt
    torch.cuda.empty_cache()
    return peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=os.path.join(
        _PHASE3, 'configs', 'crossattn_render_fixed128_spatial_B6_latent.yml'))
    ap.add_argument('--e0-config', default=os.path.join(
        _PHASE3, 'configs', 'E0_fixed128_baseline.yml'))
    ap.add_argument('--addition-config', default=os.path.join(
        _PHASE3, 'configs', 'E1_addition_render_fixed128_spatial_B6_latent.yml'))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=100)
    ap.add_argument('--out', default=os.path.join(
        _PHASE3, 'results', 'wo2_implementation',
        'smoke_results_crossattn_render.json'))
    args = ap.parse_args()

    cfg, net = build(args.config, args.seed, args.device)
    print(f'crossattn-render {cfg["name"]}   ({args.config})')

    print('\n=== identity ===')
    check('arch is RestormerDinoCrossAttnRender',
          type(net).__name__ == 'RestormerDinoCrossAttnRender', type(net).__name__)
    mro = [c.__name__ for c in type(net).__mro__]
    check('extends the SPATIAL render arm, not the pooled one',
          'RestormerDinoSpatialRender' in mro
          and 'RestormerDinoSpatialGlobalRender' not in mro,
          ' <- '.join(mro[:4]))
    check('dino_fusion is crossattn', net.dino_fusion == 'crossattn')
    check('dino_source is render', net.dino_source == 'render', net.dino_source)
    check('block B6 / index 5',
          net.dino_block_1indexed == 6 and net.dino_block_0indexed == 5)
    check('train mean is the RENDER train128 mean (no new mean file)',
          os.path.basename(net.dino_mean_paths['train128'])
          == 'render_B6_train128_dino224_mean.pt')
    check('eval mean is the RENDER eval256 mean (no new mean file)',
          os.path.basename(net.dino_mean_paths['eval256'])
          == 'render_B6_eval256_dino448_mean.pt')
    check('the residual projection P is GONE',
          not hasattr(net, 'P')
          and not any(n.startswith('P.') for n, _ in net.named_parameters()))

    print('\n=== the attention module ===')
    xa = net.xattn
    check('6 heads, head_dim 64, scale 1/sqrt(64)',
          xa.heads == 6 and xa.head_dim == 64
          and abs(xa.scale - 1 / 8.0) < 1e-12,
          f'heads {xa.heads} head_dim {xa.head_dim} scale {xa.scale:.6f}')
    for nm, mod, shape in (('W_q', xa.W_q, (384, 384)), ('W_k', xa.W_k, (384, 768)),
                           ('W_v', xa.W_v, (384, 768)), ('W_o', xa.W_o, (384, 384))):
        check(f'{nm} is Linear{shape[::-1]}', tuple(mod.weight.shape) == shape,
              tuple(mod.weight.shape))
    check('W_o is ZERO-init (weight and bias)',
          float(xa.W_o.weight.abs().max()) == 0.0
          and float(xa.W_o.bias.abs().max()) == 0.0)
    check('q/k/v are NOT zero-init (they must be ordinary Linear draws)',
          min(float(m.weight.abs().max()) for m in (xa.W_q, xa.W_k, xa.W_v)) > 0)
    check('pre-norm LayerNorms, elementwise_affine=True, dims 384 and 768',
          xa.norm_q.normalized_shape == (384,) and xa.norm_q.elementwise_affine
          and xa.norm_kv.normalized_shape == (768,) and xa.norm_kv.elementwise_affine)

    print('\n=== parameter count ===')
    _, e0 = build(args.e0_config, args.seed, 'cpu')
    _, add = build(args.addition_config, args.seed, 'cpu')
    n_e0, n_x, n_add = trainable(e0), trainable(net), trainable(add)
    delta = n_x - n_e0
    parts = {'W_q': 384*384+384, 'W_k': 768*384+384, 'W_v': 768*384+384,
             'W_o': 384*384+384, 'LN_q': 384*2, 'LN_kv': 768*2}
    check(f'parameter delta over E0 == {EXPECTED_DELTA}', delta == EXPECTED_DELTA,
          f'E0 {n_e0} -> crossattn {n_x} = +{delta}; '
          + ' + '.join(f'{k} {v}' for k, v in parts.items())
          + f' = {sum(parts.values())}')
    check('capacity ladder: addition 295,296 < concat 442,752 < crossattn',
          (n_add - n_e0) == 295296 and delta > 442752,
          f'addition +{n_add-n_e0}, concat +442752, crossattn +{delta} '
          f'-- NOT parameter-matched; recorded, not corrected')

    print('\n=== RNG order: trunk identical to E0 ===')
    e0_sd = e0.state_dict()
    x_sd = {k: v.cpu() for k, v in net.state_dict().items()}
    shared = [k for k in e0_sd if k in x_sd]
    same = [k for k in shared if torch.equal(e0_sd[k], x_sd[k])]
    check('every parameter E0 also has is byte-identical (fence held)',
          len(same) == len(shared) and shared,
          f'{len(same)}/{len(shared)} identical; differing '
          f'{[k for k in shared if k not in same][:3]}')
    extra = sorted(set(x_sd) - set(e0_sd) - {'mu_train128', 'mu_eval256'})
    check('the only extra keys are the attention module',
          all(k.startswith('xattn.') for k in extra), f'{len(extra)} keys, '
          f'all xattn.*' if extra else 'none')

    print('\n=== step 0 == E0 ===')
    net.train(); net.set_dino_mode('train128'); net._capture_dino_io = True
    torch.manual_seed(0)
    radar = torch.rand(2, 1, 128, 128, device=args.device)
    render = torch.rand(2, 1, 128, 128, device=args.device)
    stacked = torch.cat([radar, render], 1)
    gt = torch.rand(2, 1, 128, 128, device=args.device)
    out = net(stacked)
    e0 = e0.to(args.device).train()
    with torch.no_grad():
        out_e0 = e0(radar)
    max_dev = float((out - out_e0).abs().max())
    check('STEP-0 OUTPUT IS BIT-IDENTICAL TO E0',
          torch.equal(out, out_e0),
          f'max |crossattn - E0| = {max_dev:.3e}')

    cap = net._dino_capture
    b = stacked.shape[0]
    shapes = {'dino_input': list(cap['preprocessed'].shape),
              'tokens': list(cap['tokens'].shape), 'grid': list(cap['grid'].shape),
              'injected': list(cap['injected'].shape),
              'guided': list(cap['guided'].shape), 'output': list(out.shape)}
    expect = {'dino_input': [b, 3, 224, 224], 'tokens': [b, 256, 768],
              'grid': [b, 768, 16, 16], 'injected': [b, 384, 16, 16],
              'guided': [b, 384, 16, 16], 'output': [b, 1, 128, 128]}
    for k, want in expect.items():
        check(f'xattn@128 {k} == {want}', shapes[k] == want, shapes[k])
    check('SPATIAL: 16x16 token grid preserved end to end, no pooling',
          list(cap['grid'].shape[-2:]) == list(cap['guided'].shape[-2:]) == [16, 16])
    g = cap['grid']
    check('keys/values are NOT a pooled broadcast vector',
          not torch.allclose(g, g.mean(dim=(2, 3), keepdim=True).expand_as(g),
                             atol=1e-4),
          f'max |D - pool(D)| = {float((g - g.mean(dim=(2,3), keepdim=True)).abs().max()):.4f}')
    check('256 queries and 256 keys -> square attention, diag_mass defined',
          cap['grid'].shape[-1] * cap['grid'].shape[-2] == 256)
    check('DINO SOURCE IS THE RENDER CHANNEL', torch.equal(cap['source'], stacked[:, 1:2]))
    check('DINO SOURCE IS NOT THE RADAR CHANNEL',
          not torch.equal(cap['source'], stacked[:, 0:1]))
    check('DINO params frozen', all(not p.requires_grad for p in net.dino_ext.parameters()))
    net.train()
    check('DINO stays eval() after parent .train()',
          not net.dino_ext.dino.training and net.training)

    print('\n=== THE GRADIENT STAIRCASE (this arm differs from the others) ===')
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=3e-4)
    grads = {}
    for step in range(1, 11):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.l1_loss(net(stacked), gt)
        loss.backward()
        grads[step] = {nm: (0.0 if m.weight.grad is None
                            else float(m.weight.grad.abs().max()))
                       for nm, m in (('W_q', net.xattn.W_q), ('W_k', net.xattn.W_k),
                                     ('W_v', net.xattn.W_v), ('W_o', net.xattn.W_o))}
        opt.step()
    check('step 1: W_o HAS a non-zero gradient', grads[1]['W_o'] > 0,
          f"max |dL/dW_o| = {grads[1]['W_o']:.6e}")
    check('step 1: q/k/v are zero-grad -- EXPECTED, not a bug '
          '(gradient must pass through the zero W_o first)',
          max(grads[1][k] for k in ('W_q', 'W_k', 'W_v')) == 0.0,
          ' '.join(f'{k} {grads[1][k]:.3e}' for k in ('W_q', 'W_k', 'W_v')))
    check('step 2: W_q, W_k, W_v ALL have non-zero gradients',
          min(grads[2][k] for k in ('W_q', 'W_k', 'W_v')) > 0,
          ' '.join(f'{k} {grads[2][k]:.3e}' for k in ('W_q', 'W_k', 'W_v')))
    check('step 10: q/k/v still alive (dead here would mean something IS broken)',
          min(grads[10][k] for k in ('W_q', 'W_k', 'W_v')) > 0,
          ' '.join(f'{k} {grads[10][k]:.3e}' for k in ('W_q', 'W_k', 'W_v')))
    check('DINO gradients are None', all(p.grad is None for p in net.dino_ext.parameters()))

    print('\n=== monitoring ===')
    st = net.last_dino_stats
    check('last_dino_stats has the SAME three keys the gate reads',
          sorted(st) == ['injection_ratio', 'latent_norm', 'projected_norm'], sorted(st))
    check('injection_ratio finite',
          np.isfinite(st['injection_ratio']),
          f"latent {st['latent_norm']:.3f} projected {st['projected_norm']:.3f} "
          f"ratio {st['injection_ratio']:.4e}")
    a = net.last_attn_stats
    check('attention observations present: attn_entropy, attn_diag_mass',
          {'attn_entropy', 'attn_diag_mass'} <= set(a), sorted(a))
    check('attn_entropy finite and <= log(256) = 5.545',
          np.isfinite(a['attn_entropy']) and a['attn_entropy'] <= np.log(256) + 1e-4,
          f"{a['attn_entropy']:.4f} nats (uniform = {a['attn_uniform_entropy']:.4f})")
    check('attn_diag_mass finite and in [0,1]',
          np.isfinite(a['attn_diag_mass']) and 0.0 <= a['attn_diag_mass'] <= 1.0,
          f"{a['attn_diag_mass']:.6f}  (1.0 would mean attention reduced itself "
          f"to the addition arm -- a RESULT, not a failure)")
    check('gate thresholds unchanged from addition-render',
          cfg['dino_stability']['injection_ratio_max'] == 10.0
          and cfg['dino_stability']['ratio_rules_start_iter'] == 5000
          and cfg['dino_stability']['growth_factor_max'] == 10.0)
    check('NO gate rule on entropy or diag_mass (observations only)',
          not any('entropy' in str(v) or 'diag' in str(v)
                  for v in cfg['dino_stability'].values()))

    print('\n=== full-256 eval mode ===')
    net.eval(); net.set_dino_mode('eval256')
    x = torch.rand(1, 2, 256, 256, device=args.device)
    with torch.no_grad():
        o256 = net(x)
    cap = net._dino_capture
    with torch.no_grad():
        lat = net.down3_4(net.encoder_level3(net.down2_3(net.encoder_level2(
            net.down1_2(net.encoder_level1(net.patch_embed(x[:, 0:1])))))))
    s256 = {'tokens': list(cap['tokens'].shape), 'grid': list(cap['grid'].shape),
            'latent': list(lat.shape), 'output': list(o256.shape)}
    e256 = {'tokens': [1, 1024, 768], 'grid': [1, 768, 32, 32],
            'latent': [1, 384, 32, 32], 'output': [1, 1, 256, 256]}
    for k, want in e256.items():
        check(f'xattn@256 {k} == {want}', s256[k] == want, s256[k])
    check('xattn@256 no feature-grid interpolation',
          dino_shared.assert_no_interpolation_needed(cap['grid'].shape[-2:],
                                                     lat.shape[-2:]), '32x32')
    check('xattn@256 eval mean buffer in use', net.last_mean_key == 'mu_eval256')
    net._capture_dino_io = False
    del net, e0, add
    if args.device == 'cuda':
        torch.cuda.empty_cache()

    # PEAK VRAM IS MEASURED ELSEWHERE, ON PURPOSE. Measuring two arms in one
    # process does not work -- the first arm's optimizer keeps its parameters
    # alive, so the second measurement inherits the first's allocations, which
    # is what exhausted a 32 GB V100 in job 1783386. It now lives in
    # scripts/measure_peak_vram.py, one fresh process per arm, run on the a100
    # the arms actually train on: scripts/run_peak_vram_compare.sh
    v_add = v_x = ratio = float('nan')

    n_fail = sum(1 for r in RESULTS if not r['pass'])
    print(f'\n=== {len(RESULTS) - n_fail}/{len(RESULTS)} checks passed ===')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({'created': datetime.datetime.now().astimezone().isoformat(),
               'config': os.path.abspath(args.config), 'device': args.device,
               'hostname': os.uname().nodename,
               'param_counts': {'E0': n_e0, 'crossattn': n_x, 'addition': n_add,
                                'delta_over_E0': delta,
                                'expected_delta': EXPECTED_DELTA},
               'step0_max_deviation_from_E0': max_dev,
               'grad_staircase': grads,
               'attn_observations': a,
               'peak_vram_mib': {'addition': v_add, 'crossattn': v_x,
                                 'ratio': ratio},
               'shapes_train128': shapes, 'shapes_eval256': s256,
               'n_checks': len(RESULTS), 'n_failed': n_fail,
               'checks': RESULTS}, open(args.out, 'w'), indent=2)
    print(f'wrote {args.out}')
    if n_fail:
        print('SMOKE TESTS FAILED — stop and report; do not redesign.')
        sys.exit(1)


if __name__ == '__main__':
    main()
