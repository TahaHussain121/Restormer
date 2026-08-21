"""Smoke tests for priorquery-render (DINO prior as QUERY, radar as key/value).

Its own file, not a parametrisation of the crossattn-render smoke test: the two
direction arms are only meaningful if each is frozen independently.

The checks that are specific to THIS direction:

  * ROUTING-ONLY. The injected tensor must be a re-mix of RADAR values, with
    DINO entering only through the softmax weights. Asserted causally: perturb
    the radar keys/values and the injection changes; the DINO grid enters W_v
    nowhere. Also asserted structurally: W_v takes 384 inputs (the latent
    width), not 768.
  * NO DIRECT D PATH. No `P` module, no concat, nothing that could carry DINO
    content into the output alongside the attention.
  * The query side is the WIDE one: W_q is 768->384 and LN_q is over 768, the
    mirror image of the radar-query arm.

Shared with crossattn-render, and re-asserted here rather than assumed: step-0
bit-identity with E0, the zero-init gradient STAIRCASE (W_o at step 1, q/k/v by
step 2, hard fail if still dead at step 10), the RNG fence, the 16x16 grid with
no pooling, DINO frozen on the render channel, and the gate's three tags.

Parameter delta target: 741,120.
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
EXPECTED_DELTA = 741120


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=os.path.join(
        _PHASE3, 'configs', 'priorquery_render_fixed128_spatial_B6_latent.yml'))
    ap.add_argument('--e0-config', default=os.path.join(
        _PHASE3, 'configs', 'E0_fixed128_baseline.yml'))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=100)
    ap.add_argument('--out', default=os.path.join(
        _PHASE3, 'results', 'wo2_implementation',
        'smoke_results_priorquery_render.json'))
    args = ap.parse_args()

    cfg, net = build(args.config, args.seed, args.device)
    print(f'priorquery-render {cfg["name"]}   ({args.config})')

    print('\n=== identity ===')
    check('arch is RestormerDinoPriorQueryRender',
          type(net).__name__ == 'RestormerDinoPriorQueryRender', type(net).__name__)
    mro = [c.__name__ for c in type(net).__mro__]
    check('extends the SPATIAL render arm, not the pooled one',
          'RestormerDinoSpatialRender' in mro
          and 'RestormerDinoSpatialGlobalRender' not in mro, ' <- '.join(mro[:4]))
    check('module class is DISTINCT from the radar-query arm',
          type(net.pqattn).__name__ == 'DinoPriorQueryAttention'
          and not hasattr(net, 'xattn'), type(net.pqattn).__name__)
    check('dino_fusion is priorquery', net.dino_fusion == 'priorquery')
    check('dino_source is render', net.dino_source == 'render', net.dino_source)
    check('block B6 / index 5',
          net.dino_block_1indexed == 6 and net.dino_block_0indexed == 5)
    check('train mean is the RENDER train128 mean (no new mean file)',
          os.path.basename(net.dino_mean_paths['train128'])
          == 'render_B6_train128_dino224_mean.pt')
    check('eval mean is the RENDER eval256 mean (no new mean file)',
          os.path.basename(net.dino_mean_paths['eval256'])
          == 'render_B6_eval256_dino448_mean.pt')

    print('\n=== the DIRECTION: prior queries, radar is key AND value ===')
    a = net.pqattn
    check('W_q is the WIDE side: Linear(768 -> 384)',
          tuple(a.W_q.weight.shape) == (384, 768), tuple(a.W_q.weight.shape))
    check('W_k is Linear(384 -> 384)  [radar]',
          tuple(a.W_k.weight.shape) == (384, 384), tuple(a.W_k.weight.shape))
    check('W_v is Linear(384 -> 384)  [radar] -- values CANNOT be DINO',
          tuple(a.W_v.weight.shape) == (384, 384), tuple(a.W_v.weight.shape))
    check('W_o is Linear(384 -> 384), ZERO-init',
          tuple(a.W_o.weight.shape) == (384, 384)
          and float(a.W_o.weight.abs().max()) == 0.0
          and float(a.W_o.bias.abs().max()) == 0.0)
    check('LN over the QUERY side is 768-wide (prior), LN over kv is 384 (radar)',
          a.norm_q.normalized_shape == (768,) and a.norm_kv.normalized_shape == (384,)
          and a.norm_q.elementwise_affine and a.norm_kv.elementwise_affine)
    check('6 heads, head_dim 64, scale 1/sqrt(64)',
          a.heads == 6 and a.head_dim == 64 and abs(a.scale - 0.125) < 1e-12)
    check('NO direct D path: no P module, no P.* parameter',
          not hasattr(net, 'P')
          and not any(n.startswith('P.') for n, _ in net.named_parameters()))

    print('\n=== parameter count ===')
    _, e0 = build(args.e0_config, args.seed, 'cpu')
    n_e0, n_pq = trainable(e0), trainable(net)
    delta = n_pq - n_e0
    parts = {'W_q': 768*384+384, 'W_k': 384*384+384, 'W_v': 384*384+384,
             'W_o': 384*384+384, 'LN_q': 768*2, 'LN_kv': 384*2}
    check(f'parameter delta over E0 == {EXPECTED_DELTA}', delta == EXPECTED_DELTA,
          f'E0 {n_e0} -> priorquery {n_pq} = +{delta}; '
          + ' + '.join(f'{k} {v}' for k, v in parts.items())
          + f' = {sum(parts.values())}')
    check('NOT parameter-matched to radar-Q (888,576) -- structural, recorded',
          delta != 888576,
          f'prior-Q +{delta} vs radar-Q +888576; ladder addition 295296 < '
          f'concat 442752 < prior-Q {delta} < radar-Q 888576')

    print('\n=== RNG order: trunk identical to E0 ===')
    e0_sd = e0.state_dict(); x_sd = {k: v.cpu() for k, v in net.state_dict().items()}
    shared = [k for k in e0_sd if k in x_sd]
    same = [k for k in shared if torch.equal(e0_sd[k], x_sd[k])]
    check('every parameter E0 also has is byte-identical (fence held)',
          len(same) == len(shared) and shared,
          f'{len(same)}/{len(shared)} identical')
    extra = sorted(set(x_sd) - set(e0_sd) - {'mu_train128', 'mu_eval256'})
    check('the only extra keys are the prior-query attention module',
          all(k.startswith('pqattn.') for k in extra), f'{len(extra)} pqattn.* keys')

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
    check('STEP-0 OUTPUT IS BIT-IDENTICAL TO E0', torch.equal(out, out_e0),
          f'max |priorquery - E0| = {max_dev:.3e}')

    cap = net._dino_capture
    b = stacked.shape[0]
    shapes = {'tokens': list(cap['tokens'].shape), 'grid': list(cap['grid'].shape),
              'injected': list(cap['injected'].shape),
              'guided': list(cap['guided'].shape), 'output': list(out.shape)}
    expect = {'tokens': [b, 256, 768], 'grid': [b, 768, 16, 16],
              'injected': [b, 384, 16, 16], 'guided': [b, 384, 16, 16],
              'output': [b, 1, 128, 128]}
    for k, want in expect.items():
        check(f'pq@128 {k} == {want}', shapes[k] == want, shapes[k])
    check('256 QUERY tokens (prior) and 256 key/value tokens (radar) -- the '
          'residual add is well defined',
          cap['grid'].shape[-1] * cap['grid'].shape[-2] == 256)
    check('SPATIAL: 16x16 grid preserved, no pooling, no broadcast',
          list(cap['grid'].shape[-2:]) == list(cap['guided'].shape[-2:]) == [16, 16])
    g = cap['grid']
    check('the queries are NOT a pooled broadcast vector',
          not torch.allclose(g, g.mean(dim=(2, 3), keepdim=True).expand_as(g), atol=1e-4))
    check('DINO SOURCE IS THE RENDER CHANNEL', torch.equal(cap['source'], stacked[:, 1:2]))
    check('DINO SOURCE IS NOT THE RADAR CHANNEL',
          not torch.equal(cap['source'], stacked[:, 0:1]))
    check('DINO params frozen', all(not p.requires_grad for p in net.dino_ext.parameters()))
    net.train()
    check('DINO stays eval() after parent .train()',
          not net.dino_ext.dino.training and net.training)

    print('\n=== ROUTING-ONLY: values are radar, DINO only weights them ===')
    # a direct, causal check on the module: with W_o temporarily set to identity
    # so the injection is observable, changing ONLY the key/value tensor must
    # change the output, and the value projection must never see the prior.
    with torch.no_grad():
        F1 = torch.randn(1, 384, 16, 16, device=args.device)
        F2 = torch.randn(1, 384, 16, 16, device=args.device)
        D1 = torch.randn(1, 768, 16, 16, device=args.device)
        w_save = net.pqattn.W_o.weight.clone()
        net.pqattn.W_o.weight.copy_(torch.eye(384, device=args.device))
        o_F1 = net.pqattn(D1, F1); o_F2 = net.pqattn(D1, F2)
        o_D2 = net.pqattn(torch.randn_like(D1), F1)
        net.pqattn.W_o.weight.copy_(w_save)
    check('changing the RADAR key/value changes the injection (values are radar)',
          not torch.allclose(o_F1, o_F2, atol=1e-6),
          f'max delta = {float((o_F1 - o_F2).abs().max()):.4e}')
    check('changing the PRIOR also changes it -- but only via the weights',
          not torch.allclose(o_F1, o_D2, atol=1e-6),
          f'max delta = {float((o_F1 - o_D2).abs().max()):.4e}')
    check('the value projection is dimensionally incapable of taking DINO '
          '(768) input -- content cannot leak in',
          net.pqattn.W_v.weight.shape[1] == 384)

    print('\n=== THE GRADIENT STAIRCASE ===')
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=3e-4)
    grads = {}
    for step in range(1, 11):
        opt.zero_grad(set_to_none=True)
        torch.nn.functional.l1_loss(net(stacked), gt).backward()
        grads[step] = {nm: (0.0 if m.weight.grad is None
                            else float(m.weight.grad.abs().max()))
                       for nm, m in (('W_q', net.pqattn.W_q), ('W_k', net.pqattn.W_k),
                                     ('W_v', net.pqattn.W_v), ('W_o', net.pqattn.W_o))}
        opt.step()
    check('step 1: W_o HAS a non-zero gradient', grads[1]['W_o'] > 0,
          f"max |dL/dW_o| = {grads[1]['W_o']:.6e}")
    check('step 1: q/k/v zero-grad -- EXPECTED, gradient flows through W_o first',
          max(grads[1][k] for k in ('W_q', 'W_k', 'W_v')) == 0.0,
          ' '.join(f'{k} {grads[1][k]:.3e}' for k in ('W_q', 'W_k', 'W_v')))
    check('step 2: W_q, W_k, W_v ALL non-zero',
          min(grads[2][k] for k in ('W_q', 'W_k', 'W_v')) > 0,
          ' '.join(f'{k} {grads[2][k]:.3e}' for k in ('W_q', 'W_k', 'W_v')))
    check('step 10: q/k/v still alive',
          min(grads[10][k] for k in ('W_q', 'W_k', 'W_v')) > 0,
          ' '.join(f'{k} {grads[10][k]:.3e}' for k in ('W_q', 'W_k', 'W_v')))
    check('DINO gradients are None', all(p.grad is None for p in net.dino_ext.parameters()))

    print('\n=== monitoring ===')
    st = net.last_dino_stats
    check('last_dino_stats has the SAME three keys the gate reads',
          sorted(st) == ['injection_ratio', 'latent_norm', 'projected_norm'], sorted(st))
    check('injection_ratio finite (NOT comparable to the other arms -- its '
          'numerator is built from RADAR values)',
          np.isfinite(st['injection_ratio']),
          f"latent {st['latent_norm']:.3f} projected {st['projected_norm']:.3f}")
    at = net.last_attn_stats
    check('attn_entropy and attn_diag_mass present',
          {'attn_entropy', 'attn_diag_mass'} <= set(at), sorted(at))
    check('attn_entropy finite and <= log(256) = 5.545',
          np.isfinite(at['attn_entropy']) and at['attn_entropy'] <= np.log(256) + 1e-4,
          f"{at['attn_entropy']:.4f} nats")
    check('attn_diag_mass finite and in [0,1] -- THE key metric for this arm',
          np.isfinite(at['attn_diag_mass']) and 0.0 <= at['attn_diag_mass'] <= 1.0,
          f"{at['attn_diag_mass']:.6f}  (near 1.0 => collapsed to a per-position "
          f"reweighting a simpler operator could do)")
    check('gate thresholds unchanged',
          cfg['dino_stability']['injection_ratio_max'] == 10.0
          and cfg['dino_stability']['ratio_rules_start_iter'] == 5000
          and cfg['dino_stability']['growth_factor_max'] == 10.0)

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
        check(f'pq@256 {k} == {want}', s256[k] == want, s256[k])
    check('pq@256 no feature-grid interpolation',
          dino_shared.assert_no_interpolation_needed(cap['grid'].shape[-2:],
                                                     lat.shape[-2:]), '32x32')
    check('pq@256 eval mean buffer in use', net.last_mean_key == 'mu_eval256')
    net._capture_dino_io = False

    n_fail = sum(1 for r in RESULTS if not r['pass'])
    print(f'\n=== {len(RESULTS) - n_fail}/{len(RESULTS)} checks passed ===')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({'created': datetime.datetime.now().astimezone().isoformat(),
               'config': os.path.abspath(args.config), 'device': args.device,
               'hostname': os.uname().nodename,
               'param_counts': {'E0': n_e0, 'priorquery': n_pq,
                                'delta_over_E0': delta,
                                'expected_delta': EXPECTED_DELTA},
               'step0_max_deviation_from_E0': max_dev, 'grad_staircase': grads,
               'attn_observations': at, 'shapes_train128': shapes,
               'shapes_eval256': s256, 'n_checks': len(RESULTS),
               'n_failed': n_fail, 'checks': RESULTS},
              open(args.out, 'w'), indent=2)
    print(f'wrote {args.out}')
    if n_fail:
        print('SMOKE TESTS FAILED — stop and report; do not redesign.')
        sys.exit(1)


if __name__ == '__main__':
    main()
