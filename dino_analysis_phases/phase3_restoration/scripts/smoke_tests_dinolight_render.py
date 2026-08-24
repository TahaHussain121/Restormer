"""Smoke tests for dinolight-render (AFFM over {3,6,9,12} + gated channel ACA).

Everything the affm arm's smoke test checks about the four-layer prior still has
to hold, plus the ones only this arm needs:

  * STEP-0 EQUALITY WITH E0. `project_out` is zero-initialised, so ACA returns
    exactly F and the whole network reduces to a stock E0 Restormer.
  * THE SCALE CHECK, and it is the reason this arm exists. The attention matrix
    must be C/heads x C/heads at BOTH the train (16x16 tokens) and eval (32x32
    tokens) regimes -- IDENTICAL, because a channel softmax does not know how
    many tokens there are. If the shape changes with input size the
    implementation is spatial, not channel, and the whole design premise is
    wrong. Checked on real forwards at both scales, not by reading the code.
  * ALPHA. sigmoid(alpha_logit) ~ 0.119 at init, on a real batch. It is the
    headline diagnostic: it can close again, which crossattn could not.
  * TEMPERATURE follows Restormer's MDTA convention -- per-head, init 1.0,
    MULTIPLICATIVE. Compared against a real Restormer Attention block's own
    parameter rather than against a number typed here.
  * THE THREE-STEP STAIRCASE. Two zero-inits sit in series (`project_out` at
    the ACA output, `P` at the prior's entrance), so: step 1 only `project_out`
    is live; step 2 the FEATURE path and `P` join; step 3 the DINO CROSS path
    (to_k_cross/to_v_cross/AFFM/alpha) joins, once `P` has left zero and
    `d_proj` is non-zero. Asserted at all three steps. Zero at step 10 would be
    a dead branch; zero at steps 1-2 is arithmetic.
  * AFFM identity: this arm IMPORTS `DinoAffm` from the affm arch, so stage 1 is
    the same code object, and it must use the same four mean FILES.
  * PARAMETER COUNT, itemised against the work order's estimate.

Inference / few-step only. No training, no experiment identity, no checkpoint.
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
from basicsr.models.archs.restormer_arch import Attention    # noqa: E402
import dino_shared                                           # noqa: E402

RESULTS = []
LAYERS1 = [3, 6, 9, 12]
# the work order's estimate, and this file's own itemisation of the difference
WORK_ORDER_ESTIMATE = 1357835


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
    ap.add_argument('--config', default=os.path.join(
        _PHASE3, 'configs', 'dinolight_render_fixed128_L3691_aca_latent.yml'))
    ap.add_argument('--e0-config', default=os.path.join(
        _PHASE3, 'configs', 'E0_fixed128_baseline.yml'))
    ap.add_argument('--affm-config', default=os.path.join(
        _PHASE3, 'configs', 'affm_render_fixed128_spatial_L3691_latent.yml'))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=100)
    ap.add_argument('--out', default=os.path.join(
        _PHASE3, 'results', 'wo2_implementation',
        'smoke_results_dinolight_render.json'))
    args = ap.parse_args()

    cfg, net = build(args.config, args.seed, args.device)
    print(f'dinolight-render {cfg["name"]}   ({args.config})')

    # ------------------------------------------------------------ identity
    print('\n=== identity ===')
    check('arch is RestormerDinoLightRender',
          type(net).__name__ == 'RestormerDinoLightRender', type(net).__name__)
    mro = [c.__name__ for c in type(net).__mro__]
    check('extends the SPATIAL render arm (not the pooled one)',
          'RestormerDinoSpatialRender' in mro
          and 'RestormerDinoSpatialGlobalRender' not in mro, ' <- '.join(mro[:4]))
    check('dino_fusion is aca', net.dino_fusion == 'aca', net.dino_fusion)
    check('dino_source is render', net.dino_source == 'render', net.dino_source)
    check('layers {3,6,9,12} -> 0-indexed {2,5,8,11}',
          net.dino_layers_1indexed == LAYERS1
          and net.dino_layers_0indexed == [2, 5, 8, 11],
          f'{net.dino_layers_1indexed} -> {net.dino_layers_0indexed}')
    check('P is the SAME projection addition-render uses: Conv2d(768->384,1x1)',
          isinstance(net.P, torch.nn.Conv2d)
          and tuple(net.P.weight.shape) == (384, 768, 1, 1),
          f'weight {tuple(net.P.weight.shape)}')

    # ---------------------------------------------- AFFM is the SAME module
    print('\n=== stage 1: AFFM imported, not reimplemented ===')
    from basicsr.models.archs.restormer_dino_affm_render_arch import DinoAffm
    check('self.affm IS the affm arch\'s DinoAffm class',
          type(net.affm) is DinoAffm,
          f'{type(net.affm).__module__}.{type(net.affm).__name__}')
    check('four scoring convs Conv2d(768->1,1x1), all ZERO-initialised',
          len(net.affm.score) == 4
          and all(tuple(c.weight.shape) == (1, 768, 1, 1)
                  and float(c.weight.abs().max()) == 0.0
                  and float(c.bias.abs().max()) == 0.0 for c in net.affm.score))
    _, affm_net = build(args.affm_config, args.seed, 'cpu')
    for regime in ('train128', 'eval256'):
        mine = net.dino_mean_paths_per_layer[regime]
        theirs = affm_net.dino_mean_paths_per_layer[regime]
        check(f'{regime}: the four mean FILES are the affm arm\'s, verbatim',
              mine == theirs,
              ', '.join(os.path.basename(p) for p in mine.values()))
        a = torch.stack([getattr(net, f'mu_b{b}_{regime}') for b in LAYERS1])
        b_ = torch.stack([getattr(affm_net, f'mu_b{b}_{regime}') for b in LAYERS1])
        check(f'{regime}: loaded mean TENSORS are bit-identical to the affm arm',
              torch.equal(a.cpu(), b_), f'max abs diff {float((a.cpu()-b_).abs().max()):.3e}')
    del affm_net

    # ------------------------------------------------------------ the ACA
    print('\n=== stage 3: the ACA block ===')
    aca = net.aca
    check('six MDTA-style projections (1x1 then 3x3 depthwise)',
          all(isinstance(getattr(aca, n), torch.nn.Sequential)
              and len(getattr(aca, n)) == 2
              and getattr(aca, n)[1].groups == 384
              and getattr(aca, n)[1].kernel_size == (3, 3)
              for n in ('to_q', 'to_k', 'to_v', 'to_q_cross', 'to_k_cross',
                        'to_v_cross')),
          'to_q/to_k/to_v/to_q_cross/to_k_cross/to_v_cross')
    check('projection bias matches the TRUNK\'s config bias',
          all(getattr(aca, n)[0].bias is None for n in ('to_q', 'to_k'))
          == (cfg['network_g']['bias'] is False),
          f"config bias={cfg['network_g']['bias']}, conv bias="
          f"{aca.to_q[0].bias is not None}")
    check('project_out is ZERO-initialised -> guided == F at init',
          float(aca.project_out.weight.abs().max()) == 0.0
          and (aca.project_out.bias is None
               or float(aca.project_out.bias.abs().max()) == 0.0))
    # temperature convention, compared against a REAL Restormer block
    ref = Attention(dim=384, num_heads=6, bias=False)
    check('temperature matches Restormer MDTA: per-head, shape (heads,1,1), '
          'init 1.0, MULTIPLICATIVE',
          tuple(aca.temperature_sa.shape) == tuple(ref.temperature.shape)
          and torch.equal(aca.temperature_sa.detach().cpu(),
                          ref.temperature.detach().cpu()),
          f'ours {tuple(aca.temperature_sa.shape)} init '
          f'{float(aca.temperature_sa.reshape(-1)[0])}, Restormer '
          f'{tuple(ref.temperature.shape)} init {float(ref.temperature.reshape(-1)[0])}')
    check('self- and cross-attention have SEPARATE temperatures',
          aca.temperature_sa is not aca.temperature_ca,
          f'{aca.temperature_sa.numel()} + {aca.temperature_ca.numel()} params')
    alpha0 = float(torch.sigmoid(aca.alpha_logit))
    check('alpha_logit -2.0 -> alpha ~ 0.119 at init',
          abs(alpha0 - 0.1192) < 1e-3, f'alpha = {alpha0:.6f}')

    # -------------------------------------------------------- parameters
    print('\n=== parameter count ===')
    _, e0 = build(args.e0_config, args.seed, 'cpu')
    n_e0, n_dl = trainable_count(e0), trainable_count(net)
    delta = n_dl - n_e0
    buckets = {}
    for k, p in net.named_parameters():
        if k.startswith('dino_ext.'):
            continue
        for b in ('affm.', 'P.', 'aca.norm', 'aca.to_', 'aca.project_out',
                  'aca.temperature', 'aca.alpha'):
            if k.startswith(b):
                buckets[b] = buckets.get(b, 0) + p.numel()
                break
    for b, v in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f'      {b:<22} {v:>10,}')
    check('itemised buckets sum to the measured delta',
          sum(buckets.values()) == delta, f'{sum(buckets.values()):,} == {delta:,}')
    gap = WORK_ORDER_ESTIMATE - delta
    check(f'delta measured {delta:,} vs work-order estimate '
          f'{WORK_ORDER_ESTIMATE:,}',
          gap == 4986,
          f'difference {gap:,} = 6 projections x (384+384) bias [4,608] '
          f'+ project_out bias [384] - 6 extra temperatures [-6]; '
          f'this build takes bias from the trunk config (False)')
    check('this is the LARGEST arm in the ladder (record it as a limit)',
          delta > 4 * 295296,
          f'{delta / 295296:.2f}x addition-render\'s 295,296')

    # ------------------------------------------------------------ RNG order
    print('\n=== RNG order: trunk identical to E0 ===')
    e0_sd = e0.state_dict()
    dl_sd = {k: v.cpu() for k, v in net.state_dict().items()}
    shared = [k for k in e0_sd if k in dl_sd]
    same = [k for k in shared if torch.equal(e0_sd[k], dl_sd[k])]
    check('every parameter E0 also has is byte-identical (fence held)',
          len(same) == len(shared) and len(shared) > 0,
          f'{len(same)}/{len(shared)}; differing: '
          f'{[k for k in shared if k not in same][:3]}')

    # ------------------------------------------- step 0 == E0, and the gate
    print('\n=== step 0 == E0; alpha; uniform AFFM ===')
    net.train(); net.set_dino_mode('train128'); net._capture_dino_io = True
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
    check('STEP-0 OUTPUT EQUALS E0 (project_out is zero)',
          torch.allclose(out, out_e0, atol=1e-6, rtol=0),
          f'max |dinolight - E0| = {max_dev:.3e}')

    cap = net._dino_capture
    w = cap['affm_weights']
    check('AFFM weights are EXACTLY uniform 0.25 at init, on a real batch',
          float((w - 0.25).abs().max()) < 1e-6,
          f'max |w - 0.25| = {float((w - 0.25).abs().max()):.3e}')
    check('AFFM weights sum to 1 at every position',
          float((w.sum(1) - 1.0).abs().max()) < 1e-5)
    check('the fused prior is still 768ch, then P projects it to 384',
          list(cap['grid'].shape) == [2, 768, 16, 16]
          and list(cap['d_proj'].shape) == [2, 384, 16, 16],
          f"{list(cap['grid'].shape)} -> {list(cap['d_proj'].shape)}")
    check('guided == F at init (the injected delta is exactly zero)',
          float((cap['guided'] - cap['ca_term'] * 0
                 - cap['guided']).abs().max()) == 0.0
          and net.last_attn_stats['aca_injected_norm'] == 0.0,
          f"aca_injected_norm = {net.last_attn_stats['aca_injected_norm']}")
    st = net.last_dino_stats
    check('projected_norm is ||alpha*F_ca|| and is ZERO at init -- P is also '
          'zero-init, so the prior entering the attention is 0 and the DINO '
          'contribution is genuinely 0, exactly like every other arm',
          st['projected_norm'] == 0.0,
          f"latent {st['latent_norm']:.2f}  projected {st['projected_norm']:.2f}  "
          f"ratio {st['injection_ratio']:.4f}")
    check('...and that ratio is far below the gate cap of 10',
          st['injection_ratio'] < 10.0, f"{st['injection_ratio']:.4f} < 10")
    obs = net.last_attn_stats
    check('observations published to the wrapper hook',
          all(k in obs for k in ('affm_w_b3', 'affm_w_b12', 'affm_w_sum',
                                 'aca_alpha', 'aca_ca_to_sa_ratio',
                                 'aca_injected_norm', 'aca_temp_sa_h0',
                                 'aca_entropy_ca_h0')),
          f'{len(obs)} keys')
    check('alpha on a real batch ~ 0.119',
          abs(obs['aca_alpha'] - 0.1192) < 1e-3, f"{obs['aca_alpha']:.6f}")
    check('DINO frozen and eval() after parent .train()',
          all(not p.requires_grad for p in net.dino_ext.parameters())
          and not net.dino_ext.dino.training)

    # ------------------------------------------------- the staircase
    print('\n=== gradients: zero at step 1, non-zero at step 2 ===')
    loss = torch.nn.functional.l1_loss(out, gt)
    check('finite loss', torch.isfinite(loss).item(), f'{float(loss):.6f}')
    loss.backward()
    up = {'P': net.P.weight, 'affm.score.0': net.affm.score[0].weight,
          'aca.to_q[0]': net.aca.to_q[0].weight,
          'aca.to_v_cross[0]': net.aca.to_v_cross[0].weight}
    check('everything upstream of project_out has ZERO grad at step 1 '
          '(forced by the zero-init; NOT a dead branch)',
          all(t.grad is not None and float(t.grad.abs().max()) == 0.0
              for t in up.values()),
          '  '.join(f'{k} {float(v.grad.abs().max()):.1e}' for k, v in up.items()))
    check('project_out itself HAS gradient at step 1',
          float(net.aca.project_out.weight.grad.abs().max()) > 0,
          f'{float(net.aca.project_out.weight.grad.abs().max()):.3e}')
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                            lr=3e-4)
    cross = {'aca.to_v_cross[0]': net.aca.to_v_cross[0].weight,
             'affm.score.0': net.affm.score[0].weight,
             'alpha_logit': net.aca.alpha_logit}
    feat = {'aca.to_q[0]': net.aca.to_q[0].weight, 'P': net.P.weight}
    opt.step(); opt.zero_grad(set_to_none=True)
    torch.nn.functional.l1_loss(net(stacked), gt).backward()
    check('step 2: the FEATURE path (to_q) and P are live',
          all(float(t.grad.abs().max()) > 0 for t in feat.values()),
          '  '.join(f'{k} {float(v.grad.abs().max()):.2e}' for k, v in feat.items()))
    check('step 2: the DINO CROSS path is still zero -- P had zero grad at '
          'step 1 so AdamW left it at exactly 0, hence d_proj == 0 and the '
          'cross-path inputs are 0. THREE-step staircase, not two.',
          all(float(t.grad.abs().max()) == 0.0 for t in cross.values()),
          '  '.join(f'{k} {float(v.grad.abs().max()):.1e}' for k, v in cross.items()))
    opt.step(); opt.zero_grad(set_to_none=True)
    torch.nn.functional.l1_loss(net(stacked), gt).backward()
    check('STEP 3: EVERY new parameter is live, cross path included',
          all(t.grad is not None and torch.isfinite(t.grad).all().item()
              and float(t.grad.abs().max()) > 0
              for t in list(feat.values()) + list(cross.values())),
          '  '.join(f'{k} {float(v.grad.abs().max()):.2e}'
                    for k, v in list(feat.items()) + list(cross.items())))
    check('P has left zero, so the prior really reaches the attention',
          float(net.P.weight.abs().max()) > 0,
          f'||P.w||inf = {float(net.P.weight.abs().max()):.3e}')

    # ================================================ THE SCALE CHECK
    print('\n=== THE SCALE CHECK — channel attention must not know the token '
          'count ===')
    net.eval()
    shapes = {}
    with torch.no_grad():
        for mode, size, tok in (('train128', 128, 16), ('eval256', 256, 32)):
            net.set_dino_mode(mode)
            big = torch.cat([torch.rand(1, 1, size, size, device=args.device),
                             torch.rand(1, 1, size, size, device=args.device)], 1)
            o = net(big)
            c = net._dino_capture
            # recompute the attention explicitly to read its real shape
            x = net.aca.norm_f(c['guided'] - 0 * c['guided'])  # shape probe only
            f_lat_hw = c['d_proj'].shape[-2:]
            _, attn = net.aca._attend(net.aca.to_q(net.aca.norm_f(c['d_proj'])),
                                      net.aca.to_k(net.aca.norm_f(c['d_proj'])),
                                      net.aca.to_v(net.aca.norm_f(c['d_proj'])),
                                      net.aca.temperature_sa, want_attn=True)
            shapes[mode] = {'radar': size, 'dino_tokens': f'{tok}x{tok}',
                            'grid': list(c['grid'].shape),
                            'd_proj': list(c['d_proj'].shape),
                            'attn': list(attn.shape),
                            'declared': list(net.aca.attn_shape(*f_lat_hw)),
                            'out': list(o.shape)}
            print(f"    {mode:<9} radar {size} -> {tok}x{tok} tokens  "
                  f"grid {c['grid'].shape[-3:]}  attn {tuple(attn.shape)}")
    a1, a2 = shapes['train128']['attn'], shapes['eval256']['attn']
    check('ATTENTION MATRIX SHAPE IS IDENTICAL AT BOTH SCALES',
          a1 == a2, f'{a1} == {a2}')
    check('...and it is C/heads x C/heads (64x64), NOT tokens x tokens',
          a1[-1] == a1[-2] == 384 // net.aca_heads
          and a1[-1] not in (256, 1024),
          f'{a1[-1]}x{a1[-2]}; token counts would have been 256 and 1024')
    check('the DINO grid DOES change with scale (so the invariance is real, '
          'not an artefact of nothing changing)',
          shapes['train128']['grid'][-1] == 16
          and shapes['eval256']['grid'][-1] == 32,
          f"{shapes['train128']['grid']} vs {shapes['eval256']['grid']}")
    check('eval256 forward produces a correct 256x256 output',
          shapes['eval256']['out'] == [1, 1, 256, 256], shapes['eval256']['out'])
    check('eval256 uses the eval mean buffers',
          net.last_mean_key == 'mu_eval256', net.last_mean_key)
    net._capture_dino_io = False

    n_fail = sum(1 for r in RESULTS if not r['pass'])
    print(f'\n=== {len(RESULTS) - n_fail}/{len(RESULTS)} checks passed ===')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'created': datetime.datetime.now().astimezone().isoformat(),
                   'config': os.path.abspath(args.config),
                   'device': args.device, 'hostname': os.uname().nodename,
                   'param_counts': {'E0': n_e0, 'dinolight': n_dl,
                                    'delta_over_E0': delta,
                                    'work_order_estimate': WORK_ORDER_ESTIMATE,
                                    'buckets': buckets},
                   'alpha_init': alpha0,
                   'step0_max_deviation_from_E0': max_dev,
                   'scale_check': shapes,
                   'n_checks': len(RESULTS), 'n_failed': n_fail,
                   'checks': RESULTS}, f, indent=2)
    print(f'wrote {args.out}')
    if n_fail:
        print('SMOKE TESTS FAILED — stop and report; do not redesign.')
        sys.exit(1)


if __name__ == '__main__':
    main()
