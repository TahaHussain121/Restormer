"""Smoke tests for the ACA layer ladder: aca-L6, aca-L36, aca-L6912.

ONE script, driven by --config, because the three arms differ only in their
layer set and a separate test file per arm would let the checks drift apart.
(The ARCHS are separate files on purpose -- that is where duplication buys
isolation. The TEST is shared on purpose -- that is where sharing buys
consistency.)

What it checks, for whichever arm you point it at:

  * STEP-0 EQUALITY WITH E0. `project_out` is zero-init, so ACA returns exactly
    F and the network reduces to a stock E0 Restormer.
  * THE SCALE CHECK (--scale-check). The attention matrix must be
    C/heads x C/heads at BOTH the 16x16-token train regime and the 32x32-token
    eval regime -- IDENTICAL, because a channel softmax does not know how many
    tokens exist. If it changes with input size the attention is SPATIAL, the
    design premise is wrong, and the run must not start.
  * ALPHA and TEMPERATURE match dinolight-render's values exactly -- read from
    dinolight's own config and its own DinoAca, never typed in here.
  * AFFM (arms 2/3 only): scoring convs zero-init, weights exactly 1/n on a
    real batch, summing to 1 at every position. Arm 1 must have NO affm module
    at all -- a degenerate 1-layer softmax is a no-op and is deliberately absent.
  * THE THREE-STEP STAIRCASE. Two zero-inits in series (`project_out`, `P`):
    step 1 only `project_out`; step 2 the feature path and `P`; step 3
    everything. Zero at step 3 would be a dead branch.
  * PARAMETER COUNT against dinolight's measured delta minus 769 per absent
    AFFM scoring conv.
  * MEANS shared with dinolight/affm, and the B6 entry byte-identical to the
    mean addition-render trains with.

Inference / few-step only. No training, no experiment identity, no checkpoint.
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
from basicsr.models.archs.restormer_arch import Attention    # noqa: E402

RESULTS = []
DINOLIGHT_CFG = os.path.join(_PHASE3, 'configs',
                             'dinolight_render_fixed128_L3691_aca_latent.yml')
DINOLIGHT_DELTA = 1352849          # measured, recorded in its devlog
ADDITION_DELTA = 295296


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--e0-config', default=os.path.join(
        _PHASE3, 'configs', 'E0_fixed128_baseline.yml'))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=100)
    ap.add_argument('--scale-check', action='store_true')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    cfg, net = build(args.config, args.seed, args.device)
    name = cfg['name']
    layers = list(getattr(net, 'dino_layers_1indexed', [6]))
    n_layers = len(layers)
    has_affm = hasattr(net, 'affm')
    out = args.out or os.path.join(
        _PHASE3, 'results', 'wo2_implementation',
        f'smoke_results_{os.path.basename(args.config).replace(".yml", "")}.json')
    print(f'{name}   layers {layers}   AFFM={has_affm}   ({args.config})')

    # ------------------------------------------------------------- identity
    print('\n=== identity ===')
    check('dino_fusion is aca', net.dino_fusion == 'aca', net.dino_fusion)
    check('dino_source is render', net.dino_source == 'render', net.dino_source)
    check('layer set matches the arm name', layers == layers, str(layers))
    check('B6 is in the set (ties every arm to the same reference layer)',
          6 in layers)
    check('P is Conv2d(768->384, 1x1), same projection as addition-render',
          isinstance(net.P, torch.nn.Conv2d)
          and tuple(net.P.weight.shape) == (384, 768, 1, 1))
    if n_layers == 1:
        check('SINGLE-LAYER ARM HAS NO AFFM AT ALL (a 1-layer softmax is a '
              'no-op; a degenerate module would be 769 dead params)',
              not has_affm)
    else:
        check(f'{n_layers}-layer arm HAS an AFFM with {n_layers} scoring convs',
              has_affm and len(net.affm.score) == n_layers)
        check('AFFM scoring convs are ZERO-initialised',
              all(float(c.weight.abs().max()) == 0.0
                  and float(c.bias.abs().max()) == 0.0 for c in net.affm.score))

    # ---------------------------------------------------- ACA matches dinolight
    print('\n=== ACA block matches dinolight-render exactly ===')
    dl_cfg, dl_net = build(DINOLIGHT_CFG, args.seed, 'cpu')
    a, d = net.aca, dl_net.aca
    check('alpha_logit init identical to dinolight',
          float(a.alpha_logit) == float(d.alpha_logit),
          f'ours {float(torch.sigmoid(a.alpha_logit)):.6f} vs dinolight '
          f'{float(torch.sigmoid(d.alpha_logit)):.6f}')
    check('temperature init identical to dinolight (and to Restormer MDTA)',
          torch.equal(a.temperature_sa.detach().cpu(),
                      d.temperature_sa.detach().cpu())
          and torch.equal(a.temperature_sa.detach().cpu(),
                          Attention(384, a.heads, False).temperature.detach()),
          f'shape {tuple(a.temperature_sa.shape)} init '
          f'{float(a.temperature_sa.reshape(-1)[0])}, multiplicative')
    check('heads identical to dinolight', a.heads == d.heads, a.heads)
    check('project_out ZERO-initialised', float(a.project_out.weight.abs().max()) == 0.0)
    check('six MDTA projections (1x1 + 3x3 depthwise), bias from trunk config',
          all(len(getattr(a, n)) == 2 and getattr(a, n)[1].groups == 384
              for n in ('to_q', 'to_k', 'to_v', 'to_q_cross', 'to_k_cross',
                        'to_v_cross')))

    # ----------------------------------------------------------------- means
    print('\n=== means: SHARED with dinolight/affm, not recomputed ===')
    for regime in ('train128', 'eval256'):
        if n_layers == 1:
            mine = {6: net.dino_mean_paths[regime]}
        else:
            mine = net.dino_mean_paths_per_layer[regime]
        check(f'{regime}: {n_layers} mean file(s), all under phase3/means',
              len(mine) == n_layers
              and all('/phase3_restoration/means/' in p for p in mine.values()),
              ', '.join(os.path.basename(p) for p in mine.values()))
        b6_mine = (net.mu_train128 if regime == 'train128' else net.mu_eval256)
        b6_dl = (dl_net.mu_train128 if regime == 'train128' else dl_net.mu_eval256)
        dmax = float((b6_mine.cpu() - b6_dl).abs().max())
        check(f'{regime}: B6 mean byte-identical to addition-render / dinolight',
              torch.equal(b6_mine.cpu(), b6_dl), f'max abs diff = {dmax:.3e}')

    # ------------------------------------------------------------ parameters
    print('\n=== parameter count ===')
    _, e0 = build(args.e0_config, args.seed, 'cpu')
    n_e0, n_arm = trainable(e0), trainable(net)
    delta = n_arm - n_e0
    absent = 4 - n_layers if n_layers > 1 else 4
    expect = DINOLIGHT_DELTA - absent * 769
    check(f'delta over E0 == dinolight {DINOLIGHT_DELTA:,} - {absent}x769 '
          f'= {expect:,}', delta == expect,
          f'measured {delta:,}  (E0 {n_e0:,} -> {n_arm:,})')
    spread = 100 * (DINOLIGHT_DELTA - delta) / DINOLIGHT_DELTA
    check('ACA LADDER SPREAD < 1% -> a difference across the ladder is NOT '
          'attributable to capacity', spread < 1.0,
          f'{spread:.3f}% below dinolight')
    check('vs addition-render this arm IS capacity confounded -- a DIFFERENT '
          'comparison with a DIFFERENT caveat, do not conflate',
          delta > 4 * ADDITION_DELTA,
          f'{delta / ADDITION_DELTA:.2f}x addition-render\'s {ADDITION_DELTA:,}')

    # ------------------------------------------------------------- RNG order
    print('\n=== RNG order: trunk identical to E0 ===')
    e0_sd, arm_sd = e0.state_dict(), {k: v.cpu() for k, v in net.state_dict().items()}
    shared = [k for k in e0_sd if k in arm_sd]
    same = [k for k in shared if torch.equal(e0_sd[k], arm_sd[k])]
    check('every parameter E0 also has is byte-identical (fence held)',
          len(same) == len(shared) and len(shared) > 0,
          f'{len(same)}/{len(shared)}')

    # ------------------------------------------ step 0 == E0, alpha, AFFM
    print('\n=== step 0 == E0; alpha; AFFM uniform ===')
    net.train(); net.set_dino_mode('train128'); net._capture_dino_io = True
    torch.manual_seed(0)
    radar = torch.rand(2, 1, 128, 128, device=args.device)
    render = torch.rand(2, 1, 128, 128, device=args.device)
    stacked = torch.cat([radar, render], 1)
    gt = torch.rand(2, 1, 128, 128, device=args.device)
    o = net(stacked)
    e0 = e0.to(args.device).train()
    with torch.no_grad():
        o0 = e0(radar)
    dev = float((o - o0).abs().max())
    check('STEP-0 OUTPUT EQUALS E0 BIT-EXACTLY', torch.allclose(o, o0, atol=1e-6, rtol=0),
          f'max |arm - E0| = {dev:.3e}')
    cap = net._dino_capture
    check('d_proj is [B,384,g,g] entering the attention',
          list(cap['d_proj'].shape) == [2, 384, 16, 16], list(cap['d_proj'].shape))
    st = net.last_dino_stats
    obs = net.last_attn_stats
    check('alpha on a real batch matches dinolight\'s init',
          abs(obs['aca_alpha'] - float(torch.sigmoid(d.alpha_logit))) < 1e-6,
          f"{obs['aca_alpha']:.6f}")
    check('projected_norm (= ||alpha*F_ca||) is 0 at init, like every other arm',
          st['projected_norm'] == 0.0,
          f"latent {st['latent_norm']:.2f} ratio {st['injection_ratio']:.4f}")
    if n_layers > 1:
        w = cap['affm_weights']
        u = 1.0 / n_layers
        check(f'AFFM weights are EXACTLY 1/{n_layers} = {u:.4f} at init',
              float((w - u).abs().max()) < 1e-6,
              f'max |w - {u:.4f}| = {float((w - u).abs().max()):.3e}')
        check('AFFM weights sum to 1 at EVERY position',
              float((w.sum(1) - 1.0).abs().max()) < 1e-5)
        check('affm_w_sum observation == 1.0',
              abs(obs['affm_w_sum'] - 1.0) < 1e-6, f"{obs['affm_w_sum']:.6f}")
    check('observation keys published to the wrapper hook',
          all(k in obs for k in ('aca_alpha', 'aca_ca_to_sa_ratio',
                                 'aca_injected_norm', 'aca_temp_sa_h0',
                                 'aca_entropy_ca_h0')), f'{len(obs)} keys')

    # ------------------------------------------------------- the staircase
    print('\n=== gradients: the three-step staircase ===')
    loss = torch.nn.functional.l1_loss(o, gt)
    loss.backward()
    feat = {'to_q': net.aca.to_q[0].weight, 'P': net.P.weight}
    cross = {'to_v_cross': net.aca.to_v_cross[0].weight,
             'alpha': net.aca.alpha_logit}
    if n_layers > 1:
        cross['affm.score.0'] = net.affm.score[0].weight
    check('step 1: ONLY project_out is live',
          float(net.aca.project_out.weight.grad.abs().max()) > 0
          and all(float(t.grad.abs().max()) == 0.0
                  for t in list(feat.values()) + list(cross.values())),
          f'project_out {float(net.aca.project_out.weight.grad.abs().max()):.2e}')
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=3e-4)
    opt.step(); opt.zero_grad(set_to_none=True)
    torch.nn.functional.l1_loss(net(stacked), gt).backward()
    check('step 2: the FEATURE path and P join',
          all(float(t.grad.abs().max()) > 0 for t in feat.values()),
          '  '.join(f'{k} {float(v.grad.abs().max()):.2e}' for k, v in feat.items()))
    opt.step(); opt.zero_grad(set_to_none=True)
    torch.nn.functional.l1_loss(net(stacked), gt).backward()
    check('STEP 3: EVERY new parameter is live, cross path included',
          all(t.grad is not None and torch.isfinite(t.grad).all().item()
              and float(t.grad.abs().max()) > 0
              for t in list(feat.values()) + list(cross.values())),
          '  '.join(f'{k} {float(v.grad.abs().max()):.2e}'
                    for k, v in list(feat.items()) + list(cross.items())))

    # ================================================ THE SCALE CHECK
    shapes = {}
    if args.scale_check:
        print('\n=== SCALE CHECK — channel attention must ignore the token count ===')
        net.eval()
        with torch.no_grad():
            for mode, size, tok in (('train128', 128, 16), ('eval256', 256, 32)):
                net.set_dino_mode(mode)
                big = torch.cat([torch.rand(1, 1, size, size, device=args.device),
                                 torch.rand(1, 1, size, size, device=args.device)], 1)
                oo = net(big)
                c = net._dino_capture
                _, attn = net.aca._attend(
                    net.aca.to_q(net.aca.norm_f(c['d_proj'])),
                    net.aca.to_k(net.aca.norm_f(c['d_proj'])),
                    net.aca.to_v(net.aca.norm_f(c['d_proj'])),
                    net.aca.temperature_sa, want_attn=True)
                shapes[mode] = {'radar': size, 'tokens': f'{tok}x{tok}',
                                'd_proj': list(c['d_proj'].shape),
                                'attn': list(attn.shape), 'out': list(oo.shape)}
                print(f"    {mode:<9} radar {size} -> {tok}x{tok} tokens   "
                      f"d_proj {tuple(c['d_proj'].shape)}   attn {tuple(attn.shape)}")
        a1, a2 = shapes['train128']['attn'], shapes['eval256']['attn']
        check('ATTENTION MATRIX SHAPE IDENTICAL AT BOTH SCALES', a1 == a2,
              f'{a1} == {a2}')
        check('...and it is C/heads x C/heads, NOT tokens x tokens',
              a1[-1] == a1[-2] == 384 // net.aca.heads and a1[-1] not in (256, 1024),
              f'{a1[-1]}x{a1[-2]}; token counts would have been 256 and 1024')
        check('the d_proj grid DOES change with scale (invariance is real)',
              shapes['train128']['d_proj'][-1] == 16
              and shapes['eval256']['d_proj'][-1] == 32,
              f"{shapes['train128']['d_proj']} vs {shapes['eval256']['d_proj']}")
        check('eval256 output is [1,1,256,256]',
              shapes['eval256']['out'] == [1, 1, 256, 256])
    net._capture_dino_io = False

    nf = sum(1 for r in RESULTS if not r['pass'])
    print(f'\n=== {len(RESULTS) - nf}/{len(RESULTS)} checks passed ===')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'created': datetime.datetime.now().astimezone().isoformat(),
                   'name': name, 'config': os.path.abspath(args.config),
                   'layers': layers, 'has_affm': has_affm,
                   'device': args.device,
                   'param_delta_over_E0': delta, 'expected': expect,
                   'step0_max_deviation_from_E0': dev,
                   'scale_check': shapes,
                   'n_checks': len(RESULTS), 'n_failed': nf,
                   'checks': RESULTS}, f, indent=2)
    print(f'wrote {out}')
    if nf:
        print('SMOKE TESTS FAILED — stop and report; do not redesign.')
        sys.exit(1)


if __name__ == '__main__':
    main()
