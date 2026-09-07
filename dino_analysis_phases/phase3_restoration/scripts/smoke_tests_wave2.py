"""Smoke tests for the wave-2 arms: the gates, post-latent, and single depths.

One file, three arm families, because every family has to pass the SAME core
contract and only the family-specific part differs:

  CORE, checked for every arm
    * the built class is the one the config names, and it descends from the
      right parent;
    * trunk weights byte-identical to E0's for the same seed (the RNG fence);
    * the step-0 output equals E0's exactly, because P is zero-initialised;
    * parameter delta over E0 is EXACTLY the predicted number;
    * DINO is frozen, stays in eval(), and receives no gradient;
    * the eval256 regime produces a 32x32 grid against a 32x32 latent with no
      feature-grid interpolation, and uses the eval mean buffer;
    * the monitoring keys the stability gate reads are present and finite.

  GATE arms additionally
    * the gate map is [B, 384, g, g] — per position AND per channel, not a
      scalar and not one number per position;
    * every value lies strictly in (0, 1);
    * at initialisation every value is EXACTLY 0.5 (G is zero-init), which is
      seed-independent;
    * the one-step staircase runs the RIGHT WAY ROUND: P has non-zero gradient
      on the FIRST backward (because dg/dP is the gate, 0.5, not zero) while G
      has exactly zero (because it is multiplied by P(D) = 0), and G is alive on
      the second. This is what distinguishes the arm from the aca-L6-nosa
      deadlock, where BOTH paths passed through a zero weight;
    * `projected_norm` logs ||g * P(D)||, the injected quantity.

  POST-LATENT additionally
    * the parameter count is IDENTICAL to addition-render's, not merely close;
    * the prior is added to the LATENT OUTPUT: the tensor handed to the gate
      statistics is self.latent(F), verified by recomputing it;
    * the eight latent blocks receive an UNGUIDED input, verified by checking
      that the latent input equals the plain encoder output.

  SINGLE-DEPTH arms additionally
    * the configured block really is read (B3 -> index 2, B9 -> index 8);
    * the centering means are that block's files and differ from B6's;
    * the parameter count equals addition-render's exactly.

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
GATE_DELTA = 295296


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
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=100)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    cfg, net = build(args.config, args.seed, args.device)
    name = cfg['name']
    arch = cfg['network_g']['type']
    is_gate = arch in ('RestormerGatedRender', 'RestormerGatedNoisy')
    is_post = arch == 'RestormerPostLatentRender'
    is_render = cfg['network_g'].get('dino_source') == 'render'
    block = int(cfg['network_g']['dino_block'])
    expected = ADDITION_DELTA + (GATE_DELTA if is_gate else 0)
    print(f'{name}   [{arch}]   B{block}   source='
          f'{cfg["network_g"].get("dino_source")}   ({args.config})')

    print('\n=== identity ===')
    check('built class matches the config', type(net).__name__ == arch,
          type(net).__name__)
    mro = [c.__name__ for c in type(net).__mro__]
    parent = ('RestormerDinoSpatialRender' if is_render
              else 'RestormerDinoSpatial')
    check(f'descends from {parent}', parent in mro, ' <- '.join(mro[:4]))
    check(f'reads block B{block} (0-indexed {block - 1})',
          net.dino_block_1indexed == block
          and net.dino_block_0indexed == block - 1,
          f'{net.dino_block_1indexed} / {net.dino_block_0indexed}')
    check('P is Conv2d(768 -> 384, 1x1) and zero-initialised',
          tuple(net.P.weight.shape) == (384, 768, 1, 1)
          and float(net.P.weight.abs().max()) == 0.0
          and float(net.P.bias.abs().max()) == 0.0)
    dom = 'render' if is_render else '1e5'
    check(f'centering means are the {dom} B{block} files',
          all(f'{dom}_B{block}_' in os.path.basename(net.dino_mean_paths[k])
              for k in ('train128', 'eval256')),
          ', '.join(os.path.basename(net.dino_mean_paths[k])
                    for k in ('train128', 'eval256')))
    if is_gate:
        check('gate G is Conv2d(768 -> 384, 1x1), zero-initialised',
              tuple(net.gate.G.weight.shape) == (384, 768, 1, 1)
              and float(net.gate.G.weight.abs().max()) == 0.0
              and float(net.gate.G.bias.abs().max()) == 0.0)
    if is_post:
        check('dino_injection records post_latent',
              net.dino_injection == 'post_latent', net.dino_injection)

    print('\n=== parameters and the RNG fence ===')
    _, e0 = build(args.e0_config, args.seed, 'cpu')
    n_e0, n_net = trainable_count(e0), trainable_count(net)
    delta = n_net - n_e0
    check(f'parameter delta over E0 == {expected}', delta == expected,
          f'E0 {n_e0} -> {n_net} = +{delta}')
    if not is_gate:
        check('IDENTICAL to addition-render\'s delta (295,296)',
              delta == ADDITION_DELTA, f'+{delta}')
    else:
        check('exactly DOUBLE addition-render\'s delta (P + gate)',
              delta == 2 * ADDITION_DELTA, f'+{delta}')
    e0_sd = e0.state_dict()
    sd = {k: v.cpu() for k, v in net.state_dict().items()}
    shared = [k for k in e0_sd if k in sd]
    same = [k for k in shared if torch.equal(e0_sd[k], sd[k])]
    check('every trunk tensor byte-identical to E0 (fence held)',
          len(same) == len(shared) and shared, f'{len(same)}/{len(shared)}')

    print('\n=== step 0 equals E0 ===')
    net.train()
    net.set_dino_mode('train128')
    net._capture_dino_io = True
    torch.manual_seed(0)
    radar = torch.rand(2, 1, 128, 128, device=args.device)
    gt = torch.rand(2, 1, 128, 128, device=args.device)
    if is_render:
        render = torch.rand(2, 1, 128, 128, device=args.device)
        inp = torch.cat([radar, render], dim=1)
    else:
        inp = radar
    out = net(inp)
    e0 = e0.to(args.device).train()
    with torch.no_grad():
        out_e0 = e0(radar)
    max_dev = float((out - out_e0).abs().max())
    check('STEP-0 OUTPUT EQUALS E0 (P is zero)',
          torch.allclose(out, out_e0, atol=1e-6, rtol=0),
          f'max |arm - E0| = {max_dev:.3e}')
    cap = net._dino_capture
    if is_render:
        check('DINO source is the render channel',
              torch.equal(cap['source'], inp[:, 1:2]))
        check('DINO source is NOT the radar channel',
              not torch.equal(cap['source'], inp[:, 0:1]))
    else:
        check('DINO source IS the radar tensor the network restores',
              torch.equal(cap['source'], inp))
    check('DINO frozen and eval() under parent .train()',
          all(not p.requires_grad for p in net.dino_ext.parameters())
          and not net.dino_ext.dino.training and net.training)

    st = dict(net.last_dino_stats)
    check('the gate rules\' three keys are present',
          sorted(st) == ['injection_ratio', 'latent_norm', 'projected_norm'],
          sorted(st))
    check('at init projected_norm == 0', st['projected_norm'] == 0.0,
          st['projected_norm'])

    # ---------------------------------------------------------- gate family
    if is_gate:
        print('\n=== the gate ===')
        with torch.no_grad():
            g = torch.sigmoid(net.gate.G(torch.cat(
                [cap['guided'] - cap['injected'], cap['projected']], dim=1)))
        check('gate map is [B, 384, 16, 16] — per position AND per channel',
              list(g.shape) == [2, 384, 16, 16], list(g.shape))
        check('every gate value is strictly inside (0, 1)',
              float(g.min()) > 0.0 and float(g.max()) < 1.0,
              f'min {float(g.min()):.4f} max {float(g.max()):.4f}')
        check('AT INIT THE GATE IS EXACTLY 0.5 EVERYWHERE (G is zero-init, so '
              'this is seed-independent)',
              float((g - 0.5).abs().max()) < 1e-6,
              f'max |g - 0.5| = {float((g - 0.5).abs().max()):.3e}')
        check('injected == g * P(D), and is 0 at init because P is 0',
              float(cap['injected'].abs().max()) == 0.0)
        obs = dict(net.last_attn_stats)
        check('gate observations published to the wrapper hook',
              sorted(obs) == ['gate_frac_closed', 'gate_frac_open', 'gate_mean',
                              'gate_spatial_std', 'gate_std'], sorted(obs))
        check('observed gate_mean is 0.5 at init', abs(obs['gate_mean'] - 0.5) < 1e-6,
              obs['gate_mean'])

    # ----------------------------------------------------- post-latent family
    if is_post:
        print('\n=== the injection point really moved ===')
        with torch.no_grad():
            enc4 = net.down3_4(net.encoder_level3(net.down2_3(
                net.encoder_level2(net.down1_2(net.encoder_level1(
                    net.patch_embed(radar)))))))
            lat = net.latent(enc4)
        check('the latent stage received an UNGUIDED input',
              torch.allclose(cap['latent_out'], lat, atol=1e-5),
              f'max dev {float((cap["latent_out"] - lat).abs().max()):.3e}')
        check('the prior is added to the LATENT OUTPUT, not its input',
              not torch.allclose(cap['latent_out'], enc4, atol=1e-3),
              'latent(F) differs from F, as it must')
        check('latent_norm logs the latent OUTPUT (the tensor added to)',
              abs(st['latent_norm'] - float(lat.norm())) < 1e-2,
              f'logged {st["latent_norm"]:.4f} vs ||latent(F)|| {float(lat.norm()):.4f}')

    # ------------------------------------------------------------- gradients
    print('\n=== gradients: the staircase, and no dead branch ===')
    loss = torch.nn.functional.l1_loss(out, gt)
    loss.backward()
    gP = net.P.weight.grad
    check('P HAS NON-ZERO GRADIENT ON THE FIRST BACKWARD',
          gP is not None and torch.isfinite(gP).all().item()
          and float(gP.abs().max()) > 0,
          f'max |dL/dP| = {float(gP.abs().max()):.6e}')
    if is_gate:
        gG = net.gate.G.weight.grad
        check('G gets ZERO gradient on the first backward (it multiplies '
              'P(D) = 0) — NOT the aca-L6-nosa deadlock, because P is live',
              gG is not None and float(gG.abs().max()) == 0.0,
              f'max |dL/dG| = {float(gG.abs().max()):.3e}')
    opt = torch.optim.AdamW([p_ for p_ in net.parameters() if p_.requires_grad],
                            lr=3e-4)
    opt.step()
    opt.zero_grad(set_to_none=True)
    out2 = net(inp)
    torch.nn.functional.l1_loss(out2, gt).backward()
    check('P is no longer zero after one step',
          float(net.P.weight.abs().max()) > 0,
          f'max |P| = {float(net.P.weight.abs().max()):.3e}')
    if is_gate:
        gG2 = net.gate.G.weight.grad
        check('G HAS NON-ZERO GRADIENT ON THE SECOND BACKWARD (staircase is '
              'one step deep and it clears)',
              gG2 is not None and torch.isfinite(gG2).all().item()
              and float(gG2.abs().max()) > 0,
              f'max |dL/dG| = {float(gG2.abs().max()):.6e}')
    check('DINO gradients are None',
          all(p.grad is None for p in net.dino_ext.parameters()))

    # ---------------------------------------------------------- eval256 regime
    print('\n=== the full-256 regime ===')
    net.set_dino_mode('eval256')
    big = (torch.rand(1, 2, 256, 256, device=args.device) if is_render
           else torch.rand(1, 1, 256, 256, device=args.device))
    with torch.no_grad():
        out256 = net(big)
    cap = net._dino_capture
    check('eval256: DINO input 448, grid 32x32, output 256',
          list(cap['preprocessed'].shape) == [1, 3, 448, 448]
          and list(cap['grid'].shape) == [1, 768, 32, 32]
          and list(out256.shape) == [1, 1, 256, 256],
          f'{list(cap["preprocessed"].shape)} {list(cap["grid"].shape)} '
          f'{list(out256.shape)}')
    check('eval256: the eval mean buffer is in use',
          net.last_mean_key == 'mu_eval256', net.last_mean_key)
    with torch.no_grad():
        lat256 = net.down3_4(net.encoder_level3(net.down2_3(
            net.encoder_level2(net.down1_2(net.encoder_level1(
                net.patch_embed(big[:, 0:1])))))))
    check('eval256: no feature-grid interpolation (32x32 == 32x32)',
          dino_shared.assert_no_interpolation_needed(cap['grid'].shape[-2:],
                                                     lat256.shape[-2:]))
    if is_gate:
        check('eval256: the gate is still per position (32x32), not a scalar',
              list(cap['injected'].shape) == [1, 384, 32, 32],
              list(cap['injected'].shape))
    net._capture_dino_io = False

    n_fail = sum(1 for r in RESULTS if not r['pass'])
    print(f'\n=== {len(RESULTS) - n_fail}/{len(RESULTS)} checks passed ===')
    out_path = args.out or os.path.join(
        _PHASE3, 'results', 'wo2_implementation', f'smoke_results_{name}.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'created': datetime.datetime.now().astimezone().isoformat(),
                   'config': os.path.abspath(args.config), 'experiment': name,
                   'arch': arch, 'block': block, 'device': args.device,
                   'hostname': os.uname().nodename,
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
