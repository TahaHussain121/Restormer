"""Smoke tests for global-render (the SPATIAL ablation of E1-render).

Everything E1-render's smoke test proves is re-proved here, because this arm
inherits that machinery and inheritance is exactly the kind of thing that
silently stops holding:

  * CROP + AUGMENTATION IDENTITY, proved not asserted: the dataset hands back the
    crop offsets and dihedral flag it drew, and this script re-derives the render
    crop from the ORIGINAL PNG on disk and compares with torch.equal;
  * train.py's sub-crop applied to the packed tensor keeps both channels in one
    slice;
  * DINO reads the RENDER channel and not the radar channel;
  * the RENDER means are in use, train128 in train mode and eval256 in eval mode;
  * DINO frozen, gradients None; P zero at init with a non-zero first gradient.

Plus the four checks that only THIS arm needs:

  * POOLING: the tensor handed to the projection is spatially constant, and
    equals the spatial mean of the pre-pool centered grid;
  * COMMUTATION: center-then-pool == pool-then-center, on a real batch, which is
    why the render means are reused rather than recomputed;
  * SPATIAL UNIFORMITY OF THE INJECTED SIGNAL: with a randomly-weighted 1x1
    projection, P(broadcast) is still spatially constant -- i.e. the arm really
    does inject one vector everywhere, not just at init;
  * PARAMETER COUNT identical to E1-render and exactly 295,296 above E0.

The zero-init identity check runs on CPU, where it is bitwise exact. On CUDA two
separately-constructed module instances differ by ~1e-4 through float
non-determinism, which is a property of the hardware, not of this arm.

Inference/one-step only. No training, no experiment identity, no checkpoint.
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
import torch.nn.functional as F
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE3 = os.path.dirname(_HERE)
_REPO = os.path.dirname(os.path.dirname(_PHASE3))
for p in (_REPO, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from basicsr.models.archs import define_network              # noqa: E402
from basicsr.data import create_dataset                       # noqa: E402
from basicsr.data.transforms import data_augmentation         # noqa: E402
import dino_shared                                            # noqa: E402

RESULTS = []
DINO_PREFIX = 'dino_ext.'


def check(name, ok, detail=''):
    RESULTS.append({'check': name, 'pass': bool(ok), 'detail': str(detail)})
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f'  --  {detail}' if detail else ''),
          flush=True)
    return bool(ok)


def build_dataset(cfg, seed):
    opt = copy.deepcopy(cfg['datasets']['train'])
    opt['phase'] = 'train'
    opt['scale'] = cfg['scale']
    opt['dist'] = False
    opt['return_crop_meta'] = True          # smoke only: expose the drawn values
    random.seed(seed)
    return create_dataset(opt)


def trainable(net):
    """Parameter count of the TRAINED part -- the frozen ViT is excluded, since
    it is not in the checkpoint and not in the optimizer."""
    return sum(p.numel() for n, p in net.named_parameters()
               if not n.startswith(DINO_PREFIX))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=os.path.join(
        _PHASE3, 'configs', 'global_addition_render_fixed128_B6_latent.yml'))
    ap.add_argument('--render-config', default=os.path.join(
        _PHASE3, 'configs', 'E1_addition_render_fixed128_spatial_B6_latent.yml'))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=100)
    ap.add_argument('--n-samples', type=int, default=4)
    ap.add_argument('--out', default=os.path.join(
        _PHASE3, 'results', 'wo2_implementation', 'smoke_results_global_render.json'))
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(args.render_config) as f:
        rcfg = yaml.safe_load(f)
    print(f'global-render {cfg["name"]}   ({args.config})')

    # ---------------------------------------------------------------- data
    print('\n=== dataset: stacking, crop and augmentation identity (INHERITED) ===')
    ds = build_dataset(cfg, args.seed)
    check('dataset type is the stacked render dataset (same as E1-render)',
          cfg['datasets']['train']['type']
          == rcfg['datasets']['train']['type']
          == 'Dataset_PairedImage_uint16_RenderStacked',
          cfg['datasets']['train']['type'])
    check('render dataroot identical to E1-render',
          cfg['datasets']['train']['dataroot_render']
          == rcfg['datasets']['train']['dataroot_render'],
          cfg['datasets']['train']['dataroot_render'])

    samples = [ds[i] for i in range(args.n_samples)]
    s0 = samples[0]
    check('lq tensor is stacked [2,H,W] (radar, render)',
          tuple(s0['lq'].shape) == (2, 256, 256), list(s0['lq'].shape))
    check('gt tensor is [1,H,W]', tuple(s0['gt'].shape) == (1, 256, 256),
          list(s0['gt'].shape))

    all_ok, details = True, []
    for s in samples:
        top, left, flag = int(s['crop_top']), int(s['crop_left']), int(s['aug_flag'])
        gt_size = cfg['datasets']['train']['gt_size']
        raw_lq = cv2.imread(s['lq_path'], cv2.IMREAD_UNCHANGED)
        raw_rd = cv2.imread(s['render_path'], cv2.IMREAD_UNCHANGED)
        lq_np = (raw_lq.astype(np.float32) / 65535.)[:, :, None]
        rd_np = (raw_rd.astype(np.float32) / 255.)[:, :, :1]
        lq_np = lq_np[top:top + gt_size, left:left + gt_size]
        rd_np = rd_np[top:top + gt_size, left:left + gt_size]
        if flag >= 0:
            lq_np = data_augmentation(lq_np, flag).copy()
            rd_np = data_augmentation(rd_np, flag).copy()
        exp_lq = torch.from_numpy(lq_np.transpose(2, 0, 1))
        exp_rd = torch.from_numpy(rd_np.transpose(2, 0, 1))
        ok_lq = torch.equal(exp_lq, s['lq'][0:1])
        ok_rd = torch.equal(exp_rd, s['lq'][1:2])
        all_ok &= ok_lq and ok_rd
        details.append(f"{os.path.basename(s['gt_path'])} top={top} left={left} "
                       f"flag={flag} lq={ok_lq} render={ok_rd}")
        if raw_rd.ndim == 3:
            all_ok &= bool((raw_rd[:, :, 0] == raw_rd[:, :, 1]).all()
                           and (raw_rd[:, :, 0] == raw_rd[:, :, 2]).all())
    check('CROP+AUG IDENTITY: render re-derived from disk with the SAME '
          '(top,left,flag) equals the dataset render channel, torch.equal',
          all_ok, ' | '.join(details))
    check('the same (top,left,flag) also reproduces the radar channel', all_ok,
          'both streams verified against the original PNGs')

    print('\n=== train.py sub-crop on the packed tensor ===')
    batch = torch.stack([s['lq'] for s in samples])          # [N,2,256,256]
    gt_b = torch.stack([s['gt'] for s in samples])
    crop = cfg['datasets']['train']['gt_sizes'][0]
    gt_size = cfg['datasets']['train']['gt_size']
    x0 = int((gt_size - crop) * random.random())
    y0 = int((gt_size - crop) * random.random())
    sub = batch[:, :, x0:x0 + crop, y0:y0 + crop]
    gt_sub = gt_b[:, :, x0:x0 + crop, y0:y0 + crop]
    check('sub-crop keeps both channels in ONE slice',
          torch.equal(sub[:, 1:2], batch[:, 1:2, x0:x0 + crop, y0:y0 + crop])
          and torch.equal(sub[:, 0:1], batch[:, 0:1, x0:x0 + crop, y0:y0 + crop]),
          f'window ({x0},{y0}) size {crop} -- pooling happens AFTER this, so the '
          f'crop still has to be right; it cannot desync because it is one tensor')

    # ---------------------------------------------------------------- model
    print('\n=== global-render, training mode (128) ===')
    torch.manual_seed(cfg['manual_seed'])
    net = define_network(copy.deepcopy(cfg['network_g'])).to(args.device)
    net.train()
    net.set_dino_mode('train128')
    net._capture_dino_io = True

    check('arch is RestormerDinoSpatialGlobalRender',
          type(net).__name__ == 'RestormerDinoSpatialGlobalRender',
          type(net).__name__)
    check('DINO SOURCE IS STILL render', net.dino_source == 'render',
          net.dino_source)
    check('pooling mode is global_mean', net.dino_pooling == 'global_mean',
          net.dino_pooling)
    check('block B6 / index 5',
          net.dino_block_1indexed == 6 and net.dino_block_0indexed == 5)
    check('train mean is the RENDER train128 mean (reused, not recomputed)',
          os.path.basename(net.dino_mean_paths['train128'])
          == 'render_B6_train128_dino224_mean.pt',
          os.path.basename(net.dino_mean_paths['train128']))
    check('eval mean is the RENDER eval256 mean (reused, not recomputed)',
          os.path.basename(net.dino_mean_paths['eval256'])
          == 'render_B6_eval256_dino448_mean.pt',
          os.path.basename(net.dino_mean_paths['eval256']))
    check('means are NOT the 1e5 means',
          '1e5' not in net.dino_mean_paths['train128']
          and '1e5' not in net.dino_mean_paths['eval256'])

    lq = sub.to(args.device)
    gt = gt_sub.to(args.device)
    out = net(lq)
    cap = net._dino_capture
    b = lq.shape[0]
    shapes = {'stacked_input': list(lq.shape),
              'radar': list(cap['radar'].shape),
              'render': list(cap['render'].shape),
              'dino_input': list(cap['preprocessed'].shape),
              'tokens': list(cap['tokens'].shape),
              'grid_prepool': list(cap['grid'].shape),
              'pooled': list(cap['pooled'].shape),
              'broadcast': list(cap['broadcast'].shape),
              'output': list(out.shape)}
    expect = {'stacked_input': [b, 2, 128, 128], 'radar': [b, 1, 128, 128],
              'render': [b, 1, 128, 128], 'dino_input': [b, 3, 224, 224],
              'tokens': [b, 256, 768], 'grid_prepool': [b, 768, 16, 16],
              'pooled': [b, 768, 1, 1], 'broadcast': [b, 768, 16, 16],
              'output': [b, 1, 128, 128]}
    for k, want in expect.items():
        check(f'G@128 {k} == {want}', shapes[k] == want, shapes[k])

    check('DINO SOURCE IS THE RENDER CHANNEL (torch.equal)',
          torch.equal(cap['source'], lq[:, 1:2]))
    check('DINO SOURCE IS NOT THE RADAR CHANNEL',
          not torch.equal(cap['source'], lq[:, 0:1]))
    check('Restormer restores the RADAR channel', torch.equal(cap['radar'], lq[:, 0:1]))

    # ------------------------------------------------------- THE POOLING
    print('\n=== the pooling itself ===')
    grid, pooled, bc = cap['grid'], cap['pooled'], cap['broadcast']
    manual = grid.mean(dim=(2, 3), keepdim=True)
    check('pooled == spatial mean of the pre-pool centered grid',
          torch.allclose(pooled, manual, atol=0, rtol=0),
          f'max |diff| = {float((pooled - manual).abs().max()):.3e}')
    spread = (bc - bc[:, :, :1, :1]).abs().max()
    check('broadcast tensor is SPATIALLY CONSTANT (every position identical)',
          float(spread) == 0.0, f'max spatial spread = {float(spread):.3e}')
    check('broadcast carries the pooled vector at every position',
          torch.equal(bc[:, :, 0, 0], pooled[:, :, 0, 0])
          and torch.equal(bc[:, :, -1, -1], pooled[:, :, 0, 0]))
    var_pre = float(grid.var(dim=(2, 3)).mean())
    var_post = float(bc.var(dim=(2, 3)).mean())
    check('spatial variance is destroyed by pooling (that IS the ablation)',
          var_pre > 0 and var_post == 0.0,
          f'mean per-channel spatial variance: pre-pool {var_pre:.6f} -> '
          f'post-pool {var_post:.6f}')
    # the scientific claim, tested with a NON-zero projection: what reaches the
    # latent is one vector everywhere, not merely zero everywhere at init.
    g = torch.Generator(device='cpu').manual_seed(0)
    W = torch.randn(net.latent_channels, net.dino_embed_dim, 1, 1, generator=g)
    proj_rand = F.conv2d(bc.float().cpu(), W)
    check('with a RANDOM 1x1 projection, P(broadcast) is still spatially constant',
          float((proj_rand - proj_rand[:, :, :1, :1]).abs().max()) < 1e-4,
          f'max spatial spread = '
          f'{float((proj_rand - proj_rand[:, :, :1, :1]).abs().max()):.3e}')

    # -------------------------------------------------- COMMUTATION CHECK
    print('\n=== centering / pooling commutation (why the means are reused) ===')
    mu = net.mu_train128
    raw = dino_shared.extract_block(net.dino_ext, cap['render'],
                                    net.dino_block_0indexed,
                                    dino_shared.DINO_SIZE_TRAIN128)   # [B,N,768]
    order_a = dino_shared.center_tokens(raw, mu).mean(dim=1)   # center THEN pool
    order_b = raw.mean(dim=1) - mu.to(raw.device, raw.dtype)   # pool THEN center
    dmax = float((order_a - order_b).abs().max())
    scale = float(order_a.abs().max())
    check('mean_p(D_p - mu) == mean_p(D_p) - mu  (float tolerance)',
          torch.allclose(order_a, order_b, atol=1e-4, rtol=1e-4),
          f'max |diff| = {dmax:.3e} against signal scale {scale:.3f} '
          f'(relative {dmax / max(scale, 1e-12):.2e})')
    check('the ARCH actually computes the center-then-pool order',
          torch.allclose(pooled[:, :, 0, 0], order_a, atol=1e-4, rtol=1e-4),
          f'max |diff| = {float((pooled[:, :, 0, 0] - order_a).abs().max()):.3e}')

    # ------------------------------------------------------ frozen / grads
    print('\n=== freezing, zero-init and gradients ===')
    check('DINO params frozen',
          all(not p.requires_grad for p in net.dino_ext.parameters()))
    net.train()
    check('DINO stays eval() after parent .train()',
          not net.dino_ext.dino.training and net.training)
    check('P zero-initialized (weight AND bias)',
          float(net.P.weight.abs().max()) == 0.0
          and float(net.P.bias.abs().max()) == 0.0)
    check('the pooled prior is non-zero (there IS signal to inject)',
          float(pooled.abs().max()) > 0,
          f'max |pooled| = {float(pooled.abs().max()):.4f}')
    loss = F.l1_loss(out, gt)
    check('finite loss', torch.isfinite(loss).item(), f'{float(loss):.6f}')
    loss.backward()
    check('P gradient non-zero on the FIRST backward',
          net.P.weight.grad is not None
          and torch.isfinite(net.P.weight.grad).all().item()
          and float(net.P.weight.grad.abs().max()) > 0,
          f'max |dL/dW_P| = {float(net.P.weight.grad.abs().max()):.6e}')
    check('DINO gradients are None',
          all(p.grad is None for p in net.dino_ext.parameters()))
    st = net.last_dino_stats
    check('stability stats finite', np.isfinite(st['injection_ratio']),
          f"latent {st['latent_norm']:.3f} projected {st['projected_norm']:.3f} "
          f"ratio {st['injection_ratio']:.6f}")
    check('train128 mean buffer in use', net.last_mean_key == 'mu_train128')

    # ------------------------------------------------------------ eval 256
    print('\n=== global-render, full-256 eval mode ===')
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
            'tokens': list(cap['tokens'].shape),
            'grid_prepool': list(cap['grid'].shape),
            'pooled': list(cap['pooled'].shape),
            'broadcast': list(cap['broadcast'].shape),
            'latent': list(lat.shape), 'output': list(out256.shape)}
    exp256 = {'input': [1, 2, 256, 256], 'dino_input': [1, 3, 448, 448],
              'tokens': [1, 1024, 768], 'grid_prepool': [1, 768, 32, 32],
              'pooled': [1, 768, 1, 1], 'broadcast': [1, 768, 32, 32],
              'latent': [1, 384, 32, 32], 'output': [1, 1, 256, 256]}
    for k, want in exp256.items():
        check(f'G@256 {k} == {want}', s256[k] == want, s256[k])
    check('G@256 pooled over ALL 1024 positions',
          cap['grid'].shape[-1] * cap['grid'].shape[-2] == 1024)
    check('G@256 broadcast is spatially constant at 32x32',
          float((cap['broadcast'] - cap['broadcast'][:, :, :1, :1]).abs().max()) == 0.0)
    check('G@256 no feature-grid interpolation (assertion still ACTIVE '
          'on the pre-pool grid)',
          dino_shared.assert_no_interpolation_needed(cap['grid'].shape[-2:],
                                                     lat.shape[-2:]),
          '32x32 == 32x32')
    check('G@256 eval mean buffer in use', net.last_mean_key == 'mu_eval256')
    net._capture_dino_io = False

    # --------------------------------------------- parameters and identity
    print('\n=== parameter count and zero-init identity (CPU, bitwise) ===')
    e0_args = {k: v for k, v in cfg['network_g'].items()
               if not k.startswith('dino_')}
    e0_args['type'] = 'Restormer'

    torch.manual_seed(cfg['manual_seed'])
    e0 = define_network(copy.deepcopy(e0_args))
    torch.manual_seed(cfg['manual_seed'])
    gl = define_network(copy.deepcopy(cfg['network_g']))
    torch.manual_seed(cfg['manual_seed'])
    e1r = define_network(copy.deepcopy(rcfg['network_g']))

    n_e0, n_gl, n_e1r = trainable(e0), trainable(gl), trainable(e1r)
    check('parameter count IDENTICAL to E1-render', n_gl == n_e1r,
          f'global {n_gl:,} vs E1-render {n_e1r:,}')
    check('parameter delta vs E0 is exactly 295,296', n_gl - n_e0 == 295296,
          f'E0 {n_e0:,} + 295,296 = {n_gl:,} (delta {n_gl - n_e0:,})')
    check('pooling added no parameters',
          n_gl - n_e0 == 768 * 384 + 384,
          f'768*384 + 384 = {768 * 384 + 384:,} = P.weight + P.bias only')

    gl.eval(); e0.eval(); e1r.eval()
    gl.set_dino_mode('train128'); e1r.set_dino_mode('train128')
    xr = torch.rand(1, 1, 128, 128)
    xs = torch.cat([xr, torch.rand(1, 1, 128, 128)], dim=1)
    with torch.no_grad():
        o_e0 = e0(xr)
        o_gl = gl(xs)
        o_e1r = e1r(xs)
    d_e0 = float((o_gl - o_e0).abs().max())
    d_e1r = float((o_gl - o_e1r).abs().max())
    check('ZERO-INIT IDENTITY: global step-0 output == E0 step-0 output, max diff 0.0',
          d_e0 == 0.0, f'max |global - E0| = {d_e0:.1f}')
    check('ZERO-INIT IDENTITY: global step-0 output == E1-render step-0 output',
          d_e1r == 0.0, f'max |global - E1-render| = {d_e1r:.1f}')
    check('trunk weights bit-identical to E0 (the RNG fence holds)',
          all(torch.equal(a, b) for (na, a), (nb, b) in zip(
              e0.named_parameters(), [(n, p) for n, p in gl.named_parameters()
                                      if not n.startswith(DINO_PREFIX)
                                      and not n.startswith('P.')])),
          'every trunk tensor compared with torch.equal')
    sd = gl.state_dict()
    check('frozen ViT stays OUT of the checkpoint',
          not any(k.startswith(DINO_PREFIX) for k in sd),
          f'{len(sd)} entries, 0 dino entries')

    n_fail = sum(1 for r in RESULTS if not r['pass'])
    print(f'\n=== {len(RESULTS) - n_fail}/{len(RESULTS)} checks passed ===')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'created': datetime.datetime.now().astimezone().isoformat(),
                   'config': os.path.abspath(args.config),
                   'reference_config': os.path.abspath(args.render_config),
                   'device': args.device, 'hostname': os.uname().nodename,
                   'shapes_train128': shapes, 'shapes_eval256': s256,
                   'params': {'E0': n_e0, 'E1_render': n_e1r, 'global': n_gl,
                              'delta_vs_E0': n_gl - n_e0},
                   'commutation_max_abs_diff': dmax,
                   'zero_init_max_diff_vs_E0': d_e0,
                   'zero_init_max_diff_vs_E1render': d_e1r,
                   'stats_train128': st,
                   'n_checks': len(RESULTS), 'n_failed': n_fail,
                   'checks': RESULTS}, f, indent=2)
    print(f'wrote {args.out}')
    if n_fail:
        print('SMOKE TESTS FAILED — stop and report; do not redesign.')
        sys.exit(1)


if __name__ == '__main__':
    main()
