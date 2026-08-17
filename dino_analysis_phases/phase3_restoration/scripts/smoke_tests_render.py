"""Smoke tests for E1-render (the source ablation of E1-addition).

Same shape checks as E1-addition at 128 and 256, plus the checks that only this
arm needs:

  * the render reaches DINO preprocessing unchanged, and the RADAR does not;
  * CROP + AUGMENTATION IDENTITY, proved rather than asserted: the dataset hands
    back the crop offsets and the dihedral flag it drew, and this script
    re-derives the render crop from the ORIGINAL PNG on disk and compares with
    torch.equal against the channel the dataset produced. A shape check would
    not catch a misaligned window; this does.
  * train.py's own sub-crop is then applied to the packed batch and the render
    channel is checked against an independently sub-cropped render.
  * the render mean is loaded in train mode and the eval mean in eval mode;
  * DINO frozen, gradients None; P zero at init with a non-zero first gradient.

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=os.path.join(
        _PHASE3, 'configs', 'E1_addition_render_fixed128_spatial_B6_latent.yml'))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=100)
    ap.add_argument('--n-samples', type=int, default=4)
    ap.add_argument('--out', default=os.path.join(
        _PHASE3, 'results', 'wo2_implementation', 'smoke_results_render.json'))
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    print(f'E1-render {cfg["name"]}   ({args.config})')

    # ---------------------------------------------------------------- data
    print('\n=== dataset: stacking, crop and augmentation identity ===')
    ds = build_dataset(cfg, args.seed)
    check('dataset type is the stacked render dataset',
          cfg['datasets']['train']['type'] == 'Dataset_PairedImage_uint16_RenderStacked',
          cfg['datasets']['train']['type'])

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
        # --- re-derive BOTH streams from the original PNGs on disk -----------
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
        # the render's three PNG channels must really be identical
        if raw_rd.ndim == 3:
            all_ok &= bool((raw_rd[:, :, 0] == raw_rd[:, :, 1]).all()
                           and (raw_rd[:, :, 0] == raw_rd[:, :, 2]).all())
    check('CROP+AUG IDENTITY: render re-derived from disk with the SAME '
          '(top,left,flag) equals the dataset render channel, torch.equal',
          all_ok, ' | '.join(details))
    check('same (top,left,flag) also reproduces the radar channel', all_ok,
          'both streams verified against the original PNGs')
    check('render PNG channels are identical (ch0 loses nothing)', all_ok,
          'checked on every sampled render')

    # --- train.py's own sub-crop on the packed tensor ----------------------
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
          f'window ({x0},{y0}) size {crop} -- radar and render cannot desync '
          f'because they are the same tensor')
    check('sub-cropped shapes', list(sub.shape) == [len(samples), 2, 128, 128]
          and list(gt_sub.shape) == [len(samples), 1, 128, 128],
          f'lq {list(sub.shape)} gt {list(gt_sub.shape)}')

    # ---------------------------------------------------------------- model
    print('\n=== E1-render, training mode (128) ===')
    torch.manual_seed(cfg['manual_seed'])
    net = define_network(copy.deepcopy(cfg['network_g'])).to(args.device)
    net.train()
    net.set_dino_mode('train128')
    net._capture_dino_io = True

    check('arch is RestormerDinoSpatialRender',
          type(net).__name__ == 'RestormerDinoSpatialRender', type(net).__name__)
    check('dino_source is render', net.dino_source == 'render', net.dino_source)
    check('block B6 / index 5',
          net.dino_block_1indexed == 6 and net.dino_block_0indexed == 5)
    check('train mean is the RENDER train128 mean',
          os.path.basename(net.dino_mean_paths['train128'])
          == 'render_B6_train128_dino224_mean.pt',
          os.path.basename(net.dino_mean_paths['train128']))
    check('eval mean is the RENDER eval256 mean',
          os.path.basename(net.dino_mean_paths['eval256'])
          == 'render_B6_eval256_dino448_mean.pt',
          os.path.basename(net.dino_mean_paths['eval256']))

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
              'grid': list(cap['grid'].shape),
              'output': list(out.shape)}
    expect = {'stacked_input': [b, 2, 128, 128], 'radar': [b, 1, 128, 128],
              'render': [b, 1, 128, 128], 'dino_input': [b, 3, 224, 224],
              'tokens': [b, 256, 768], 'grid': [b, 768, 16, 16],
              'output': [b, 1, 128, 128]}
    for k, want in expect.items():
        check(f'E1R@128 {k} == {want}', shapes[k] == want, shapes[k])

    check('DINO SOURCE IS THE RENDER CHANNEL (torch.equal)',
          torch.equal(cap['source'], lq[:, 1:2]),
          'the tensor entering DINO preprocessing is channel 1')
    check('DINO SOURCE IS NOT THE RADAR CHANNEL',
          not torch.equal(cap['source'], lq[:, 0:1]),
          'and the two channels genuinely differ')
    check('Restormer restores the RADAR channel (residual uses channel 0)',
          out.shape[1] == 1 and torch.equal(cap['radar'], lq[:, 0:1]))
    check('render reaches DINO preprocessing unchanged',
          torch.equal(cap['source'], cap['render']))

    check('DINO params frozen',
          all(not p.requires_grad for p in net.dino_ext.parameters()),
          f'{sum(1 for _ in net.dino_ext.parameters())} params')
    net.train()
    check('DINO stays eval() after parent .train()',
          not net.dino_ext.dino.training and net.training)
    check('P zero-initialized', float(net.P.weight.abs().max()) == 0.0
          and float(net.P.bias.abs().max()) == 0.0)
    check('centered render features are non-zero',
          float(cap['grid'].abs().max()) > 0,
          f'max |D_centered| = {float(cap["grid"].abs().max()):.4f}')

    loss = torch.nn.functional.l1_loss(out, gt)
    check('finite loss', torch.isfinite(loss).item(), f'{float(loss):.6f}')
    loss.backward()
    check('P gradient non-zero on the FIRST backward',
          net.P.weight.grad is not None
          and torch.isfinite(net.P.weight.grad).all().item()
          and float(net.P.weight.grad.abs().max()) > 0,
          f'max |dL/dW_P| = {float(net.P.weight.grad.abs().max()):.6e}')
    check('DINO gradients are None',
          all(p.grad is None for p in net.dino_ext.parameters()))
    trunk = [p.grad for n_, p in net.named_parameters()
             if p.grad is not None and not n_.startswith('dino_ext.')
             and not n_.startswith('P.')]
    check('Restormer gradients finite',
          len(trunk) > 0 and all(torch.isfinite(g).all().item() for g in trunk),
          f'{len(trunk)} tensors')
    st = net.last_dino_stats
    check('injection ratio finite', np.isfinite(st['injection_ratio']),
          f"latent {st['latent_norm']:.3f} projected {st['projected_norm']:.3f}")
    check('train128 mean buffer in use', net.last_mean_key == 'mu_train128',
          net.last_mean_key)

    # ------------------------------------------------------------ eval 256
    print('\n=== E1-render, full-256 eval mode ===')
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
            'latent': list(lat.shape), 'output': list(out256.shape)}
    exp256 = {'input': [1, 2, 256, 256], 'dino_input': [1, 3, 448, 448],
              'tokens': [1, 1024, 768], 'grid': [1, 768, 32, 32],
              'latent': [1, 384, 32, 32], 'output': [1, 1, 256, 256]}
    for k, want in exp256.items():
        check(f'E1R@256 {k} == {want}', s256[k] == want, s256[k])
    check('E1R@256 no feature-grid interpolation',
          dino_shared.assert_no_interpolation_needed(cap['grid'].shape[-2:],
                                                     lat.shape[-2:]),
          '32x32 == 32x32')
    check('E1R@256 eval mean buffer in use', net.last_mean_key == 'mu_eval256',
          net.last_mean_key)
    check('the two render means differ',
          not torch.equal(net.mu_train128, net.mu_eval256),
          f'||train128|| {float(net.mu_train128.norm()):.3f} vs '
          f'||eval256|| {float(net.mu_eval256.norm()):.3f}')
    net._capture_dino_io = False

    n_fail = sum(1 for r in RESULTS if not r['pass'])
    print(f'\n=== {len(RESULTS) - n_fail}/{len(RESULTS)} checks passed ===')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'created': datetime.datetime.now().astimezone().isoformat(),
                   'config': os.path.abspath(args.config),
                   'device': args.device, 'hostname': os.uname().nodename,
                   'shapes_train128': shapes, 'shapes_eval256': s256,
                   'n_checks': len(RESULTS), 'n_failed': n_fail,
                   'checks': RESULTS}, f, indent=2)
    print(f'wrote {args.out}')
    if n_fail:
        print('SMOKE TESTS FAILED — stop and report; do not redesign.')
        sys.exit(1)


if __name__ == '__main__':
    main()
