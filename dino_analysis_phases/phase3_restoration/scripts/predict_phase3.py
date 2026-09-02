"""Phase-3 inference for E0 and E1, under BOTH evaluation protocols.

This script produces PREDICTIONS ONLY. It deliberately computes no metric:
the metric definitions stay in the repository's existing scripts
(`Deraining_Holo/masked_metrics.py`, `Deraining_Holo/analyze_sharpness.py`),
which are run afterwards, unchanged, over the directories written here. Nothing
is redefined, so a Phase-3 number and an Exp-2 number mean the same thing.

PROTOCOLS -- reported separately, never mixed.

  full256   scale-consistent full-image evaluation. Radar 256 -> (E1) DINO 448
            -> 32x32 tokens == the 32x32 latent. Uses the eval256 mean. No
            feature-grid interpolation anywhere.
  crop128   matched-128: one 128x128 crop per image, coordinates read from the
            manifest written by make_crop_manifest.py. The SAME crop is applied
            to the 1e5 input, the 1e7 target and every model's prediction, and
            the same manifest is used for E0 and E1. (E1) DINO 224 -> 16x16
            tokens == the 16x16 latent. Uses the train128 mean -- this is the
            training conditioning distribution exactly.

Output layout (both protocols):

    <out_root>/<protocol>_<split>/
        raw/   uint16 PNG predictions      -> --pred_dir for the metric scripts
        gt/    uint16 PNG targets          -> --gt_dir   (crop128 only; full256
               input/ uint16 PNG inputs        reuses the dataset directories)
        predict_metadata.json
"""

import argparse
import csv
import datetime
import json
import os
import subprocess
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

from basicsr.models.archs import define_network            # noqa: E402
import dino_shared                                          # noqa: E402

DATASET = '/home/woody/iwnt/iwnt174h/thesis_dino/holographic_image_dataset'
PROTOCOL_DINO_MODE = {'full256': 'eval256', 'crop128': 'train128'}


def git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       cwd=_REPO).decode().strip()
    except Exception:                                       # noqa: BLE001
        return 'unknown'


def load_uint16(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f'failed to read {path}')
    if img.ndim == 3:
        img = img[:, :, 0]
    return img


def shift_render(render, dx):
    """Displace the render by dx pixels along x, filling the vacated strip with
    zeros.

    The renders are drawn on a black background, so zero is the correct fill:
    a displaced render is what the pipeline would produce if the render stream
    were misregistered against the radar, and the region the object vacates is
    genuinely background. np.roll is deliberately NOT used -- wrapping would
    reintroduce object structure on the opposite edge and understate the damage.

    Only the DINO input is displaced. The radar, the target and the crop window
    are untouched, so this isolates render-to-radar misalignment and nothing
    else.
    """
    if dx == 0:
        return render
    out = np.zeros_like(render)
    if dx > 0:
        out[:, dx:] = render[:, :-dx]
    else:
        out[:, :dx] = render[:, -dx:]
    return out


def read_manifest(path):
    rows = {}
    with open(path) as f:
        lines = [ln for ln in f if not ln.startswith('#')]
    for r in csv.DictReader(lines):
        rows[r['image_id']] = (int(r['x']), int(r['y']), int(r['size']))
    if not rows:
        raise SystemExit(f'empty manifest {path}')
    return rows


def build_model(cfg, weights, device):
    net_opt = dict(cfg['network_g'])
    net = define_network(net_opt)
    ckpt = torch.load(weights, map_location='cpu', weights_only=False)
    sd = ckpt.get('params', ckpt)
    net.load_state_dict(sd, strict=True)          # strict: no silent mismatch
    return net.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='the experiment YAML')
    ap.add_argument('--weights', required=True)
    ap.add_argument('--split', required=True, choices=['val', 'test'])
    ap.add_argument('--protocol', required=True, choices=['full256', 'crop128'])
    ap.add_argument('--manifest', default=None, help='required for crop128')
    ap.add_argument('--out-root', required=True,
                    help='results/<experiment>/predictions')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, default=0, help='smoke tests only')
    ap.add_argument('--render-shift', type=int, default=0,
                    help='displace the render by N pixels along x before it '
                         'reaches DINO (render arms only). 0 = the normal '
                         'aligned path. Controls only; never used for a '
                         'headline number.')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    exp_name = cfg['name']
    # ANY DINO arm, detected by CAPABILITY and never by class name.
    #
    # This was a name prefix ('RestormerDinoSpatial') and it broke the moment an
    # arm was named outside that convention: RestormerDinoConcatRender and
    # RestormerDinoCrossAttnRender both fail the prefix test, so set_dino_mode()
    # was silently skipped and the arm stayed in the train128 regime while being
    # fed 256x256 images. Under full256 that produced a 16x16 DINO grid against
    # a 32x32 latent -- caught, loudly, by assert_no_interpolation_needed
    # (concat full256 jobs 1783319 / 1783321). Under crop128 it was accidentally
    # correct, because train128 is the default. A duck-typed check cannot rot
    # the same way: if the network can switch regimes, it is a DINO arm.
    arch_type = cfg['network_g']['type']
    # the render arms take a STACKED [B,2,H,W] input: radar ch0, render ch1
    # (Dataset_PairedImage_uint16_RenderStacked). Detected from the config, so a
    # new subclass cannot quietly fall through to the single-channel path.
    needs_render = cfg['network_g'].get('dino_source') == 'render'
    if args.render_shift and not needs_render:
        raise SystemExit(f'--render-shift {args.render_shift} given, but '
                         f'{arch_type} has no render stream to displace. '
                         f'Refusing to run a control that does nothing.')

    net = build_model(cfg, args.weights, args.device)
    is_dino = hasattr(net, 'set_dino_mode')
    if bool(cfg['network_g'].get('dino_enabled')) != is_dino:
        raise SystemExit(
            f'{arch_type}: config says dino_enabled='
            f'{cfg["network_g"].get("dino_enabled")!r} but the built network '
            f'{"has" if is_dino else "has no"} set_dino_mode(). Refusing to '
            f'guess which is right.')
    if is_dino:
        mode = PROTOCOL_DINO_MODE[args.protocol]
        net.set_dino_mode(mode)                   # EXPLICIT, from the protocol
        print(f'  DINO mode {mode} -> mean buffer '
              f'{"mu_train128" if mode == "train128" else "mu_eval256"}, '
              f'path {net.dino_mean_paths[mode]}')

    lq_dir = os.path.join(DATASET, f'{args.split}_verynoisy')
    gt_dir = os.path.join(DATASET, f'{args.split}_clean')
    render_dir = os.path.join(DATASET, f'{args.split}_renders_blackbg')
    if needs_render and not os.path.isdir(render_dir):
        raise SystemExit(f'{arch_type} needs the render stream, but '
                         f'{render_dir} does not exist')
    ids = sorted(os.path.splitext(f)[0] for f in os.listdir(gt_dir)
                 if f.endswith('.png'))
    if args.limit:
        ids = ids[:args.limit]

    crops = None
    if args.protocol == 'crop128':
        if not args.manifest:
            raise SystemExit('crop128 requires --manifest')
        crops = read_manifest(args.manifest)
        missing = [i for i in ids if i not in crops]
        if missing:
            raise SystemExit(f'{len(missing)} ids missing from the manifest, '
                             f'e.g. {missing[:5]}')

    out_dir = os.path.join(args.out_root, f'{args.protocol}_{args.split}')
    raw_dir = os.path.join(out_dir, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    if args.protocol == 'crop128':
        gt_out = os.path.join(out_dir, 'gt')
        in_out = os.path.join(out_dir, 'input')
        os.makedirs(gt_out, exist_ok=True)
        os.makedirs(in_out, exist_ok=True)
    else:
        gt_out, in_out = gt_dir, lq_dir           # full images: reuse the dataset

    shapes = {}
    with torch.no_grad():
        for n, image_id in enumerate(ids, 1):
            lq = load_uint16(os.path.join(lq_dir, f'{image_id}.png'))
            gt = load_uint16(os.path.join(gt_dir, f'{image_id}.png'))
            if lq.shape != gt.shape:
                raise SystemExit(f'{image_id}: lq {lq.shape} != gt {gt.shape}')

            render = None
            if needs_render:
                # 8-bit RGB with all three channels identical; keep channel 0,
                # exactly as Dataset_PairedImage_uint16_RenderStacked does.
                rp = os.path.join(render_dir, f'{image_id}.png')
                raw_r = cv2.imread(rp, cv2.IMREAD_COLOR)
                if raw_r is None:
                    raise SystemExit(f'missing render {rp}')
                render = raw_r[:, :, 0].astype(np.float32) / 255.
                if render.shape != lq.shape:
                    raise SystemExit(f'{image_id}: render {render.shape} != '
                                     f'lq {lq.shape}')
                # displace BEFORE the crop, so crop128 sees the misalignment
                # the same way the full frame does
                render = shift_render(render, args.render_shift)

            if args.protocol == 'crop128':
                x, y, s = crops[image_id]
                lq = lq[y:y + s, x:x + s]
                gt = gt[y:y + s, x:x + s]
                if render is not None:
                    # the SAME window as the radar -- the render must never be
                    # cropped independently
                    render = render[y:y + s, x:x + s]
                cv2.imwrite(os.path.join(gt_out, f'{image_id}.png'), gt)
                cv2.imwrite(os.path.join(in_out, f'{image_id}.png'), lq)

            inp = torch.from_numpy(
                lq.astype(np.float32) / 65535.)[None, None].to(args.device)
            if render is not None:
                inp = torch.cat(
                    [inp, torch.from_numpy(render)[None, None].to(args.device)],
                    dim=1)                            # [1,2,H,W] radar, render
            out = net(inp)
            pred = (out.clamp(0, 1)[0, 0].float().cpu().numpy() * 65535.
                    ).round().astype(np.uint16)
            if pred.shape != gt.shape:
                raise SystemExit(f'{image_id}: prediction {pred.shape} != '
                                 f'target {gt.shape} -- alignment broken')
            cv2.imwrite(os.path.join(raw_dir, f'{image_id}.png'), pred)
            shapes = {'input': list(inp.shape), 'output': list(out.shape)}
            if n % 50 == 0 or n == len(ids):
                print(f'    {n}/{len(ids)}', flush=True)

    meta = {
        'experiment': exp_name, 'config': os.path.abspath(args.config),
        'weights': os.path.abspath(args.weights),
        'split': args.split, 'protocol': args.protocol,
        'n_images': len(ids),
        'render_shift_px': args.render_shift,
        'manifest': os.path.abspath(args.manifest) if args.manifest else None,
        'dino': ({'mode': net.dino_mode,
                  'block_1indexed': net.dino_block_1indexed,
                  'mean_path': net.dino_mean_paths[net.dino_mode],
                  'grid': dino_shared.DINO_SIZE_EVAL256 // dino_shared.PATCH_SIZE
                  if net.dino_mode == 'eval256'
                  else dino_shared.DINO_SIZE_TRAIN128 // dino_shared.PATCH_SIZE}
                 if is_dino else None),
        'tensor_shapes': shapes,
        'pred_dir': os.path.abspath(raw_dir),
        'gt_dir': os.path.abspath(gt_out),
        'input_dir': os.path.abspath(in_out),
        'created': datetime.datetime.now().astimezone().isoformat(),
        'git_commit': git_commit(), 'hostname': os.uname().nodename,
        'device': args.device, 'torch': torch.__version__,
    }
    with open(os.path.join(out_dir, 'predict_metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'\n  predictions -> {raw_dir}')
    print(f'  targets     -> {gt_out}')
    print(f'  inputs      -> {in_out}')
    print(f'  metadata    -> {os.path.join(out_dir, "predict_metadata.json")}')


if __name__ == '__main__':
    main()
