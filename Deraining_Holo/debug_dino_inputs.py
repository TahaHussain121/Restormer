"""Pre-launch verification for the two DINO arms. Saves REAL images + numbers.

Run:  python Deraining_Holo/debug_dino_inputs.py
Outputs -> Deraining_Holo/debug/

Four checks (see the thesis request):
  1. dump 8-sample grids for both arms: LQ | GT | render | actual DINO input
     (repeat->resize->ImageNet-norm, de-normalised for display), each panel
     labelled with shape/dtype/min/max/mean (also printed to stdout).
  2. alignment proof that render and LQ share the SAME crop + flip (shared RNG),
     not just "same function called".
  3. DINO input pipeline report: resize size/interp, whether ImageNet norm is
     applied and the resulting range, whether Restormer's input is reused, and
     a real DINO forward -> [B,3072].
  4. background-fraction of the DINO input at every progressive crop size.

Nothing is changed in the training path. This only inspects it.
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from basicsr.data.paired_image_uint16_render_dataset import Dataset_PairedImage_uint16_Render
from basicsr.data.transforms import paired_random_crop, random_augmentation
from basicsr.models.archs.restormer_dino_arch import (
    dino_preprocess, dino_denormalize, _IMAGENET_MEAN, _IMAGENET_STD, DINOv2Extractor)

R = '/home/woody/iwnt/iwnt174h/thesis_dino/holographic_image_dataset'
OUT = os.path.join(os.path.dirname(__file__), 'debug')
os.makedirs(OUT, exist_ok=True)
MEAN = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
STD = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
CROPS = [128, 160, 192, 256]
BG_THR = 0.05   # render pixel <= this counts as background (black)


def make_ds(gt_size):
    return Dataset_PairedImage_uint16_Render(dict(
        phase='train', scale=1, gt_size=gt_size, geometric_augs=True,
        dataroot_gt=f'{R}/train_clean', dataroot_lq=f'{R}/train_verynoisy',
        dataroot_render=f'{R}/train_renders_blackbg', io_backend=dict(type='disk')))


def stats(t):
    return f'{tuple(t.shape)} {str(t.dtype).replace("torch.","")} min={t.min():.3f} max={t.max():.3f} mean={t.mean():.3f}'


def to_disp(t):
    # t: CHW in [0,1]-ish -> HWC clipped for imshow
    a = t.detach().cpu().numpy()
    if a.shape[0] == 1:
        a = np.repeat(a, 3, 0)
    return np.clip(a.transpose(1, 2, 0), 0, 1)


# ---------------------------------------------------------------- 1 + 4: dumps
def dump_arm(arm, gt_size, n=8):
    ds = make_ds(gt_size)
    fig, ax = plt.subplots(n, 4, figsize=(14, 3.2 * n))
    print(f'\n==== ARM {arm} @ crop {gt_size} (DINO sees '
          f'{"RENDER" if arm=="A" else "LQ"}) ====')
    bg_fracs = []
    for i in range(n):
        s = ds[i]
        lq, gt, render = s['lq'], s['gt'], s['dino']
        src = render if arm == 'A' else lq
        dino_in = dino_preprocess(src.unsqueeze(0), 224, MEAN, STD)   # actual DINO input
        dino_disp = dino_denormalize(dino_in, MEAN, STD)[0]
        # background fraction of the render as DINO would see it (resized to 224)
        rend224 = torch.nn.functional.interpolate(
            render.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)[0]
        bg = (rend224.mean(0) <= BG_THR).float().mean().item()
        bg_fracs.append(bg)

        panels = [(lq, 'LQ crop (Restormer in)'), (gt, 'GT crop'),
                  (render, 'render crop'),
                  (dino_disp, f'DINO input (denorm)\nbg={bg:.0%}')]
        for j, (t, title) in enumerate(panels):
            ax[i, j].imshow(to_disp(t)); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            ax[i, j].set_title(f'{title}\n{stats(t)}', fontsize=7)
        if i < 3:
            print(f'  sample {i}: LQ     {stats(lq)}')
            print(f'            GT     {stats(gt)}')
            print(f'            render {stats(render)}')
            print(f'            DINO_in(normed) {stats(dino_in[0])}  bg={bg:.1%}')
    fig.suptitle(f'Arm {arm} ({"renderDINO" if arm=="A" else "lqDINO"}) @ crop {gt_size}',
                 fontweight='bold')
    fig.tight_layout()
    p = os.path.join(OUT, f'arm_{arm}_crop{gt_size}.png')
    fig.savefig(p, dpi=95, bbox_inches='tight'); plt.close(fig)
    print(f'  saved -> {p}   mean bg over {n} = {np.mean(bg_fracs):.1%}')


# ---------------------------------------------------------------- 2: alignment
def alignment_proof():
    print('\n==== ALIGNMENT PROOF (shared crop + flip RNG) ====')
    # asymmetric marker so a wrong flip OR wrong crop would show up
    H = 256
    yy, xx = np.mgrid[0:H, 0:H].astype(np.float32)
    marker = (0.3 * xx / H + 0.6 * yy / H)               # diagonal ramp
    marker[10:60, 10:40] = 1.0                            # corner block -> breaks symmetry
    m1 = marker[..., None].copy(); m3 = np.repeat(marker[..., None], 3, 2).copy()
    ident = 0
    for trial in range(200):
        gt = m1.copy()
        gt2, (lq2, rend2) = paired_random_crop(gt, [m1.copy(), m3.copy()], 128, 1, 'x')
        gt2, lq2, rend2 = random_augmentation(gt2, lq2, rend2)
        # lq2 is 1ch, rend2 is 3ch of the SAME marker; if crop+flip identical they match
        if np.array_equal(lq2[..., 0], rend2[..., 0]):
            ident += 1
    print(f'  same marker through lq-slot & render-slot: identical in '
          f'{ident}/200 random trials  -> {"SHARED RNG (aligned)" if ident==200 else "NOT ALIGNED"}')

    # visual: real sample, render silhouette contour over GT, aligned vs broken
    ds = make_ds(128)
    s = ds[3]
    gt = s['gt'][0].numpy(); render = s['dino'].mean(0).numpy()
    mask = (render > BG_THR).astype(np.float32)
    # deliberately-broken reference: flip the mask left-right
    broken = mask[:, ::-1]
    fig, ax = plt.subplots(1, 3, figsize=(11, 4))
    ax[0].imshow(render, cmap='gray'); ax[0].set_title('render crop (DINO)', fontsize=9)
    for a, m, t in [(ax[1], mask, 'GT + render outline\n(ALIGNED, from __getitem__)'),
                    (ax[2], broken, 'GT + FLIPPED outline\n(deliberately broken)')]:
        a.imshow(gt, cmap='inferno')
        a.contour(m, levels=[0.5], colors='cyan', linewidths=1.2)
        a.set_title(t, fontsize=9)
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); p = os.path.join(OUT, 'alignment_overlay.png')
    fig.savefig(p, dpi=110, bbox_inches='tight'); plt.close(fig)
    print(f'  saved overlay -> {p}  (aligned outline should hug the GT object;'
          f' flipped should not)')


# ---------------------------------------------------------------- 3: pipeline
def pipeline_report():
    print('\n==== DINO INPUT PIPELINE ====')
    ds = make_ds(128); s = ds[0]
    lq, render = s['lq'], s['dino']
    print(f'  resize: F.interpolate -> (224,224), mode=bilinear, align_corners=False')
    print(f'  patch 14 does NOT divide 128/160/192/256 -> resize to 224 (=16*14) is required')
    for name, src in [('LQ (arm B source)', lq), ('render (arm A source)', render)]:
        pre = src.unsqueeze(0)
        post = dino_preprocess(pre, 224, MEAN, STD)
        print(f'  {name}: BEFORE range [{pre.min():.3f},{pre.max():.3f}] '
              f'(1ch->repeat? {pre.shape[1]==1}) '
              f'-> AFTER ImageNet-norm range [{post.min():.3f},{post.max():.3f}]')
    print('  => range moves off [0,1] to ~[-2.1,2.6] => ImageNet norm IS applied.')
    print('  => arm B feeds DINO Restormer\'s own [0,1] input; [0,1] is the correct')
    print('     pre-normalisation range, and ImageNet norm is applied on top. Not a')
    print('     double-normalisation bug. (Content is still grayscale = OOD for DINO.)')

    # real DINO forward -> [B,3072]
    os.environ.setdefault('TORCH_HOME', '/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub')
    torch.hub.set_dir(os.path.join(os.environ['TORCH_HOME'], 'hub'))
    try:
        ext = DINOv2Extractor(layers=(0, 3, 7, 11), hub_source='local',
                              hub_dir=os.path.join(os.environ['TORCH_HOME'], 'hub'))
        with torch.no_grad():
            feat = ext(render.unsqueeze(0))
        print(f'  real DINO forward: output {tuple(feat.shape)} '
              f'(expected (1, 3072)) -> {"OK" if feat.shape==(1,3072) else "MISMATCH"}')
    except Exception as e:
        print(f'  real DINO forward FAILED: {e!r}')


# ---------------------------------------------------------------- 4: bg numbers
def background_report():
    print('\n==== BACKGROUND FRACTION of the render DINO input, per crop size ====')
    print('  (how empty the render is that DINO gets; N=200 random crops each)')
    for gt in CROPS:
        ds = make_ds(gt)
        fr = []
        for i in range(200):
            render = ds[i]['dino']
            r224 = torch.nn.functional.interpolate(
                render.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)[0]
            fr.append((r224.mean(0) <= BG_THR).float().mean().item())
        fr = np.array(fr)
        print(f'  crop {gt:>3}: mean bg = {fr.mean():5.1%} | median {np.median(fr):5.1%} '
              f'| >90% empty: {(fr>0.9).mean():4.0%} of crops '
              f'| >99% empty: {(fr>0.99).mean():4.0%}')


if __name__ == '__main__':
    print('N samples = 8 per arm; crop sizes =', CROPS)
    for arm in ('A', 'B'):
        dump_arm(arm, 128, n=8)     # dominant crop (first 92k iters)
    dump_arm('A', 256, n=8)         # also the largest, for comparison
    alignment_proof()
    pipeline_report()
    background_report()
    print('\nDONE. Open Deraining_Holo/debug/*.png')
