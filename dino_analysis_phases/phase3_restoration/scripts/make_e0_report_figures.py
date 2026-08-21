"""The three E0-Fixed baseline report figures, in one reproducible place.

  curves      val PSNR / SSIM, L1 loss and the LR schedule on a shared axis
  qualitative 1e5 input | E0 | 1e7 target, cases from E0's own ranking
  protocols   the same images under full256 and crop128, with the 128 window
              drawn on the full-256 panels so the two protocols can be compared

LEGIBILITY: every number a reader needs is drawn INSIDE the panel it belongs to,
on a dark badge, at a size meant to be read at a glance. Values come from the
training logs and the evaluation CSVs; nothing here computes a metric.
"""

import argparse, csv, glob, json, os, re
import cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE3 = os.path.dirname(_HERE)
_REPO = os.path.dirname(os.path.dirname(_PHASE3))
DATASET = '/home/woody/iwnt/iwnt174h/thesis_dino/holographic_image_dataset'
EXP = 'Holo_E0_fixed128_baseline'
OUT = os.path.join(_PHASE3, 'results', 'comparisons')

plt.rcParams.update({'font.size': 13, 'axes.labelsize': 15,
                     'xtick.labelsize': 13, 'ytick.labelsize': 13})
BADGE = dict(boxstyle='round,pad=0.34', facecolor='black', alpha=0.75,
             edgecolor='none')


def badge(ax, text, color='white', loc='top', size=16):
    y, va = (0.965, 'top') if loc == 'top' else (0.035, 'bottom')
    ax.text(0.035, y, text, transform=ax.transAxes, ha='left', va=va,
            color=color, fontsize=size, fontweight='bold', bbox=BADGE)


def load16(p):
    im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise IOError(p)
    return im.squeeze().astype(np.float64) / 65535.


def per_image(cell):
    """-> {filename: (psnr_full, psnr_mask)}.

    BOTH metrics, because they answer different questions here: the frames are
    mostly near-black background, which inflates full-image PSNR, so the
    foreground-masked value (mask = GT > 0.01, no dilation) is the one that
    reflects object reconstruction. Ranking still uses psnr_full so the case
    selection matches earlier versions of these figures.
    """
    p = os.path.join(_PHASE3, 'results', EXP, 'metrics', f'{cell}_per_image.csv')
    return {r['filename']: (float(r['psnr_full']), float(r['psnr_mask']))
            for r in csv.DictReader(open(p))}


def input_psnr(cell):
    p = os.path.join(OUT, 'input_baseline', f'input_{cell}.csv')
    if not os.path.isfile(p):
        return {}
    return {r['filename']: (float(r['psnr_full']), float(r['psnr_mask']))
            for r in csv.DictReader(open(p))}


# ---------------------------------------------------------------- curves
def curves():
    IT = re.compile(r'iter:\s*([\d,]+)'); LR = re.compile(r'lr:\((.*?)\)')
    LP = re.compile(r'l_pix:\s*([\d.eE+-]+)')
    VA = re.compile(r'Validation ValSet.*?psnr:\s*([\d.]+)\s+.*?ssim:\s*([\d.]+)')
    val, loss, lr, it = {}, {}, {}, None
    for f in sorted(glob.glob(os.path.join(_REPO, 'experiments', EXP, 'train_*.log'))):
        for ln in open(f):
            m = IT.search(ln)
            if m:
                it = int(m.group(1).replace(',', ''))
                l, r = LP.search(ln), LR.search(ln)
                if l: loss[it] = float(l.group(1))
                if r: lr[it] = float(r.group(1).split(',')[0])
            v = VA.search(ln)
            if v and it: val[it] = (float(v.group(1)), float(v.group(2)))
    vi, li = sorted(val), sorted(loss)
    bi = max(val, key=lambda k: val[k][0])

    fig, ax = plt.subplots(3, 1, figsize=(14, 12), sharex=True,
                           gridspec_kw={'height_ratios': [2.3, 1.25, 1]})
    ax[0].plot(vi, [val[i][0] for i in vi], color='#2b7bba', lw=2.4,
               label='validation PSNR')
    ax[0].plot([bi], [val[bi][0]], marker='*', ms=26, color='#c44e52',
               mec='white', mew=1.6, zorder=5, label='best checkpoint')
    ax[0].annotate(f'BEST  {val[bi][0]:.4f} dB  @ {bi // 1000}k\nthis is the '
                   f'checkpoint every E0 number uses',
                   xy=(bi, val[bi][0]), xytext=(bi - 132000, val[bi][0] - 1.5),
                   fontsize=15, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', fc='#fff3cd', ec='#c44e52', lw=1.6),
                   arrowprops=dict(arrowstyle='->', color='#c44e52', lw=2.2))
    ax[0].set_ylabel('val PSNR (dB)\n8-bit training path', fontsize=15)
    ax[0].set_ylim(17.4, 22.9); ax[0].grid(alpha=.3, lw=1)
    ax[0].legend(loc='lower right', fontsize=14, framealpha=.95)
    ax[0].set_title('E0-Fixed baseline  —  Holo_E0_fixed128_baseline\n'
                    'stock Restormer, no DINO, fixed 128 crop, batch 8, '
                    '300k iterations, seed 100, n=339 val images',
                    fontsize=18, pad=16)
    axb = ax[0].twinx()
    axb.plot(vi, [val[i][1] for i in vi], color='#9a9a9a', lw=1.6, alpha=.8)
    axb.set_ylabel('val SSIM', color='#6a6a6a', fontsize=14)
    axb.tick_params(labelsize=12, colors='#6a6a6a')

    ax[1].plot(li, [loss[i] for i in li], color='#333333', lw=1.8)
    ax[1].set_ylabel('L1 training loss', fontsize=15); ax[1].set_yscale('log')
    ax[1].grid(alpha=.3, lw=1)
    ax[1].text(.99, .88, f'{loss[min(loss)]:.4f}  →  {loss[max(loss)]:.4f}',
               transform=ax[1].transAxes, ha='right', fontsize=15,
               fontweight='bold', bbox=dict(boxstyle='round,pad=0.4',
                                            fc='white', ec='#999999'))

    si = sorted(lr)
    ax[2].plot(si, [lr[i] for i in si], color='#dd8452', lw=2.4)
    ax[2].set_ylabel('learning rate', fontsize=15); ax[2].set_yscale('log')
    ax[2].set_xlabel('iteration', fontsize=16); ax[2].grid(alpha=.3, lw=1)
    ax[2].text(.99, .80, 'CosineAnnealingRestartCyclicLR   periods [92k, 208k]\n'
               f'{max(lr.values()):.1e}  →  {min(lr.values()):.1e}',
               transform=ax[2].transAxes, ha='right', fontsize=14,
               bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#999999'))

    for a in ax:
        a.axvline(92000, color='k', ls=':', lw=2, alpha=.65)
        a.set_xlim(0, 305000)
    ax[0].text(96000, 17.75, 'cosine restart @ 92k', fontsize=14, alpha=.85,
               bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#bbbbbb'))
    fig.tight_layout()
    p = os.path.join(OUT, 'E0_training_curves.png')
    fig.savefig(p, dpi=155, bbox_inches='tight', facecolor='white'); plt.close(fig)
    print('  wrote', p)


# ----------------------------------------------------------- qualitative
def qualitative(split='test'):
    psnr = per_image(f'full256_{split}'); inp = input_psnr(f'full256_{split}')
    ranked = sorted(psnr, key=lambda f: psnr[f][0])
    n = len(ranked) - 1
    picks = [('worst', ranked[0]), ('25th pct', ranked[n // 4]),
             ('median', ranked[n // 2]), ('best', ranked[n])]
    fig, ax = plt.subplots(len(picks), 3, figsize=(13.5, 4.4 * len(picks)))
    for r, (tag, fn) in enumerate(picks):
        noisy = load16(f'{DATASET}/{split}_verynoisy/{fn}')
        gt = load16(f'{DATASET}/{split}_clean/{fn}')
        pred = load16(os.path.join(_PHASE3, 'results', EXP, 'predictions',
                                   f'full256_{split}', 'raw', fn))
        vmax = max(gt.max(), noisy.max(), pred.max())
        gain = psnr[fn][0] - inp[fn][0]
        gain_m = psnr[fn][1] - inp[fn][1]
        for c, (img, col) in enumerate([(noisy, 'in'), (pred, 'e0'), (gt, 'gt')]):
            ax[r, c].imshow(img, cmap='inferno', vmin=0, vmax=vmax)
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            for sp in ax[r, c].spines.values(): sp.set_visible(False)
        badge(ax[r, 0], f'{inp[fn][0]:.2f} full\n{inp[fn][1]:.2f} mask', '#ffd166',
              size=14)
        badge(ax[r, 1], f'{psnr[fn][0]:.2f} full\n{psnr[fn][1]:.2f} mask', size=14)
        badge(ax[r, 1], f'{gain:+.2f} full / {gain_m:+.2f} mask  vs input',
              loc='bottom', size=13, color='#7CFC9A')
        badge(ax[r, 2], 'target', '#ffd166')
        ax[r, 0].set_ylabel(f'{tag}\n{fn}', fontsize=16, fontweight='bold',
                            labelpad=16)
    for c, t in enumerate(['1e5 INPUT', 'E0 PREDICTION', '1e7 TARGET']):
        ax[0, c].annotate(t, xy=(0.5, 1.05), xycoords='axes fraction',
                          ha='center', va='bottom', fontsize=18, fontweight='bold')
    fig.suptitle(f'E0-Fixed  —  {split}/full256, net_g_268000.pth\n'
                 "cases ranked by E0's own full-image PSNR    "
                 'badges: full-image and FOREGROUND-MASKED PSNR (mask = GT > 0.01)'
                 '    bottom = gain over the raw input',
                 fontsize=18, y=1.035)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    p = os.path.join(OUT, f'E0_qualitative_{split}.png')
    fig.savefig(p, dpi=155, bbox_inches='tight', facecolor='white'); plt.close(fig)
    print('  wrote', p)


# ------------------------------------------------------------- protocols
def protocols(split='test'):
    full, crop = per_image(f'full256_{split}'), per_image(f'crop128_{split}')
    fi, ci = input_psnr(f'full256_{split}'), input_psnr(f'crop128_{split}')
    crops = {}
    for ln in open(os.path.join(_PHASE3, 'results', 'crop_manifests',
                                f'matched128_{split}.csv')):
        if ln.startswith('#') or ln.startswith('image_id'): continue
        i, x, y, s = ln.strip().split(','); crops[i + '.png'] = (int(x), int(y), int(s))
    ids = sorted(set(full) & set(crop) & set(crops))
    ranked = sorted(ids, key=lambda f: full[f][0])
    picks = [('worst for E0', ranked[0]), ('median', ranked[len(ranked) // 2]),
             ('best for E0', ranked[-1]),
             ('largest gap', max(ids, key=lambda f: abs(full[f][0] - crop[f][0])))]
    base = os.path.join(_PHASE3, 'results', EXP, 'predictions')
    fig, ax = plt.subplots(len(picks), 6, figsize=(22, 4.0 * len(picks)))
    for r, (tag, fn) in enumerate(picks):
        x, y, s = crops[fn]
        imgs = [load16(f'{DATASET}/{split}_verynoisy/{fn}'),
                load16(f'{base}/full256_{split}/raw/{fn}'),
                load16(f'{DATASET}/{split}_clean/{fn}'),
                load16(f'{base}/crop128_{split}/input/{fn}'),
                load16(f'{base}/crop128_{split}/raw/{fn}'),
                load16(f'{base}/crop128_{split}/gt/{fn}')]
        vmax = max(i.max() for i in imgs[:3])
        for c, img in enumerate(imgs):
            ax[r, c].imshow(img, cmap='inferno', vmin=0, vmax=vmax)
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            for sp in ax[r, c].spines.values(): sp.set_visible(False)
            if c < 3:
                ax[r, c].add_patch(Rectangle((x, y), s, s, fill=False,
                                             ec='#00e5ff', lw=3.2, ls='--'))
        badge(ax[r, 0], f'{fi[fn][0]:.2f} full\n{fi[fn][1]:.2f} mask', '#ffd166',
              size=13.5)
        badge(ax[r, 1], f'{full[fn][0]:.2f} full\n{full[fn][1]:.2f} mask', size=13.5)
        badge(ax[r, 2], 'target', '#ffd166')
        badge(ax[r, 3], f'{ci[fn][0]:.2f} full\n{ci[fn][1]:.2f} mask', '#ffd166',
              size=13.5)
        badge(ax[r, 4], f'{crop[fn][0]:.2f} full\n{crop[fn][1]:.2f} mask', size=13.5)
        badge(ax[r, 4],
              f'{crop[fn][0] - full[fn][0]:+.2f} full / '
              f'{crop[fn][1] - full[fn][1]:+.2f} mask   vs full-256',
              loc='bottom', size=12.5,
              color='#7CFC9A' if crop[fn][0] > full[fn][0] else '#FF8A80')
        badge(ax[r, 5], 'target', '#ffd166')
        ax[r, 0].set_ylabel(f'{tag}\n{fn}\ncrop @ ({x},{y})', fontsize=15,
                            fontweight='bold', labelpad=16)
    for c, t in enumerate(['1e5 input', 'E0 prediction', '1e7 target',
                           '1e5 input', 'E0 prediction', '1e7 target']):
        ax[0, c].annotate(t, xy=(0.5, 1.05), xycoords='axes fraction',
                          ha='center', va='bottom', fontsize=16, fontweight='bold')
    fig.text(0.30, 1.005, 'FULL-256 PROTOCOL   (whole image)', ha='center',
             fontsize=20, fontweight='bold')
    fig.text(0.775, 1.005, 'MATCHED-128 PROTOCOL   (the cyan window only)',
             ha='center', fontsize=20, fontweight='bold')
    fig.suptitle(f'E0-Fixed, {split} split  —  the SAME images under both '
                 f'evaluation protocols\nthe cyan dashed box on the left half is '
                 f'exactly the 128x128 window shown on the right half   '
                 f'(shared manifest, seed 1234)    badges: full-image and '
                 f'FOREGROUND-MASKED PSNR', fontsize=17, y=1.06)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    p = os.path.join(OUT, f'E0_full256_vs_crop128_{split}.png')
    fig.savefig(p, dpi=140, bbox_inches='tight', facecolor='white'); plt.close(fig)
    print('  wrote', p)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--split', default='test')
    a = ap.parse_args()
    curves(); qualitative(a.split); protocols(a.split)
