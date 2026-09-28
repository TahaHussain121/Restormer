"""One self-contained figure: the dataset-level table AND the images, side by side.

The three arms of the closing comparison - E0 (no DINO), multi-level addition,
multi-level ACA - on the TEST split, full256 protocol, each at its own
best-VALIDATION checkpoint. The top block is the mean over all 338 test images;
the grid below shows five of them.

Case selection is E0's own full-image PSNR ranking (worst, 25th, 50th, 75th,
best), exactly as in make_arm_panel.py: the choice never sees a DINO arm's
score, so it cannot be cherry-picked for or against one of them. Every number
is READ from the evaluation chain's per-image CSVs; nothing is recomputed here.
"""

import csv, os
import cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_P3 = os.path.dirname(_HERE)
DATASET = '/home/woody/iwnt/iwnt174h/thesis_dino/holographic_image_dataset'
PROTO, SPLIT, NCASE = 'full256', 'test', 5

ARMS = [
    ('E0 baseline', 'Holo_E0_fixed128_baseline',                   'no DINO'),
    ('ml-addition', 'Holo_multilevel_addition_render_fixed128_B6', 'post-latent + dec3 + dec2, addition'),
    ('ml-ACA',      'Holo_multilevel_aca_render_fixed128_B6',      'post-latent + dec3 + dec2, ACA'),
]
BADGE = dict(boxstyle='round,pad=0.34', facecolor='black', alpha=0.78, edgecolor='none')
GREEN, RED, AMBER = '#7CFC9A', '#FF8A80', '#ffd166'


def read_csv(p):
    with open(p) as f:
        return {r['filename']: r for r in csv.DictReader(f)}


def load16(p):
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(p)
    return img.squeeze().astype(np.float64) / 65535.


def badge(ax, text, color='white', loc='top', size=15):
    y, va = (0.965, 'top') if loc == 'top' else (0.035, 'bottom')
    ax.text(0.035, y, text, transform=ax.transAxes, ha='left', va=va,
            color=color, fontsize=size, fontweight='bold', bbox=BADGE)


tables = {n: read_csv(os.path.join(_P3, 'results', e, 'metrics',
                                   f'{PROTO}_{SPLIT}_per_image.csv'))
          for n, e, _ in ARMS}
inp = read_csv(os.path.join(_P3, 'results', 'comparisons', 'input_baseline',
                            f'input_{PROTO}_{SPLIT}.csv'))

common = sorted(set.intersection(*(set(t) for t in tables.values())) & set(inp))
mean = lambda t, k: float(np.mean([float(t[f][k]) for f in common]))
e0_full, e0_mask = mean(tables['E0 baseline'], 'psnr_full'), mean(tables['E0 baseline'], 'psnr_mask')

ranked = sorted(common, key=lambda f: float(tables['E0 baseline'][f]['psnr_full']))
n = len(ranked) - 1
idx = [int(round(i * n / (NCASE - 1))) for i in range(NCASE)]
lab = lambda i: 'worst' if i == 0 else ('best' if i == n else f'{round(100 * i / n)}th pct')
chosen = [(lab(i), ranked[i]) for i in idx]

ncol, panel = 2 + len(ARMS), 5.0
fig = plt.figure(figsize=(panel * ncol, 5.1 + (panel + 0.5) * NCASE))
gs = fig.add_gridspec(NCASE + 1, ncol, height_ratios=[0.78] + [1] * NCASE,
                      hspace=0.055, wspace=0.03)

# ---- the table, spanning the full width -------------------------------------
tax = fig.add_subplot(gs[0, :]); tax.axis('off')
rows = [['', 'full256 PSNR', 'foreground PSNR', 'vs E0 (full / fg)', 'what it is']]
rows.append(['1e5 input', f"{mean(inp, 'psnr_full'):.3f}", f"{mean(inp, 'psnr_mask'):.3f}", '-', 'unrestored'])
for name, _, note in ARMS:
    f_, m_ = mean(tables[name], 'psnr_full'), mean(tables[name], 'psnr_mask')
    d = '-' if name == 'E0 baseline' else f'{f_ - e0_full:+.3f} / {m_ - e0_mask:+.3f}'
    rows.append([name, f'{f_:.3f}', f'{m_:.3f}', d, note])
tb = tax.table(cellText=rows, cellLoc='left', bbox=[0.04, 0.30, 0.92, 0.68],
               colWidths=[0.15, 0.13, 0.15, 0.19, 0.38])
tb.auto_set_font_size(False); tb.set_fontsize(15)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor('#cccccc')
    if r == 0:
        cell.set_facecolor('#eeeeee'); cell.set_text_props(fontweight='bold')
    elif r == 1:
        cell.set_facecolor('#fbf5e6')
    if r > 1 and c == 3 and rows[r][3] != '-':
        cell.set_text_props(fontweight='bold',
                            color='#1a7f37' if rows[r][3].startswith('+') else '#b3261e')
tax.set_title(
    f'E0 vs the two multi-level arms  -  {PROTO} / {SPLIT} split, '
    f'mean over all {len(common)} test images, best-validation checkpoint per arm\n'
    'foreground PSNR uses mask = GT > 0.01 (the frames are mostly near-black '
    'background, which inflates the full-image number)',
    fontsize=19, fontweight='bold', pad=26)

# ---- the images --------------------------------------------------------------
bs = max(13.0, 3.1 * panel)
for r, (tag, fn) in enumerate(chosen, start=1):
    noisy = load16(os.path.join(DATASET, f'{SPLIT}_verynoisy', fn))
    gt = load16(os.path.join(DATASET, f'{SPLIT}_clean', fn))
    preds = [(nm, load16(os.path.join(_P3, 'results', e, 'predictions',
                                      f'{PROTO}_{SPLIT}', 'raw', fn)),
              float(tables[nm][fn]['psnr_full']), float(tables[nm][fn]['psnr_mask']))
             for nm, e, _ in ARMS]
    vmax = max([gt.max(), noisy.max()] + [p.max() for _, p, _, _ in preds])
    best, best_m = max(p[2] for p in preds), max(p[3] for p in preds)

    ax = fig.add_subplot(gs[r, 0]); ax.imshow(noisy, cmap='inferno', vmin=0, vmax=vmax)
    badge(ax, f"{float(inp[fn]['psnr_full']):.2f} full\n{float(inp[fn]['psnr_mask']):.2f} fg",
          AMBER, size=bs)
    ax.set_ylabel(f'{tag}\n{fn}', fontsize=bs + 2, fontweight='bold', labelpad=18)
    axes_row = [ax]
    for c, (nm, img, ps, pm) in enumerate(preds, start=1):
        a = fig.add_subplot(gs[r, c]); a.imshow(img, cmap='inferno', vmin=0, vmax=vmax)
        win, winm = ps == best, pm == best_m
        badge(a, f'{ps:.2f} full' + ('  *BEST*' if win else '') +
                 f'\n{pm:.2f} fg' + ('  *BEST*' if winm else ''),
              GREEN if (win or winm) else 'white', size=bs)
        if nm != 'E0 baseline':
            d, dm = ps - preds[0][2], pm - preds[0][3]
            badge(a, f'{d:+.2f} full / {dm:+.2f} fg  vs E0', loc='bottom',
                  size=bs - 1.5, color=GREEN if d > 0 else RED)
        axes_row.append(a)
    a = fig.add_subplot(gs[r, ncol - 1]); a.imshow(gt, cmap='inferno', vmin=0, vmax=vmax)
    badge(a, 'target', AMBER, size=bs); axes_row.append(a)
    for a in axes_row:
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    if r == 1:
        heads = [('1e5 INPUT', '')] + [(nm, note) for nm, _, note in ARMS] + [('1e7 TARGET', '')]
        for a, (h, sub) in zip(axes_row, heads):
            a.annotate(h, xy=(0.5, 1.16 if sub else 1.05), xycoords='axes fraction',
                       ha='center', va='bottom', fontsize=17, fontweight='bold')
            if sub:
                a.annotate(sub, xy=(0.5, 1.045), xycoords='axes fraction', ha='center',
                           va='bottom', fontsize=12.5, color='#555555')

fig.text(0.5, 0.005,
         "rows: five test images picked by E0's OWN full-image PSNR ranking "
         "(worst to best), so the choice never sees a DINO arm's score    "
         "badges: per-image full / foreground PSNR; bottom = delta vs E0 "
         "(green better, red worse)",
         ha='center', fontsize=14, color='#444444')

out = os.path.join(_P3, 'results', 'comparisons',
                   f'E0_vs_multilevel_pair_{PROTO}_{SPLIT}.png')
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print('wrote', os.path.relpath(out, _P3))
