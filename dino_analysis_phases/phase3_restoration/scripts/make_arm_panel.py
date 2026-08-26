"""Side-by-side predictions for every EVALUATED Phase-3 arm, one figure.

Selection rule, unchanged from make_three_arm_figures.py and for the same
reason: cases are chosen from **E0's own** per-image ranking, so the choice
never sees any DINO arm's numbers and cannot be cherry-picked for or against
one of them.

  --mode harsh    the N images E0 does worst on
  --mode spread   an even sweep of E0's ranking: worst, p25, median, p75, best

BOTH METRICS ARE SHOWN. Every badge carries the full-image PSNR and the
FOREGROUND-MASKED PSNR, because they answer different questions on this data:
the frames are mostly near-black background, which inflates the full-image
number, so masked PSNR (mask = GT > 0.01, no dilation, ~31% coverage at
full256) is the one that reflects object reconstruction. Deltas against E0 are
given for both. Case SELECTION still uses E0's full-image ranking, unchanged,
so the choice of images is identical to earlier versions of this figure.

LEGIBILITY. Numbers are drawn INSIDE each panel on a dark badge rather than as
small captions underneath: a caption sitting between two images is both hard to
read and hard to attribute to the right one. Each arm panel carries its PSNR at
the top and its delta against E0 at the bottom, green when it beats E0 and red
when it does not, so a row can be scanned without consulting a table.

An arm with no per-image CSV for the requested protocol/split is dropped with a
printed warning, so this script also serves as a status view of which arms have
been evaluated. Every number is READ from the evaluation chain's CSVs; nothing
is computed here.
"""

import argparse, csv, os
import cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE3 = os.path.dirname(_HERE)
DATASET = '/home/woody/iwnt/iwnt174h/thesis_dino/holographic_image_dataset'

plt.rcParams.update({'font.size': 13})
BADGE = dict(boxstyle='round,pad=0.34', facecolor='black', alpha=0.75,
             edgecolor='none')

ARMS = [
    ('E0',                'Holo_E0_fixed128_baseline',                             'no DINO'),
    ('E1-noisy',          'Holo_E1_addition_noisy_fixed128_spatial_B6_latent',     'DINO<-1e5, add'),
    ('addition-render',   'Holo_E1_addition_render_fixed128_spatial_B6_latent',    'DINO<-render, add'),
    ('global-render',     'Holo_global_addition_render_fixed128_B6_latent',        'render POOLED, add'),
    ('concat-render',     'Holo_concat_render_fixed128_spatial_B6_latent',         'render, concat'),
    ('crossattn-render',  'Holo_crossattn_render_fixed128_spatial_B6_latent',      'render, attn radar-Q'),
    ('priorquery-render', 'Holo_priorquery_render_fixed128_spatial_B6_latent',     'render, attn prior-Q'),
    ('affm-render',       'Holo_affm_render_fixed128_spatial_L3691_latent',        'layers {3,6,9,12}, add'),
    ('dinolight-render',  'Holo_dinolight_render_fixed128_L3691_aca_latent',       'layers {3,6,9,12}, ACA'),
]


def badge(ax, text, color='white', loc='top', size=15):
    """Numbers drawn inside the panel; size is set by the caller from --panel."""
    y, va = (0.965, 'top') if loc == 'top' else (0.035, 'bottom')
    ax.text(0.035, y, text, transform=ax.transAxes, ha='left', va=va,
            color=color, fontsize=size, fontweight='bold', bbox=BADGE)


def load16(p):
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(p)
    return img.squeeze().astype(np.float64) / 65535.


def read_csv(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return {r['filename']: r for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', default='full256')
    ap.add_argument('--split', default='test')
    ap.add_argument('--mode', default='spread', choices=['harsh', 'spread'])
    ap.add_argument('--n-cases', type=int, default=5)
    ap.add_argument('--panel', type=float, default=4.6,
                    help='inches per panel; raise it for fewer, larger cases')
    ap.add_argument('--out', default=None)
    ap.add_argument('--exclude', default=None,
                    help='comma-separated filenames to drop from the ranking '
                         'BEFORE selection, so a second figure shows different '
                         'cases without the choice ever seeing an arm score')
    ap.add_argument('--arms', default=None,
                    help='comma-separated arm names to include; E0 is always '
                         'kept and forced first, because it is both the case '
                         'selection reference and the delta baseline')
    args = ap.parse_args()

    if args.arms:
        want = [a.strip() for a in args.arms.split(',') if a.strip()]
        unknown = [a for a in want if a not in {n for n, _, _ in ARMS}]
        if unknown:
            raise SystemExit(f'unknown arm(s): {unknown}')
        keep = ['E0'] + [a for a in want if a != 'E0']
        arm_defs = sorted((a for a in ARMS if a[0] in keep),
                          key=lambda a: keep.index(a[0]))
    else:
        arm_defs = ARMS

    arms, tables = [], {}
    for name, exp, note in arm_defs:
        t = read_csv(os.path.join(_PHASE3, 'results', exp, 'metrics',
                                  f'{args.protocol}_{args.split}_per_image.csv'))
        if t is None:
            print(f'  (not evaluated yet, dropped: {name})')
            continue
        arms.append((name, exp, note)); tables[name] = t
    if 'E0' not in tables:
        raise SystemExit('E0 is the selection reference and is missing')

    e0 = tables['E0']
    common = sorted(set.intersection(*(set(t) for t in tables.values())))
    if args.exclude:
        drop = {f.strip() for f in args.exclude.split(',') if f.strip()}
        missing = drop - set(common)
        if missing:
            print(f'  (--exclude names not in the common set, ignored: '
                  f'{sorted(missing)})')
        hit = drop & set(common)
        common = [fn for fn in common if fn not in drop]
        print(f'  excluded {len(hit)} case(s); {len(common)} remain')
    ranked = sorted(common, key=lambda fn: float(e0[fn]['psnr_full']))
    if args.mode == 'harsh':
        chosen = [(f'E0 rank {i + 1}', fn) for i, fn in enumerate(ranked[:args.n_cases])]
        title_mode = f'the {args.n_cases} images E0 does WORST on'
    else:
        n = len(ranked) - 1
        k = args.n_cases
        idx = [int(round(i * n / (k - 1))) for i in range(k)] if k > 1 else [n // 2]
        def lab(i):
            if i == 0: return 'worst'
            if i == n: return 'best'
            return f'{round(100 * i / n)}th pct'
        chosen = [(lab(i), ranked[i]) for i in idx]
        title_mode = "an even sweep of E0's ranking (worst to best)"

    # the raw 1e5 input scored against the target, so each row shows where it
    # started as well as where every arm got to
    inp_t = read_csv(os.path.join(_PHASE3, 'results', 'comparisons',
                                  'input_baseline',
                                  f'input_{args.protocol}_{args.split}.csv'))
    inp_psnr = ({k: (float(v['psnr_full']), float(v['psnr_mask']))
                 for k, v in inp_t.items()} if inp_t else {})

    gt_dir = os.path.join(DATASET, f'{args.split}_clean')
    in_dir = os.path.join(DATASET, f'{args.split}_verynoisy')
    if args.protocol == 'crop128':
        base = os.path.join(_PHASE3, 'results', arms[0][1], 'predictions',
                            f'{args.protocol}_{args.split}')
        gt_dir, in_dir = os.path.join(base, 'gt'), os.path.join(base, 'input')

    ncol = 2 + len(arms)
    fig, axes = plt.subplots(len(chosen), ncol,
                             figsize=(args.panel * ncol, (args.panel + 0.45) * len(chosen)))
    if len(chosen) == 1:
        axes = axes[None, :]

    bs = max(13.0, 3.1 * args.panel)          # badge size follows panel size
    for r, (tag, fn) in enumerate(chosen):
        noisy, gt = load16(os.path.join(in_dir, fn)), load16(os.path.join(gt_dir, fn))
        preds = []
        for name, exp, _ in arms:
            preds.append((name, load16(os.path.join(
                _PHASE3, 'results', exp, 'predictions',
                f'{args.protocol}_{args.split}', 'raw', fn)),
                float(tables[name][fn]['psnr_full']),
                float(tables[name][fn]['psnr_mask'])))
        vmax = max([gt.max(), noisy.max()] + [p.max() for _, p, _, _ in preds])
        best = max(p[2] for p in preds)
        best_m = max(p[3] for p in preds)

        axes[r, 0].imshow(noisy, cmap='inferno', vmin=0, vmax=vmax)
        badge(axes[r, 0],
              (f'{inp_psnr[fn][0]:.2f} full\n{inp_psnr[fn][1]:.2f} mask'
               if fn in inp_psnr else 'input'), '#ffd166', size=bs)
        axes[r, 0].set_ylabel(f'{tag}\n{fn}', fontsize=bs + 2, fontweight='bold',
                              labelpad=18)
        for c, (name, img, psnr, pmask) in enumerate(preds, start=1):
            axes[r, c].imshow(img, cmap='inferno', vmin=0, vmax=vmax)
            win, winm = psnr == best, pmask == best_m
            badge(axes[r, c],
                  f'{psnr:.2f} full' + ('  *BEST*' if win else '') +
                  f'\n{pmask:.2f} mask' + ('  *BEST*' if winm else ''),
                  '#7CFC9A' if (win or winm) else 'white', size=bs)
            if name != 'E0':
                d, dm = psnr - preds[0][2], pmask - preds[0][3]
                badge(axes[r, c], f'{d:+.2f} full / {dm:+.2f} mask  vs E0',
                      loc='bottom', size=bs - 1.5,
                      color='#7CFC9A' if d > 0 else '#FF8A80')
        axes[r, ncol - 1].imshow(gt, cmap='inferno', vmin=0, vmax=vmax)
        badge(axes[r, ncol - 1], 'target', '#ffd166', size=bs)
        for c in range(ncol):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            for sp in axes[r, c].spines.values():
                sp.set_visible(False)

    axes[0, 0].annotate('1e5 INPUT', xy=(0.5, 1.05), xycoords='axes fraction',
                        ha='center', va='bottom', fontsize=16, fontweight='bold')
    for c, (name, _, note) in enumerate(arms, start=1):
        axes[0, c].annotate(name, xy=(0.5, 1.15), xycoords='axes fraction',
                            ha='center', va='bottom', fontsize=16.5,
                            fontweight='bold')
        axes[0, c].annotate(note, xy=(0.5, 1.045), xycoords='axes fraction',
                            ha='center', va='bottom', fontsize=12.5,
                            color='#555555')
    axes[0, ncol - 1].annotate('1e7 TARGET', xy=(0.5, 1.05), xycoords='axes fraction',
                               ha='center', va='bottom', fontsize=16,
                               fontweight='bold')
    fig.suptitle(
        f'Phase-3 arms side by side  —  {args.protocol} / {args.split},  '
        f'best-validation checkpoint per arm\n{title_mode}    '
        f'badges: full-image PSNR and FOREGROUND-MASKED PSNR (mask = GT > 0.01)'
        f'    bottom = delta vs E0 (green better, red worse)',
        fontsize=17, y=1.05)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    out = args.out or os.path.join(
        _PHASE3, 'results', 'comparisons',
        f'all_arms_{args.mode}_{args.protocol}_{args.split}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=155, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  wrote {out}')


if __name__ == '__main__':
    main()
