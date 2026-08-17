"""Training-time validation PSNR curves for the four Phase-3 arms.

READS ONLY the basicsr training logs in experiments/<name>/train_*.log. It
computes nothing: every point is a `Validation ValSet ... # psnr:` line that
basicsr already wrote at its own val_freq (4k).

SCALE WARNING, printed on the figure: this is the 8-BIT training-time val path
(`val.use_image: true`). It is NOT the uint16 full256/crop128 evaluation the
arm tables quote, and the two must never be mixed. It is still the right curve
for "how did training go" and it is the criterion the best-validation
checkpoints were selected on.

RESUMES. The self-chaining driver restarts from the last training state, so a
few iterations appear in more than one log. Logs are read in chronological
order and a repeated iteration keeps the LAST value written, which is what the
run actually ended up with.
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE3 = os.path.dirname(_HERE)
_REPO = os.path.dirname(os.path.dirname(_PHASE3))

ARMS = [
    ('E0-Fixed',    'Holo_E0_fixed128_baseline',                            '#4c4c4c'),
    ('E1-noisy',    'Holo_E1_addition_noisy_fixed128_spatial_B6_latent',    '#c44e52'),
    ('E1-render',   'Holo_E1_addition_render_fixed128_spatial_B6_latent',   '#2b7bba'),
    ('global-render', 'Holo_global_addition_render_fixed128_B6_latent',     '#dd8452'),
]

ITER_RE = re.compile(r'iter:\s*([\d,]+)')
VAL_RE = re.compile(r'Validation ValSet.*?psnr:\s*([\d.]+)\s+.*?ssim:\s*([\d.]+)')


def read_curve(exp):
    """-> {iteration: (psnr, ssim)} from every train log of that experiment."""
    logs = sorted(glob.glob(os.path.join(_REPO, 'experiments', exp,
                                         'train_*.log')))
    points, it = {}, None
    for f in logs:
        with open(f) as fh:
            for ln in fh:
                m = ITER_RE.search(ln)
                if m:
                    it = int(m.group(1).replace(',', ''))
                v = VAL_RE.search(ln)
                if v and it is not None:
                    points[it] = (float(v.group(1)), float(v.group(2)))
    return points, logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(
        _PHASE3, 'results', 'comparisons', 'val_psnr_curves_four_arms.png'))
    args = ap.parse_args()

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(11, 9), gridspec_kw={'height_ratios': [2.1, 1]})
    print(f'  {"arm":<16}{"n val":>7}{"last iter":>11}{"best PSNR":>11}'
          f'{"@ iter":>9}{"final PSNR":>12}')
    best_rows = []
    for label, exp, color in ARMS:
        pts, logs = read_curve(exp)
        if not pts:
            print(f'  {label:<16} NO LOGS FOUND in experiments/{exp}')
            continue
        its = sorted(pts)
        psnr = [pts[i][0] for i in its]
        bi = max(its, key=lambda k: pts[k][0])
        last = its[-1]
        best_rows.append((label, exp, len(its), last, pts[bi][0], pts[bi][1],
                          bi, pts[last][0]))
        print(f'  {label:<16}{len(its):>7}{last:>11}{pts[bi][0]:>11.4f}'
              f'{bi:>9}{pts[last][0]:>12.4f}')

        for a in (ax, ax2):
            a.plot(its, psnr, color=color, lw=1.4, label=label, alpha=0.95)
            a.plot([bi], [pts[bi][0]], marker='*', ms=14, color=color,
                   markeredgecolor='white', markeredgewidth=0.8, zorder=5)

    ax.set_title('Phase-3 arms — training-time validation PSNR (8-bit val path, '
                 'n=339)\n★ = best-validation checkpoint, the one each arm was '
                 'evaluated with', fontsize=12)
    ax.set_ylabel('val PSNR (dB, 8-bit training path)')
    ax.grid(alpha=0.25)
    ax.legend(loc='lower right', fontsize=10)
    ax2.set_xlabel('iteration')
    ax2.set_ylabel('val PSNR (dB) — zoom')
    ax2.grid(alpha=0.25)
    ax2.set_ylim(19.0, 23.5)
    ax2.set_xlim(0, 305000)
    ax2.axvline(92000, color='k', ls=':', lw=1, alpha=0.5)
    ax2.text(93500, 19.15, 'cosine restart @92k', fontsize=8, alpha=0.7)

    note = ('NOT comparable to the uint16 full256 / crop128 numbers in the arm '
            'tables — this is basicsr\'s 8-bit training-time val metric.')
    fig.text(0.5, 0.005, note, ha='center', fontsize=8.5, style='italic',
             alpha=0.8)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'\n  wrote {args.out}')

    print(f'\n  best-validation checkpoint per arm')
    print(f'  {"arm":<16}{"iter":>8}{"val PSNR":>10}{"val SSIM":>10}  checkpoint')
    for label, exp, _, _, bp, bs, bi, _ in best_rows:
        ck = os.path.join('experiments', exp, 'models', f'net_g_{bi}.pth')
        ok = 'OK' if os.path.isfile(os.path.join(_REPO, ck)) else 'MISSING'
        print(f'  {label:<16}{bi:>8}{bp:>10.4f}{bs:>10.4f}  {ck}  [{ok}]')


if __name__ == '__main__':
    main()
