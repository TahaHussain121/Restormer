"""One figure with every evaluated arm's numbers side by side.

FORM. Two jobs, two panels, one axis each (never a dual axis):
  top     MAGNITUDE  - absolute PSNR per arm, with the do-nothing input floor
                       and the E0 baseline drawn as reference rules
  bottom  POLARITY   - delta against E0 with a 95% bootstrap CI, so "better or
                       worse than no prior" is the thing being read, and the
                       uncertainty is visible rather than implied

COLOR. Diverging blue<->red with a neutral zero, because the bottom panel encodes
polarity, not identity. Values are the data-viz reference palette's validated
hexes used unchanged (blue #2a78d6, red #e34948) -- the node validator is not
installed on this cluster, so no new hex was invented that would need checking.
Text stays in ink colors; the bars carry the identity.

Every number is READ from the evaluation chain's per-image CSVs.
"""

import argparse, csv, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE3 = os.path.dirname(_HERE)
OUT = os.path.join(_PHASE3, 'results', 'comparisons')

BLUE, RED, NEUTRAL = '#2a78d6', '#e34948', '#b9b8b4'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#8a8985'

ARMS = [
    ('E0-Fixed',        'Holo_E0_fixed128_baseline'),
    ('E1-noisy',        'Holo_E1_addition_noisy_fixed128_spatial_B6_latent'),
    ('addition-render', 'Holo_E1_addition_render_fixed128_spatial_B6_latent'),
    ('global-render',   'Holo_global_addition_render_fixed128_B6_latent'),
    ('concat-render',   'Holo_concat_render_fixed128_spatial_B6_latent'),
    ('crossattn-render','Holo_crossattn_render_fixed128_spatial_B6_latent'),
    ('priorquery-render','Holo_priorquery_render_fixed128_spatial_B6_latent'),
]


def per_image(exp, cell, key):
    p = os.path.join(_PHASE3, 'results', exp, 'metrics', f'{cell}_per_image.csv')
    if not os.path.isfile(p):
        return None
    return {r['filename']: float(r[key]) for r in csv.DictReader(open(p))}


def input_floor(cell, key):
    p = os.path.join(OUT, 'input_baseline', f'input_{cell}.csv')
    if not os.path.isfile(p):
        return None
    v = [float(r[key]) for r in csv.DictReader(open(p))]
    return float(np.mean(v))


def boot_ci(d, n=10000, seed=0):
    rng = np.random.RandomState(seed)
    b = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(n)])
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    cells = [(f'full256_{a.split}', 'full256'), (f'crop128_{a.split}', 'crop128')]

    arms = [(n, e) for n, e in ARMS
            if per_image(e, cells[0][0], 'psnr_full') is not None]
    dropped = [n for n, e in ARMS if (n, e) not in arms]
    if dropped:
        print('  (not evaluated yet, dropped: ' + ', '.join(dropped) + ')')

    fig, axes = plt.subplots(2, 2, figsize=(16, 11.5),
                             gridspec_kw={'height_ratios': [1, 1.15]})
    y = np.arange(len(arms))[::-1]

    for col, (cell, plabel) in enumerate(cells):
        # ---------------- top: absolute magnitude ----------------
        ax = axes[0, col]
        vals_f = [np.mean(list(per_image(e, cell, 'psnr_full').values())) for _, e in arms]
        vals_m = [np.mean(list(per_image(e, cell, 'psnr_mask').values())) for _, e in arms]
        h = 0.36
        ax.barh(y + h / 2, vals_f, height=h, color=BLUE, label='full-image PSNR')
        ax.barh(y - h / 2, vals_m, height=h, color=NEUTRAL, label='masked PSNR')
        for yy, v in zip(y + h / 2, vals_f):
            ax.text(v + 0.18, yy, f'{v:.2f}', va='center', fontsize=11.5,
                    color=INK, fontweight='bold')
        for yy, v in zip(y - h / 2, vals_m):
            ax.text(v + 0.18, yy, f'{v:.2f}', va='center', fontsize=11, color=INK2)
        e0f = vals_f[0]
        ax.axvline(e0f, color=MUTED, lw=1.6, ls='--', zorder=0)
        ax.text(e0f, -0.92, 'E0 baseline', fontsize=11, color=MUTED,
                va='center', ha='center',
                bbox=dict(boxstyle='round,pad=0.22', fc='white', ec='none'))
        fl = input_floor(cell, 'psnr_full')
        if fl:
            ax.axvline(fl, color=MUTED, lw=1.6, ls=':', zorder=0)
            ax.text(fl, -0.92, '1e5 input', fontsize=11, color=MUTED,
                    va='center', ha='center',
                    bbox=dict(boxstyle='round,pad=0.22', fc='white', ec='none'))
        ax.set_yticks(y); ax.set_yticklabels([n for n, _ in arms], fontsize=12.5)
        ax.set_xlim(min(fl or 12, 12) - 0.6, max(vals_f) + 2.0)
        ax.set_xlabel('PSNR (dB)', fontsize=12.5, color=INK2)
        ax.set_title(f'{plabel} / {a.split}   —   absolute', fontsize=15,
                     fontweight='bold', pad=10)
        ax.legend(fontsize=11, frameon=False, ncol=2,
                  loc='upper center', bbox_to_anchor=(0.5, -0.16))
        ax.grid(axis='x', alpha=.25); ax.set_axisbelow(True)
        ax.set_ylim(-1.35, len(arms) - 0.35)
        for sp in ('top', 'right', 'left'): ax.spines[sp].set_visible(False)

        # ---------------- bottom: polarity vs E0 ----------------
        ax = axes[1, col]
        e0_exp = arms[0][1]
        sub = arms[1:]
        yy = np.arange(len(sub))[::-1]
        all_means, all_los, all_his = [], [], []
        for k, (key, off, alpha, lbl) in enumerate(
                [('psnr_full', +0.19, 1.0, 'full-image'),
                 ('psnr_mask', -0.19, 0.55, 'masked')]):
            base = per_image(e0_exp, cell, key)
            means, los, his = [], [], []
            for _, exp in sub:
                cur = per_image(exp, cell, key)
                ids = sorted(set(base) & set(cur))
                d = np.array([cur[i] - base[i] for i in ids])
                lo, hi = boot_ci(d)
                means.append(d.mean()); los.append(d.mean() - lo); his.append(hi - d.mean())
            all_means += means; all_los += los; all_his += his
            colors = [BLUE if m > 0 else RED for m in means]
            ax.barh(yy + off, means, height=0.34, color=colors, alpha=alpha)
            ax.errorbar(means, yy + off, xerr=[los, his], fmt='none',
                        ecolor=INK, elinewidth=1.6, capsize=4, capthick=1.6)
            for v, ypos, lo_, hi_ in zip(means, yy + off, los, his):
                end = (v + hi_ + 0.12) if v > 0 else (v - lo_ - 0.12)
                ax.text(end, ypos, f'{v:+.2f}',
                        va='center', ha='left' if v > 0 else 'right',
                        fontsize=11.5, color=INK,
                        fontweight='bold' if k == 0 else 'normal')
        ax.axvline(0, color=INK, lw=1.8)
        # legend in INK, not in the series colours: the bar colour means SIGN
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(fc=MUTED, alpha=1.0, label='upper bar: full-image'),
                           Patch(fc=MUTED, alpha=0.55, label='lower bar: masked')],
                  fontsize=11, frameon=False, ncol=2,
                  loc='upper center', bbox_to_anchor=(0.5, -0.13))
        ax.set_yticks(yy); ax.set_yticklabels([n for n, _ in sub], fontsize=12.5)
        ax.set_xlabel('PSNR delta vs E0 (dB)   —   bars = mean, whiskers = 95% bootstrap CI',
                      fontsize=12, color=INK2)
        ax.set_title(f'{plabel} / {a.split}   —   vs the no-DINO baseline',
                     fontsize=15, fontweight='bold', pad=10)
        ax.grid(axis='x', alpha=.25); ax.set_axisbelow(True)
        for sp in ('top', 'right', 'left'): ax.spines[sp].set_visible(False)
        # limits follow the data (a hardcoded range clipped crossattn at -3.15)
        lo_all = min(0.0, min(all_means) - max(all_los) - 0.9)
        hi_all = max(0.0, max(all_means) + max(all_his) + 0.9)
        ax.set_xlim(lo_all, hi_all)

    fig.suptitle('Phase-3 arms — every evaluated arm, best-validation checkpoint, '
                 f'{a.split} split\nblue = better than E0, red = worse; '
                 'a CI crossing zero means the difference is not established',
                 fontsize=17, y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = a.out or os.path.join(OUT, f'all_arms_results_summary_{a.split}.png')
    fig.savefig(out, dpi=155, bbox_inches='tight', facecolor='white')
    print('  wrote', out)


if __name__ == '__main__':
    main()
