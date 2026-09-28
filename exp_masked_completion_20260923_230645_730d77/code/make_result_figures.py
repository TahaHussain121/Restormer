"""Presentation figures for the already-computed per-window metrics.

Post-hoc PRESENTATION only: it reads results/metrics/windows_per_window.csv and
plots what is already in docs/REPORT.md. No new analysis, no new model run, no
new metric definition. Writes only inside the experiment root.

Palette: the dataviz reference palette, slots 1-3 (blue / orange / aqua), used
unchanged in its documented all-pairs-validated configuration. Light surface
only: a PNG cannot carry a second theme.
"""
import csv, os, sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                              # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mc_common as mc                                        # noqa: E402

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#dedcd5'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'text.color': INK, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.edgecolor': GRID, 'font.size': 9, 'axes.titlesize': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
})


def load(tag=''):
    p = os.path.join(mc.RESULTS, 'metrics', f'windows_per_window{tag}.csv')
    rows = defaultdict(dict)
    with open(p) as f:
        for r in csv.DictReader(f):
            rows[(r['group'], r['variant'], r['method'])][r['id']] = r
    return rows


def val(r, k):
    v = r.get(k, '')
    if v in ('', None):
        return np.nan
    try:
        x = float(v)
    except ValueError:
        return np.nan
    return x if np.isfinite(x) else np.nan


def deltas(rows, group, method, metric, ref='E0', variant='oracle'):
    a, b = rows[(group, variant, ref)], rows[(group, variant, method)]
    ids = sorted(set(a) & set(b))
    d = np.array([val(b[i], metric) - val(a[i], metric) for i in ids])
    return d[np.isfinite(d)]


def dist(ax, y, data, color, label=None):
    """A thin horizontal distribution: violin outline + IQR bar + median tick."""
    if len(data) < 2:
        return
    v = ax.violinplot([data], positions=[y], vert=False, widths=0.72,
                      showextrema=False, showmedians=False)
    for b in v['bodies']:
        b.set_facecolor(color); b.set_alpha(0.22); b.set_edgecolor(color); b.set_linewidth(1.2)
    q1, med, q3 = np.percentile(data, [25, 50, 75])
    ax.plot([q1, q3], [y, y], color=color, lw=2.0, solid_capstyle='butt', zorder=3)
    ax.plot([med], [y], marker='|', ms=12, mew=2.2, color=color, zorder=4)
    ax.annotate(f'{med:+.2f}' if abs(med) >= 1 else f'{med:+.3f}',
                (q3, y), xytext=(6, 0), textcoords='offset points',
                ha='left', va='center', fontsize=8, color=color, zorder=5)


def figure_one(rows, out):
    panels = [('psnr', 'in-mask PSNR (dB)', 'intensity'),
              ('fill_ratio', 'fill ratio  (sum pred / sum target)', 'intensity'),
              ('frac_above_0p02', 'fraction of pixels above 0.02', 'intensity'),
              ('target_corr', 'target correlation', 'structure'),
              ('grad_ncc', 'gradient NCC', 'structure'),
              ('detail_corr', 'detail correlation', 'structure')]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.4))
    cats = [('recovery', 'COMP', BLUE), ('recovery', 'FILL-DIFF', ORANGE),
            ('control', 'COMP', BLUE), ('control', 'FILL-DIFF', ORANGE)]
    ylabels = ['recovery · COMP', 'recovery · FILL-DIFF', 'control · COMP', 'control · FILL-DIFF']
    for k, (metric, title, kind) in enumerate(panels):
        ax = axes[k // 3, k % 3]
        for j, (grp, meth, col) in enumerate(cats):
            dist(ax, 3 - j, deltas(rows, grp, meth, metric), col)
        ax.axvline(0, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.set_yticks([3, 2, 1, 0]); ax.set_yticklabels(ylabels if k % 3 == 0 else [])
        ax.set_ylim(-0.7, 3.9)
        ax.margins(x=0.10)
        ax.set_title(title, loc='left', color=INK)
        ax.grid(axis='x', color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.text(0.99, 0.02, kind, transform=ax.transAxes, ha='right', va='bottom',
                fontsize=8, color=AQUA if kind == 'structure' else INK2, style='italic')
    fig.suptitle('Change from the frozen E0 inside the oracle mask, per window '
                 '(228 recovery / 188 control validation windows)\n'
                 'Blue = the learned completion, orange = a parameter-free Laplace fill '
                 'given the SAME mask. Median labelled; bar = IQR. ORACLE-ASSISTED.',
                 fontsize=10.5, color=INK, ha='left', x=0.008, y=0.995)
    handles = [plt.Line2D([], [], color=BLUE, lw=3, label='COMP (learned completion)'),
               plt.Line2D([], [], color=ORANGE, lw=3, label='FILL-DIFF (Laplace fill, same mask)')]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=[0, 0.045, 1, 0.915])
    fig.savefig(out, dpi=150)
    print('wrote', out)


def figure_two(rows, out):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))

    # A: the measurement ablation
    ax = axes[0]
    for grp, col, lbl in (('recovery', BLUE, 'recovery windows'), ('control', ORANGE, 'control windows')):
        a, b = rows[(grp, 'oracle', 'COMP')], rows[(grp, 'oracle', 'COMP-SHUF')]
        ids = sorted(set(a) & set(b))
        x = np.array([val(a[i], 'psnr') for i in ids]); y = np.array([val(b[i], 'psnr') for i in ids])
        ax.scatter(x, y, s=16, color=col, alpha=0.55, linewidths=0, label=lbl, zorder=3)
    lo, hi = 5, 27
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
    ax.set_xlabel('in-mask PSNR with the REAL noisy frame (dB)')
    ax.set_ylabel('… with ANOTHER image\'s noisy frame (dB)')
    ax.set_title('The measurement is barely used', loc='left', color=INK)
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    ax.annotate('points on the line = swapping the\nmeasurement changes nothing',
                xy=(0.97, 0.06), xycoords='axes fraction', ha='right', fontsize=8, color=INK2)

    # B: false additions in the 3-px ring of a too-generous mask
    ax = axes[1]
    meths = ['E0', 'FILL-TELEA', 'FILL-DIFF', 'COMP']
    w = 0.36
    for j, (grp, col) in enumerate((('recovery', BLUE), ('control', ORANGE))):
        v = [np.nanmean([val(r, 'false_add_frac')
                         for r in rows[(grp, 'expanded_ring', m)].values()]) * 100 for m in meths]
        xs = np.arange(len(meths)) + (j - 0.5) * (w + 0.02)
        ax.bar(xs, v, width=w, color=col, linewidth=0, label=f'{grp} windows')
        for x, y in zip(xs, v):
            ax.annotate(f'{y:.1f}', (x, y), ha='center', va='bottom', fontsize=8, color=INK2)
    ax.set_xticks(range(len(meths))); ax.set_xticklabels(meths)
    ax.set_ylabel('% of ring pixels with pred > 0.05 where target ≤ 0.05')
    ax.set_title('A 3-px-too-large mask invents structure', loc='left', color=INK)
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)

    # C: background-mask control
    ax = axes[2]
    meths = ['E0', 'FILL-TELEA', 'COMP', 'FILL-DIFF']
    for j, (grp, col) in enumerate((('recovery', BLUE), ('control', ORANGE))):
        v = [np.nanmean([val(r, 'frac_above_0p02')
                         for r in rows[(grp, 'background', m)].values()]) * 100 for m in meths]
        xs = np.arange(len(meths)) + (j - 0.5) * (w + 0.02)
        ax.bar(xs, v, width=w, color=col, linewidth=0, label=f'{grp}-window images')
        for x, y in zip(xs, v):
            ax.annotate(f'{y:.1f}', (x, y), ha='center', va='bottom', fontsize=8, color=INK2)
    ax.set_xticks(range(len(meths))); ax.set_xticklabels(meths)
    ax.set_ylabel('% of masked pixels above 0.02')
    ax.set_title('A mask on true background: what gets added', loc='left', color=INK)
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle('Controls: is the completion using the measurement, and what does it do when the mask is wrong?  '
                 'ORACLE-ASSISTED, validation windows.', fontsize=10.5, color=INK, ha='left', x=0.008)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=150)
    print('wrote', out)


if __name__ == '__main__':
    rows = load()
    d = os.path.join(mc.RESULTS, 'figures')
    mc.ensure_dir(d)
    figure_one(rows, mc.assert_inside_root(os.path.join(d, 'presentation_intensity_vs_structure_v2.png')))
    figure_two(rows, mc.assert_inside_root(os.path.join(d, 'presentation_controls_v2.png')))
