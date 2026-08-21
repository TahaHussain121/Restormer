"""Visual evaluation report for the crossattn-render arm (radar queries, DINO K/V).

READ-ONLY. Computes no predictions and trains nothing. Every number comes from
artefacts that already exist:

  results/<arm>/metrics/*_summary.json          the four evaluated cells
  results/<arm>/metrics/*_per_image.csv         paired per-image deltas
  results/<arm>/metadata/best_checkpoint.json   the selection record
  results/comparisons/input_baseline/*.csv      the do-nothing floor
  experiments/<arm>/train_*.log                 val curve + attention stats
  experiments/<arm>/dino_stability.csv          gate telemetry
  results/throwaway_crossattn_late_checkpoint/  the 176k late-checkpoint probe

Writes ONLY into results/comparisons/crossattn_report/. Nothing under
experiments/ is opened for writing; no existing figure is overwritten.

THREE FIGURES
  1_results   what the arm scores, on both protocols, both splits, both PSNR
              definitions, plus the paired per-image delta against E0.
  2_mechanism why it failed: the val curve that peaked at iteration 4,000, the
              attention-entropy collapse, and the norm co-inflation the gate
              never caught.
  3_qualitative  images. Cases are picked from E0's OWN ranking so the choice
              cannot be cherry-picked for or against any DINO arm.
"""

import csv
import glob
import json
import os
import re

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE3 = os.path.dirname(_HERE)
_REPO = os.path.dirname(os.path.dirname(_PHASE3))
DATASET = '/home/woody/iwnt/iwnt174h/thesis_dino/holographic_image_dataset'
OUT = os.path.join(_PHASE3, 'results', 'comparisons', 'crossattn_report')

E0 = 'Holo_E0_fixed128_baseline'
ADD = 'Holo_E1_addition_render_fixed128_spatial_B6_latent'
XAT = 'Holo_crossattn_render_fixed128_spatial_B6_latent'
ARMS = [('E0-Fixed', E0, '#4c4c4c'),
        ('addition-render', ADD, '#2b7bba'),
        ('crossattn-render', XAT, '#c44e52')]
CELLS = [('full256', 'val'), ('full256', 'test'),
         ('crop128', 'val'), ('crop128', 'test')]

plt.rcParams.update({'font.size': 12})
BADGE = dict(boxstyle='round,pad=0.34', facecolor='black', alpha=0.75,
             edgecolor='none')


# ---------------------------------------------------------------- readers ---
def summary(exp, protocol, split):
    p = os.path.join(_PHASE3, 'results', exp, 'metrics',
                     f'{protocol}_{split}_summary.json')
    return json.load(open(p)) if os.path.isfile(p) else None


def per_image(exp, protocol, split):
    p = os.path.join(_PHASE3, 'results', exp, 'metrics',
                     f'{protocol}_{split}_per_image.csv')
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return {r['filename']: r for r in csv.DictReader(f)}


def input_baseline(protocol, split):
    p = os.path.join(_PHASE3, 'results', 'comparisons', 'input_baseline',
                     f'input_{protocol}_{split}.csv')
    if not os.path.isfile(p):
        return {}
    with open(p) as f:
        return {r['filename']: r for r in csv.DictReader(f)}


ITER_RE = re.compile(r'iter:\s*([\d,]+)')
VAL_RE = re.compile(r'Validation ValSet.*?psnr:\s*([\d.]+)\s+.*?ssim:\s*([\d.]+)')
STAT_RE = re.compile(r'dino/(\w+):\s*([\d.eE+-]+)')


def read_log(exp):
    """-> (val {iter: psnr}, stats {name: {iter: value}}) from the train logs."""
    val, stats, it = {}, {}, None
    for f in sorted(glob.glob(os.path.join(_REPO, 'experiments', exp,
                                           'train_*.log'))):
        with open(f) as fh:
            for ln in fh:
                m = ITER_RE.search(ln)
                if m:
                    it = int(m.group(1).replace(',', ''))
                if it is not None:
                    for k, v in STAT_RE.findall(ln):
                        stats.setdefault(k, {})[it] = float(v)
                v = VAL_RE.search(ln)
                if v and it is not None:
                    val[it] = float(v.group(1))
    return val, stats


def curve(d):
    ks = sorted(d)
    return np.array(ks), np.array([d[k] for k in ks])


def load16(p):
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(p)
    return img.squeeze().astype(np.float64) / 65535.


def load8(p):
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(p)
    if img.ndim == 3:
        img = img[:, :, 0]
    return img.astype(np.float64) / 255.


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def badge(ax, text, color='white', loc='top', size=13):
    y, va = (0.965, 'top') if loc == 'top' else (0.035, 'bottom')
    ax.text(0.035, y, text, transform=ax.transAxes, ha='left', va=va,
            color=color, fontsize=size, fontweight='bold', bbox=BADGE)


# ------------------------------------------------------------- figure 1 -----
def figure_results(table):
    fig, axes = plt.subplots(2, 2, figsize=(17, 11))

    for ax, metric, label in ((axes[0, 0], 'psnr_full', 'full-image PSNR (dB)'),
                              (axes[0, 1], 'psnr_mask', 'foreground-masked PSNR (dB)')):
        x = np.arange(len(CELLS))
        w = 0.26
        for i, (name, exp, color) in enumerate(ARMS):
            vals = [table.get((exp, p, s), {}).get(metric, np.nan) for p, s in CELLS]
            bars = ax.bar(x + (i - 1) * w, vals, w, label=name, color=color)
            base = [table.get((E0, p, s), {}).get(metric, np.nan) for p, s in CELLS]
            for b, v, e in zip(bars, vals, base):
                if np.isnan(v):
                    continue
                txt = f'{v:.2f}' if exp == E0 else f'{v:.2f}\n{v - e:+.2f}'
                ax.text(b.get_x() + b.get_width() / 2, v + 0.15, txt,
                        ha='center', va='bottom', fontsize=10.5,
                        fontweight='bold',
                        color='#4c4c4c' if exp == E0 else
                        ('#1a7f37' if v > e else '#c0392b'))
        ax.set_xticks(x)
        ax.set_xticklabels([f'{p}\n{s}' for p, s in CELLS])
        ax.set_ylabel(label)
        ax.set_ylim(0, 29)
        ax.grid(axis='y', alpha=0.3)
        ax.set_title(label + '  — bar labels give the delta vs E0', fontsize=12)
        if metric == 'psnr_full':
            handles, labels_ = ax.get_legend_handles_labels()
            fig.legend(handles, labels_, loc='upper center', ncol=3,
                       fontsize=12, bbox_to_anchor=(0.5, 0.925), frameon=False)

    for ax, (protocol, split) in ((axes[1, 0], ('full256', 'test')),
                                  (axes[1, 1], ('crop128', 'test'))):
        a = per_image(XAT, protocol, split)
        b = per_image(E0, protocol, split)
        common = sorted(set(a) & set(b))
        d = np.array([float(a[f]['psnr_full']) - float(b[f]['psnr_full'])
                      for f in common])
        lo, hi = boot_ci(d)
        ax.hist(d, bins=45, color='#c44e52', alpha=0.85)
        ax.axvline(0, color='black', lw=1.5)
        ax.axvline(d.mean(), color='#1a1a1a', ls='--', lw=2,
                   label=f'mean {d.mean():+.3f} dB  95% CI [{lo:+.3f}, {hi:+.3f}]')
        wins = int((d > 0).sum())
        ax.set_title(f'crossattn − E0, paired per image · {protocol}/{split} '
                     f'(n={len(d)})\nimproves {wins}/{len(d)} '
                     f'({100 * wins / len(d):.1f}%)', fontsize=12)
        ax.set_xlabel('Delta PSNR (dB)')
        ax.set_ylabel('images')
        ax.legend(fontsize=10.5, loc='upper left')
        ax.grid(alpha=0.3)

    fig.suptitle('crossattn-render (radar queries, DINO key/value) — evaluated at its '
                 'best-VALIDATION checkpoint, iteration 4,000\n'
                 'uint16 evaluation path; never comparable to the 8-bit training-time '
                 'val PSNR', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.905])
    p = os.path.join(OUT, '1_results.png')
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


# ------------------------------------------------------------- figure 2 -----
def figure_mechanism(best):
    val_x, stats_x = read_log(XAT)
    val_e0, _ = read_log(E0)
    val_ad, _ = read_log(ADD)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    ax = axes[0]
    for label, d, color in (('E0-Fixed', val_e0, '#4c4c4c'),
                            ('addition-render', val_ad, '#2b7bba'),
                            ('crossattn-render', val_x, '#c44e52')):
        if not d:
            continue
        k, v = curve(d)
        ax.plot(k / 1000, v, color=color, lw=1.8, label=label)
    bi, bp = best['best_iter'], best['best_val_psnr']
    ax.plot([bi / 1000], [bp], 'o', ms=11, mfc='none', mec='#c44e52', mew=2.5)
    ax.annotate(f'best val = iter {bi:,} ({bp:.2f} dB),\nnever beaten in the next '
                f'{176000 - bi:,}',
                xy=(bi / 1000, bp), xytext=(72, 19.3), fontsize=10.5,
                color='#c44e52', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#c44e52', lw=1.8))
    ax.axvline(92, color='#888', ls=':', lw=1.8)
    ax.text(94, 13.9, 'LR restart (92k)', color='#666', fontsize=10.5)
    ax.set_xlabel('iteration (thousands)')
    ax.set_ylabel('validation PSNR (dB, 8-bit training-time path)')
    ax.set_title('It peaked at iteration 4,000 and decayed', fontsize=12.5)
    ax.legend(fontsize=10.5, loc='lower right')
    ax.grid(alpha=0.3)

    ax = axes[1]
    k, e = curve(stats_x['attn_entropy'])
    ax.plot(k / 1000, e, color='#c44e52', lw=2, label='attn_entropy')
    if 'attn_uniform_entropy' in stats_x:
        u = curve(stats_x['attn_uniform_entropy'])[1]
        ax.axhline(u[0], color='#888', ls='--', lw=1.6,
                   label=f'uniform attention = {u[0]:.3f} (ln 256)')
    ax.set_xlabel('iteration (thousands)')
    ax.set_ylabel('attention entropy (nats)', color='#c44e52')
    ax.set_ylim(0, 6)
    ax2 = ax.twinx()
    kd, dm = curve(stats_x['attn_diag_mass'])
    ax2.plot(kd / 1000, dm, color='#2b7bba', lw=2, label='attn_diag_mass')
    ax2.axhline(1.0, color='#2b7bba', ls=':', lw=1.5)
    ax2.text(5, 1.02, 'diag_mass = 1.0 would be perfect positional routing',
             color='#2b7bba', fontsize=9.5)
    ax2.set_ylabel('attention diagonal mass', color='#2b7bba')
    ax2.set_ylim(0, 1.15)
    ax.set_title(f'Entropy collapsed {e[0]:.2f} → {e[-1]:.3f} '
                 f'({100 * (1 - e[-1] / e[0]):.0f}%) while diagonal mass\n'
                 f'reached only {dm.max():.2f}: confident routing to the WRONG '
                 f'positions', fontsize=12.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=10, loc='center right')
    ax.grid(alpha=0.3)

    ax = axes[2]
    rows = list(csv.DictReader(open(os.path.join(
        _REPO, 'experiments', XAT, 'dino_stability.csv'))))
    it = np.array([float(r['iter']) for r in rows]) / 1000
    ln = np.array([float(r['latent_norm']) for r in rows])
    pn = np.array([float(r['projected_norm']) for r in rows])
    ratio = np.array([float(r['injection_ratio']) for r in rows])
    ax.plot(it, ln, color='#4c4c4c', lw=2, label='latent_norm')
    ax.plot(it, pn, color='#2b7bba', lw=2, label='projected_norm')
    ax.set_yscale('log')
    ax.set_xlabel('iteration (thousands)')
    ax.set_ylabel('norm (log scale)')
    ax3 = ax.twinx()
    ax3.plot(it, ratio, color='#c44e52', lw=2, label='injection_ratio')
    ax3.axhline(10.0, color='#c44e52', ls='--', lw=1.8,
                label='gate rule 1 cap = 10.0 (never approached)')
    ax3.set_ylabel('injection_ratio', color='#c44e52')
    ax3.set_ylim(0, 11)
    ref = int(np.argmin(np.abs(it - 5.0)))          # the gate's own 5k reference
    ax3.annotate(f'injection_ratio never leaves '
                 f'{ratio.min():.2f}–{ratio.max():.2f}',
                 xy=(120, ratio[-1]), xytext=(60, 2.4), fontsize=10.5,
                 color='#c44e52', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#c44e52', lw=1.6))
    ax.set_title(f'vs the gate\'s 5k reference: latent_norm {ln[-1] / ln[ref]:.0f}x, '
                 f'projected_norm {pn[-1] / pn[ref]:.0f}x\n'
                 f'— but only their RATIO is gated, so nothing fired',
                 fontsize=12.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=10, loc='center right')
    ax.grid(alpha=0.3)

    fig.suptitle('Why crossattn-render failed — three views of the same run '
                 '(179k iterations, stopped early)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p = os.path.join(OUT, '2_mechanism.png')
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


# ------------------------------------------------------------- figure 3 -----
def figure_qualitative(protocol='full256', split='test'):
    tables = {name: per_image(exp, protocol, split) for name, exp, _ in ARMS}
    inp = input_baseline(protocol, split)
    common = sorted(set.intersection(*(set(t) for t in tables.values())))
    ranked = sorted(common, key=lambda fn: float(tables['E0-Fixed'][fn]['psnr_full']))
    n = len(ranked) - 1
    chosen = [('worst for E0', ranked[0]),
              ('median for E0', ranked[n // 2]),
              ('best for E0', ranked[n])]

    gt_dir = os.path.join(DATASET, f'{split}_clean')
    in_dir = os.path.join(DATASET, f'{split}_verynoisy')
    rn_dir = os.path.join(DATASET, f'{split}_renders_blackbg')

    cols = ['1e5 input', 'render (DINO sees this)', 'E0-Fixed',
            'addition-render', 'crossattn-render', '1e7 target']
    fig, axes = plt.subplots(len(chosen), len(cols),
                             figsize=(4.2 * len(cols), 4.6 * len(chosen)))
    for r, (tag, fn) in enumerate(chosen):
        noisy, gt = load16(os.path.join(in_dir, fn)), load16(os.path.join(gt_dir, fn))
        render = load8(os.path.join(rn_dir, fn))
        preds = [(name, load16(os.path.join(_PHASE3, 'results', exp, 'predictions',
                                            f'{protocol}_{split}', 'raw', fn)),
                  float(tables[name][fn]['psnr_full']),
                  float(tables[name][fn]['psnr_mask']))
                 for name, exp, _ in ARMS]
        vmax = max([gt.max(), noisy.max()] + [p.max() for _, p, _, _ in preds])

        axes[r, 0].imshow(noisy, cmap='inferno', vmin=0, vmax=vmax)
        if fn in inp:
            badge(axes[r, 0], f"{float(inp[fn]['psnr_full']):.2f} full\n"
                              f"{float(inp[fn]['psnr_mask']):.2f} mask", '#ffd166')
        axes[r, 0].set_ylabel(f'{tag}\n{fn}', fontsize=14, fontweight='bold',
                              labelpad=16)
        axes[r, 1].imshow(render, cmap='gray', vmin=0, vmax=1)
        for c, (name, img, psnr, pmask) in enumerate(preds, start=2):
            axes[r, c].imshow(img, cmap='inferno', vmin=0, vmax=vmax)
            badge(axes[r, c], f'{psnr:.2f} full\n{pmask:.2f} mask')
            if name != 'E0-Fixed':
                d, dm = psnr - preds[0][2], pmask - preds[0][3]
                badge(axes[r, c], f'{d:+.2f} full / {dm:+.2f} mask  vs E0',
                      loc='bottom', size=11.5,
                      color='#7CFC9A' if d > 0 else '#FF8A80')
        axes[r, len(cols) - 1].imshow(gt, cmap='inferno', vmin=0, vmax=vmax)
        badge(axes[r, len(cols) - 1], 'target', '#ffd166')
        for c in range(len(cols)):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            for sp in axes[r, c].spines.values():
                sp.set_visible(False)
    for c, t in enumerate(cols):
        axes[0, c].set_title(t, fontsize=14, fontweight='bold', pad=10)

    fig.suptitle(f'crossattn-render against E0 and addition-render · {protocol}/{split} · '
                 'cases picked from E0\'s own PSNR ranking, never from a DINO arm\'s',
                 fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    p = os.path.join(OUT, '3_qualitative.png')
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ main ----
def main():
    os.makedirs(OUT, exist_ok=True)
    table, ckpts = {}, {}
    for _, exp, _ in ARMS:
        for protocol, split in CELLS:
            s = summary(exp, protocol, split)
            if s is None:
                continue
            table[(exp, protocol, split)] = {k: v['mean'] for k, v in
                                             s['metrics'].items()}
            ckpts[exp] = s['prediction']['weights'].split('net_g_')[1][:-4]
    best = json.load(open(os.path.join(_PHASE3, 'results', XAT, 'metadata',
                                       'best_checkpoint.json')))

    print('=' * 96)
    print('crossattn-render — evaluated at net_g_%s.pth (best VALIDATION PSNR)'
          % ckpts[XAT])
    print('=' * 96)
    hdr = f'{"cell":<16}{"E0":>10}{"addition":>11}{"crossattn":>11}{"xat-E0":>10}{"n":>7}'
    for metric, label in (('psnr_full', 'full-image PSNR'),
                          ('psnr_mask', 'masked PSNR'),
                          ('ssim_full', 'SSIM'),
                          ('hf_ratio', 'HF-energy ratio')):
        print(f'\n{label}\n{hdr}')
        for protocol, split in CELLS:
            e = table.get((E0, protocol, split), {}).get(metric)
            a = table.get((ADD, protocol, split), {}).get(metric)
            x = table.get((XAT, protocol, split), {}).get(metric)
            s = summary(XAT, protocol, split)
            print(f'{protocol + "/" + split:<16}'
                  f'{e:>10.3f}'
                  f'{(f"{a:.3f}" if a is not None else "--"):>11}'
                  f'{x:>11.3f}{x - e:>+10.3f}{s["n_images"]:>7}')

    print('\npaired per-image, crossattn − E0:')
    for protocol, split in CELLS:
        a, b = per_image(XAT, protocol, split), per_image(E0, protocol, split)
        if a is None or b is None:
            continue
        common = sorted(set(a) & set(b))
        d = np.array([float(a[f]['psnr_full']) - float(b[f]['psnr_full'])
                      for f in common])
        lo, hi = boot_ci(d)
        print(f'  {protocol}/{split:<6} mean {d.mean():+7.3f} dB  '
              f'95% CI [{lo:+.3f}, {hi:+.3f}]  '
              f'improves {int((d > 0).sum())}/{len(d)}')

    print(f'\nselection: best val {best["best_val_psnr"]:.4f} dB at iteration '
          f'{best["best_iter"]:,}; final {best["final_val_psnr"]:.4f} dB at '
          f'{best["final_iter"]:,} ({best["n_val_points"]} val points)')
    late = os.path.join(_PHASE3, 'results', 'throwaway_crossattn_late_checkpoint',
                        'ckpt_176000', 'masked_full256_val.csv')
    if os.path.isfile(late):
        rows = list(csv.DictReader(open(late)))
        lf = np.mean([float(r['psnr_full']) for r in rows])
        lm = np.mean([float(r['psnr_mask']) for r in rows])
        e0v = table[(E0, 'full256', 'val')]
        print(f'late-checkpoint probe (176k, full256/val, n={len(rows)}): '
              f'{lf:.3f} dB full / {lm:.3f} masked '
              f'— {lf - e0v["psnr_full"]:+.3f} vs E0')

    for p in (figure_results(table), figure_mechanism(best),
              figure_qualitative()):
        print('wrote', p)


if __name__ == '__main__':
    main()
