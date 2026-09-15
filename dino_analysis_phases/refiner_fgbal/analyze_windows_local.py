"""Additional window analysis requested by the author on 2026-09-15.

Written BEFORE any fg-balanced window output was viewed; it is not part of the
pre-registration and is reported as an added, descriptive analysis. Validation
only. Windows and display cases: refiner_e0/qualitative_cases.json, fixed before
refiner_e0 was trained (228 recovery, 191 control windows of 48x48; 8 cases).
Every method is scored from its uint16-quantised evaluation output:
E0 and refiner_e0 from refiner_e0's evaluation, fg-balanced from its own.

Definitions (c = method - E0, err = |method - target|, all in [0,1] units):
  change          |c| > TAU = 0.005 (about 328 uint16 levels); smaller is "unchanged"
  positive/negative corrections, each with pixel fraction, mean magnitude, mean
                  change in error (negative = closer to the target) and the
                  fraction of those pixels whose error fell
  supported brightening    c > TAU where target > E0 and target > 0.01
  unsupported brightening  c > TAU elsewhere (target <= E0, or background)
  overshoot                c > TAU and method - target > 0.05
  suppressing weak structure  c < -TAU where target > E0 and target > 0.01
  justified darkening         c < -TAU where E0 > target
  missing structure pixels    target > 0.05 and E0 < 0.02 (E0 removed them):
                  fill ratio sum(method)/sum(target) and fraction above 0.02
  existing structure pixels   target > 0.05 and E0 >= 0.02: mean change, error change
  target correlation  Pearson over the window. UNDEFINED when the prediction
                  (or target) is constant in the window; reported two ways:
                  (a) undefined windows excluded, (b) a constant prediction
                  scored 0 (it carries no structure). Counts are reported.
  false additions     fraction of window pixels with method - target > 0.05
  retention           sum(method*m)/sum(target*m), m = target > 0.01 -- an
                  INTENSITY measure, not evidence of geometric recovery.
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                     # noqa: E402
import numpy as np                                                  # noqa: E402

import fgbal_common as fc                                           # noqa: E402
import refiner_common as rc                                         # noqa: E402
from evaluate_refiner import paired                                 # noqa: E402

TAU, MISS_GT, MISS_E0, OVER, W = 0.005, 0.05, 0.02, 0.05, 48
GAMMA, DMAX, EMAX = 0.5, 0.15, 0.30
METHODS = ('E0', 'original', 'fgbal')
DIRS = {'E0': os.path.join(fc.OLD_RESULTS, 'predictions', 'full256_val', 'e0'),
        'original': os.path.join(fc.OLD_RESULTS, 'predictions', 'full256_val', 'refined'),
        'fgbal': os.path.join(fc.RESULTS, 'predictions', 'full256_val', 'refined_fgbal')}


def u(a):
    return a.astype(np.float64) / 65535.


def load(m, i):
    return u(rc.load_uint16(os.path.join(DIRS[m], f'{i}.png')))


def target(i):
    return u(rc.load_uint16(os.path.join(rc.DATASET, 'val_clean', f'{i}.png')))


def pearson(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return (float((a * b).sum() / den), False) if den > 0 else (np.nan, True)


def mean_or_nan(x):
    return float(x.mean()) if x.size else np.nan


def window_stats(g, e0, p, cy, cx):
    sl = (slice(cy - W // 2, cy + W // 2), slice(cx - W // 2, cx + W // 2))
    g, e0, p = g[sl], e0[sl], p[sl]
    c = p - e0
    derr = np.abs(p - g) - np.abs(e0 - g)
    m = g > rc.FG_THRESHOLD
    under = (g > e0) & m
    pos, neg = c > TAU, c < -TAU
    r, const = pearson(p, g)
    mse = float(((p - g) ** 2).mean())
    miss = (g > MISS_GT) & (e0 < MISS_E0)
    exist = (g > MISS_GT) & (e0 >= MISS_E0)
    s = {'psnr': 10 * np.log10(1 / mse) if mse > 0 else np.inf,
         'mae': float(np.abs(p - g).mean()),
         'corr': r, 'corr_undefined': const, 'corr_const0': 0.0 if const else r,
         'retention': float((p * m).sum() / max((g * m).sum(), 1e-12)),
         'false_add_frac': float((p - g > OVER).mean()),
         'n_missing_px': int(miss.sum()),
         'missing_fill_ratio': float(p[miss].sum() / g[miss].sum()) if miss.any() else np.nan,
         'missing_frac_above_0p02': mean_or_nan((p[miss] > MISS_E0).astype(float)),
         'existing_mean_change': mean_or_nan(c[exist]),
         'existing_err_change': mean_or_nan(derr[exist])}
    for name, sel in (('pos', pos), ('neg', neg)):
        s[f'{name}_frac'] = float(sel.mean())
        s[f'{name}_mean_mag'] = mean_or_nan(np.abs(c[sel]))
        s[f'{name}_err_change'] = mean_or_nan(derr[sel])
        s[f'{name}_frac_err_reduced'] = mean_or_nan((derr[sel] < 0).astype(float))
    s['pos_supported_frac'] = float((pos & under).mean())
    s['pos_unsupported_frac'] = float((pos & ~under).mean())
    s['pos_overshoot_frac'] = float((pos & (p - g > OVER)).mean())
    s['neg_suppress_weak_frac'] = float((neg & under).mean())
    s['neg_justified_frac'] = float((neg & (e0 > g)).mean())
    return s


def main():
    with open(rc.CASES_JSON) as f:
        cases = json.load(f)
    out = {'note': 'author-requested added analysis; defined before viewing fg-balanced '
                   'window outputs; validation only; descriptive',
           'definitions': __doc__, 'groups': {}}
    for grp in ('recovery', 'control'):
        per = {m: [] for m in METHODS}
        base = []
        for w in cases['population'][grp]:
            g, e0 = target(w['id']), load('E0', w['id'])
            for m in METHODS:
                per[m].append(window_stats(g, e0, load(m, w['id']) if m != 'E0' else e0,
                                           w['cy'], w['cx']))
            base.append((w['gt_mean'], w['noisy_support_corr']))
        keys = [k for k in per['E0'][0] if k != 'corr_undefined']
        G = {'n_windows': len(per['E0']),
             'starting_condition': {'gt_mean': float(np.mean([b[0] for b in base])),
                                    'noisy_support_corr': float(np.mean([b[1] for b in base]))},
             'mean': {m: {k: float(np.nanmean([s[k] for s in per[m]])) for k in keys} for m in METHODS},
             'n_corr_undefined': {m: int(sum(s['corr_undefined'] for s in per[m])) for m in METHODS},
             'paired_vs_E0': {}}
        for m in ('original', 'fgbal'):
            G['paired_vs_E0'][m] = {}
            for k, hib in (('psnr', True), ('mae', False), ('corr', True), ('corr_const0', True),
                           ('retention', True), ('false_add_frac', False),
                           ('missing_fill_ratio', True)):
                a = np.array([s[k] for s in per['E0']], float)
                b = np.array([s[k] for s in per[m]], float)
                ok = np.isfinite(a) & np.isfinite(b)
                G['paired_vs_E0'][m][k] = paired(a[ok], b[ok], hib) if ok.sum() > 1 else None
        out['groups'][grp] = G
        print(f'\n== {grp}: {G["n_windows"]} windows; start: gt mean {G["starting_condition"]["gt_mean"]:.3f}, '
              f'noisy support {G["starting_condition"]["noisy_support_corr"]:+.2f}; corr undefined '
              f'{G["n_corr_undefined"]}')
        for k in keys:
            print(f'  {k:26s} ' + '  '.join(f'{m} {G["mean"][m][k]:+.4f}' for m in METHODS))
        for m in ('original', 'fgbal'):
            for k, v in G['paired_vs_E0'][m].items():
                if v:
                    print(f'  {m:8s}-E0 {k:18s} {v["mean_delta_b_minus_a"]:+.4f} [{v["ci95"][0]:+.4f}, '
                          f'{v["ci95"][1]:+.4f}] better {v["n_improved"]} worse {v["n_worsened"]} '
                          f'n={v["n"]} p={v["wilcoxon_p"]:.1e}')

    # --- enlarged figure, identical limits for corresponding panels -----------
    cs = cases['cases']
    cols = ['target (full frame)', 'noisy', 'target', 'E0', 'original refiner', 'fg-balanced',
            'orig - E0', 'fg-bal - E0', 'E0 - target', 'orig - target', 'fg-bal - target']
    fig, ax = plt.subplots(len(cs), len(cols), figsize=(27, 2.6 * len(cs)))
    out['cases'] = []
    for r, cd in enumerate(cs):
        i = cd['id']
        y, x, sz = cd['crop_yx_size']
        wy, wx, ws = cd['window_yx_size']
        g = target(i)
        n = u(rc.load_uint16(os.path.join(rc.DATASET, 'val_verynoisy', f'{i}.png')))
        P = {m: load(m, i) for m in METHODS}
        st = {m: window_stats(g, P['E0'], P[m], wy + ws // 2, wx + ws // 2) for m in METHODS}
        out['cases'].append({'role': cd['role'], 'id': i, **st})
        sl = (slice(y, y + sz), slice(x, x + sz))
        panels = [(g, 'full'), (n, 'img'), (g, 'img'), (P['E0'], 'img'), (P['original'], 'img'),
                  (P['fgbal'], 'img'), (P['original'] - P['E0'], 'd'), (P['fgbal'] - P['E0'], 'd'),
                  (P['E0'] - g, 'e'), (P['original'] - g, 'e'), (P['fgbal'] - g, 'e')]
        for cidx, (im, kind) in enumerate(panels):
            A = ax[r, cidx]
            if kind == 'full':
                A.imshow(np.clip(im, 0, 1) ** GAMMA, cmap='inferno', vmin=0, vmax=1)
                A.add_patch(plt.Rectangle((x - .5, y - .5), sz, sz, fill=False, ec='cyan', lw=0.8))
            elif kind == 'img':
                A.imshow(np.clip(im[sl], 0, 1) ** GAMMA, cmap='inferno', vmin=0, vmax=1)
            else:
                v = DMAX if kind == 'd' else EMAX
                A.imshow(im[sl], cmap='RdBu_r', vmin=-v, vmax=v)
            if kind != 'full':
                A.add_patch(plt.Rectangle((wx - x - .5, wy - y - .5), ws, ws, fill=False,
                                          ec='cyan', lw=0.7))
            A.set_xticks([]); A.set_yticks([])
            if r == 0:
                A.set_title(cols[cidx], fontsize=9)
        ax[r, 0].set_ylabel(f"{cd['role']}\n{i}", fontsize=9)
        ax[r, 5].set_xlabel('PSNR ' + '/'.join(f"{st[m]['psnr']:.2f}" for m in METHODS) +
                            '  corr ' + '/'.join('undef' if st[m]['corr_undefined'] else f"{st[m]['corr']:.2f}"
                                                 for m in METHODS), fontsize=7)
    fig.suptitle(f'Fixed validation cases (chosen before refiner_e0 was trained), 96x96 enlargements; '
                 f'cyan = 48x48 window. Images gamma {GAMMA} on [0,1] (all methods); corrections +-{DMAX}; '
                 f'errors +-{EMAX}. Window stats E0/original/fg-bal', fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.join(fc.RESULTS, 'figures'), exist_ok=True)
    plt.savefig(os.path.join(fc.RESULTS, 'figures', 'cases_val_enlarged_all_methods.png'), dpi=85)
    plt.close(fig)
    for c_ in out['cases']:
        print(c_['role'], c_['id'], {m: {k: round(c_[m][k], 3) if isinstance(c_[m][k], float) else c_[m][k]
                                         for k in ('psnr', 'corr', 'retention', 'false_add_frac',
                                                   'missing_fill_ratio', 'pos_frac', 'neg_frac')}
                                     for m in METHODS})
    rc.write_json(os.path.join(fc.RESULTS, 'metrics', 'window_local_analysis.json'), out)
    print('WINDOW ANALYSIS DONE')


if __name__ == '__main__':
    main()
