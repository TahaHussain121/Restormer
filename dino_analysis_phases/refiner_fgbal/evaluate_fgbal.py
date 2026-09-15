"""Evaluate, once, the validation-selected foreground-balanced refiner (C)
against frozen E0 (A) and the original refiner_e0 (B).

A and B are refiner_e0's recorded evaluation (same quantisation, same unchanged
masked_metrics.py, same background indicators); C is produced here by the
identical path. Paired statistics use refiner_e0's `paired` (bootstrap 5000,
seed 0, Wilcoxon): sample-level uncertainty over images for ONE training run
each, not training-seed variability.

Window analysis uses the cases and populations fixed in
refiner_e0/qualitative_cases.json before either refiner was trained; nothing is
re-selected. All three methods are scored from their uint16-quantised outputs.
"""

import argparse
import csv
import json
import os

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                     # noqa: E402
import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402

import fgbal_common as fc                                           # noqa: E402
import refiner_common as rc                                         # noqa: E402
from evaluate_refiner import paired, masked_metrics, corr, HIGHER   # noqa: E402
from refiner_arch import ResidualRefinerUNet                        # noqa: E402

LOWER = ['bg_mae', 'bg_rmse', 'bg_frac_pred_gt_0p05', 'bg_frac_pred_gt_0p10']
W = 48
DEV, SPLITS = 'cuda', ('val', 'test')           # --device / --splits; cpu+val only for a dry run
GAMMA, DMAX, EMAX = 0.5, 0.15, 0.30


def u(a):
    return a.astype(np.float64) / 65535.


def window(g, p, e0, cy, cx):
    sl = (slice(cy - W // 2, cy + W // 2), slice(cx - W // 2, cx + W // 2))
    gw, pw, ew = g[sl], p[sl], e0[sl]
    m = gw > rc.FG_THRESHOLD
    mse = float(((gw - pw) ** 2).mean())
    return {'retention': float((pw * m).sum() / max((gw * m).sum(), 1e-12)),
            'target_corr': corr(pw.ravel(), gw.ravel()),
            'psnr': 10 * np.log10(1 / mse) if mse > 0 else float('inf'),
            'false_add_frac': float((pw - gw > 0.05).mean()),
            'mean_change_vs_e0': float((pw - ew).mean())}


def main():
    global DEV, SPLITS
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--splits', nargs='+', default=['val', 'test'], choices=['val', 'test'])
    args = ap.parse_args()
    DEV, SPLITS = args.device, tuple(args.splits)
    rc.strict_fp32()
    os.makedirs(os.path.join(fc.RESULTS, 'metrics'), exist_ok=True)
    with open(fc.SELECTION_JSON) as f:
        sel = json.load(f)
    t_sel = sel['new_rule']['selected_update']
    ck = torch.load(os.path.join(fc.MODEL_DIR, 'refiner_selected.pth'), map_location='cpu',
                    weights_only=False)
    if ck['update'] != t_sel:
        raise SystemExit('refiner_selected.pth does not match selection.json')
    net = ResidualRefinerUNet().to(DEV)
    net.load_state_dict(ck['params'], strict=True)
    net.eval()
    print(f'selected update {t_sel}{" (E0 fallback: identity)" if t_sel == 0 else ""}')

    rep = {'experiment': fc.EXP, 'selected_update': t_sel, 'e0_fallback': t_sel == 0,
           'uncertainty': 'sample-level over images; one training run per refiner; '
                          'NOT training-seed variability',
           'test_note': 'the test split was already inspected for refiner_e0; this is '
                        'not a new untouched holdout',
           'splits': {}, 'device': rc.device_record(DEV), 'created': rc.now()}
    for split in SPLITS:
        ids = rc.split_ids(split)
        y0, _, _ = rc.load_cache(split, ids)
        xu, gu = rc.load_split_uint16(split, ids)
        dC = os.path.join(fc.RESULTS, 'predictions', f'full256_{split}', 'refined_fgbal')
        os.makedirs(dC, exist_ok=True)
        extra = {}
        with torch.no_grad():
            for s in range(0, len(ids), 16):
                x = torch.from_numpy(rc.to_unit(xu[s:s + 16]))[:, None].to(DEV)
                yb = torch.from_numpy(np.ascontiguousarray(y0[s:s + 16]))[:, None].to(DEV)
                y, d = net(x, yb)
                y, d = y[:, 0].cpu().numpy(), d[:, 0].cpu().numpy()
                for j in range(y.shape[0]):
                    k, i = s + j, ids[s + j]
                    qa, qc = rc.quantize_np(y0[k]), rc.quantize_np(y[j])
                    cv2.imwrite(os.path.join(dC, f'{i}.png'), qc)
                    g = u(gu[k])
                    fg = g > rc.FG_THRESHOLD
                    chg = u(qc) - u(qa)
                    extra[i] = {**{f'C_{m}': v for m, v in rc.background_metrics(gu[k], qc).items()},
                                'C_mean_signed_change_fg': float(chg[fg].mean()),
                                'C_frac_fg_brighter': float((chg[fg] > 0).mean()),
                                'C_frac_bg_brighter': float((chg[~fg] > 0).mean()),
                                'C_corr_delta_vs_noisy_minus_e0_bg':
                                    corr(d[j][~fg].astype(np.float64), (u(xu[k]) - y0[k])[~fg]),
                                'C_corr_delta_vs_true_residual_fg':
                                    corr(d[j][fg].astype(np.float64), (g - y0[k])[fg])}
        mC = masked_metrics(dC, os.path.join(rc.DATASET, f'{split}_clean'),
                            os.path.join(fc.RESULTS, 'metrics', f'masked_full256_{split}_fgbal.csv'))
        with open(os.path.join(fc.OLD_RESULTS, 'metrics', f'full256_{split}_per_image.csv')) as f:
            old = {r['filename'][:-4]: r for r in csv.DictReader(f)}
        rows = []
        for i in ids:
            r = {'filename': f'{i}.png'}
            for m in HIGHER:
                r[f'A_{m}'], r[f'B_{m}'], r[f'C_{m}'] = (float(old[i][f'A_{m}']),
                                                        float(old[i][f'B_{m}']), float(mC[i][m]))
            for m in LOWER:
                r[f'A_{m}'], r[f'B_{m}'] = float(old[i][f'A_{m}']), float(old[i][f'B_{m}'])
            r.update(extra[i])
            rows.append(r)
        os.makedirs(os.path.join(fc.RESULTS, 'metrics'), exist_ok=True)
        with open(os.path.join(fc.RESULTS, 'metrics', f'full256_{split}_per_image.csv'), 'w',
                  newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        comp = {}
        for pair, (a, b) in {'C_minus_A': ('A', 'C'), 'C_minus_B': ('B', 'C'),
                             'B_minus_A': ('A', 'B')}.items():
            comp[pair] = {m: paired([r[f'{a}_{m}'] for r in rows], [r[f'{b}_{m}'] for r in rows],
                                    m in HIGHER) for m in HIGHER + LOWER}
        diag = {k: float(np.nanmean([r[k] for r in rows])) for k in extra[ids[0]]
                if not k.startswith('C_bg')}
        rep['splits'][split] = {'n': len(rows), 'paired': comp, 'C_diagnostics': diag,
                                'means': {f'{x}_{m}': float(np.mean([r[f'{x}_{m}'] for r in rows]))
                                          for x in 'ABC' for m in HIGHER + LOWER}}
        c, o = comp['C_minus_A'], comp['C_minus_B']
        print(f'\n[{split}] n={len(rows)}')
        for m in HIGHER + LOWER:
            print(f'  {m:22s} A {c[m]["mean_a"]:.5f}  B {o[m]["mean_a"]:.5f}  C {c[m]["mean_b"]:.5f} | '
                  f'C-A {c[m]["mean_delta_b_minus_a"]:+.5f} [{c[m]["ci95"][0]:+.5f},{c[m]["ci95"][1]:+.5f}] '
                  f'{c[m]["n_improved"]}/{c[m]["n_worsened"]} p={c[m]["wilcoxon_p"]:.1e} | '
                  f'C-B {o[m]["mean_delta_b_minus_a"]:+.5f} [{o[m]["ci95"][0]:+.5f},{o[m]["ci95"][1]:+.5f}] '
                  f'p={o[m]["wilcoxon_p"]:.1e}')
        for k, v in diag.items():
            print(f'  {k:38s} {v:+.5f}')

    # --- windows fixed before training (validation) --------------------------
    with open(rc.CASES_JSON) as f:
        cases = json.load(f)
    ids = rc.split_ids('val')
    dirs = {'E0': os.path.join(fc.OLD_RESULTS, 'predictions', 'full256_val', 'e0'),
            'original': os.path.join(fc.OLD_RESULTS, 'predictions', 'full256_val', 'refined'),
            'fgbal': os.path.join(fc.RESULTS, 'predictions', 'full256_val', 'refined_fgbal')}
    load = lambda d, i: u(rc.load_uint16(os.path.join(d, f'{i}.png')))
    gtv = lambda i: u(rc.load_uint16(os.path.join(rc.DATASET, 'val_clean', f'{i}.png')))
    win = {}
    for grp in ('recovery', 'control'):
        per = {k: [] for k in dirs}
        for wdef in cases['population'][grp]:
            g, e0 = gtv(wdef['id']), load(dirs['E0'], wdef['id'])
            for k, d in dirs.items():
                per[k].append(window(g, load(d, wdef['id']), e0, wdef['cy'], wdef['cx']))
        out = {'n_windows': len(per['E0'])}
        for stat in ('retention', 'target_corr', 'psnr', 'false_add_frac', 'mean_change_vs_e0'):
            for k in dirs:
                out[f'{k}_{stat}_mean'] = float(np.nanmean([w_[stat] for w_ in per[k]]))
            for k in ('original', 'fgbal'):
                a = np.array([w_[stat] for w_ in per['E0']])
                b = np.array([w_[stat] for w_ in per[k]])
                ok = np.isfinite(a) & np.isfinite(b)
                out[f'{k}_minus_E0_{stat}'] = (paired(a[ok], b[ok], stat not in ('false_add_frac',))
                                               if ok.sum() > 1 else None)
        win[grp] = out
        print(f'\n{grp} windows (n={out["n_windows"]}):')
        for stat in ('retention', 'target_corr', 'psnr', 'false_add_frac'):
            f_ = out[f'fgbal_minus_E0_{stat}']
            print(f'  {stat:15s} E0 {out[f"E0_{stat}_mean"]:.4f}  orig {out[f"original_{stat}_mean"]:.4f}  '
                  f'fgbal {out[f"fgbal_{stat}_mean"]:.4f} | fgbal-E0 {f_["mean_delta_b_minus_a"]:+.4f} '
                  f'[{f_["ci95"][0]:+.4f},{f_["ci95"][1]:+.4f}] better {f_["n_improved"]}/worse {f_["n_worsened"]}')
    rep['windows_val'] = win
    rep['window_contrast_recovery_minus_control'] = {
        s: win['recovery'][f'fgbal_minus_E0_{s}']['mean_delta_b_minus_a']
        - win['control'][f'fgbal_minus_E0_{s}']['mean_delta_b_minus_a']
        for s in ('retention', 'target_corr', 'psnr', 'false_add_frac')}
    print('contrast (recovery - control), fgbal - E0:', rep['window_contrast_recovery_minus_control'])

    # --- figures ---------------------------------------------------------------
    figdir = os.path.join(fc.RESULTS, 'figures')
    os.makedirs(figdir, exist_ok=True)
    pos = {i: k for k, i in enumerate(ids)}
    y0v, _, _ = rc.load_cache('val', ids)
    xv, _ = rc.load_split_uint16('val', ids)

    def delta(i):
        k = pos[i]
        with torch.no_grad():
            return net(torch.from_numpy(rc.to_unit(xv[k]))[None, None].to(DEV),
                       torch.from_numpy(np.ascontiguousarray(y0v[k]))[None, None].to(DEV))[1][0, 0].cpu().numpy()

    cols = ['noisy', 'clean target', 'E0', 'original refiner', 'fg-balanced refiner',
            'fg-bal correction', 'fg-bal - target']

    def panel(ax_row, i, crop=None, wdef=None):
        n = u(xv[pos[i]])
        g, a, b, c = gtv(i), load(dirs['E0'], i), load(dirs['original'], i), load(dirs['fgbal'], i)
        sl = (slice(None), slice(None)) if crop is None else \
            (slice(crop[0], crop[0] + crop[2]), slice(crop[1], crop[1] + crop[2]))
        imgs = [(n, 'img'), (g, 'img'), (a, 'img'), (b, 'img'), (c, 'img'), (delta(i), 'd'), (c - g, 'e')]
        for cidx, (im, kind) in enumerate(imgs):
            A = ax_row[cidx]
            if kind == 'img':
                A.imshow(np.clip(im[sl], 0, 1) ** GAMMA, cmap='inferno', vmin=0, vmax=1)
            else:
                v = DMAX if kind == 'd' else EMAX
                A.imshow(im[sl], cmap='RdBu_r', vmin=-v, vmax=v)
            if wdef is not None:
                A.add_patch(plt.Rectangle((wdef[1] - crop[1] - .5, wdef[0] - crop[0] - .5), wdef[2],
                                          wdef[2], fill=False, ec='cyan', lw=0.8))
            A.set_xticks([]); A.set_yticks([])

    cs = cases['cases']
    fig, ax = plt.subplots(len(cs), 7, figsize=(19, 2.9 * len(cs)))
    rep['cases_val'] = []
    for r, cdef in enumerate(cs):
        panel(ax[r], cdef['id'], cdef['crop_yx_size'], cdef['window_yx_size'])
        wy, wx, _ = cdef['window_yx_size']
        g, e0 = gtv(cdef['id']), load(dirs['E0'], cdef['id'])
        st = {k: window(g, load(d, cdef['id']), e0, wy + W // 2, wx + W // 2) for k, d in dirs.items()}
        rep['cases_val'].append({'role': cdef['role'], 'id': cdef['id'], **st})
        ax[r, 0].set_ylabel(f"{cdef['role']}\n{cdef['id']}", fontsize=9)
        ax[r, 4].set_xlabel(f"PSNR {st['E0']['psnr']:.2f}/{st['original']['psnr']:.2f}/{st['fgbal']['psnr']:.2f}  "
                            f"corr {st['E0']['target_corr']:.2f}/{st['original']['target_corr']:.2f}/"
                            f"{st['fgbal']['target_corr']:.2f}  ret {st['E0']['retention']:.2f}/"
                            f"{st['original']['retention']:.2f}/{st['fgbal']['retention']:.2f}", fontsize=7)
    for cidx, t_ in enumerate(cols):
        ax[0, cidx].set_title(t_, fontsize=10)
    fig.suptitle(f'Cases fixed before refiner_e0 was trained (validation). fg-balanced update {t_sel}. '
                 f'Window stats E0/original/fg-bal. Images gamma {GAMMA}; correction +-{DMAX}; error +-{EMAX}',
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, 'cases_val_three_way.png'), dpi=100)
    plt.close(fig)

    with open(os.path.join(fc.RESULTS, 'metrics', 'full256_val_per_image.csv')) as f:
        pr = list(csv.DictReader(f))
    pr.sort(key=lambda r: float(r['C_psnr_mask']) - float(r['A_psnr_mask']))
    picks = [('worst', pr[0]), ('worst-2', pr[1]), ('median', pr[len(pr) // 2]),
             ('best-2', pr[-2]), ('best', pr[-1])]
    fig, ax = plt.subplots(len(picks), 7, figsize=(19, 2.9 * len(picks)))
    rep['posthoc_val'] = []
    for r, (lab, row) in enumerate(picks):
        i = row['filename'][:-4]
        panel(ax[r], i)
        dp = float(row['C_psnr_mask']) - float(row['A_psnr_mask'])
        ax[r, 0].set_ylabel(f'{lab}\n{i}\nfg {dp:+.2f} dB', fontsize=9)
        rep['posthoc_val'].append({'label': lab, 'id': i, 'delta_fg_psnr_vs_E0': dp})
    for cidx, t_ in enumerate(cols):
        ax[0, cidx].set_title(t_, fontsize=10)
    fig.suptitle('POST HOC: validation frames by foreground-PSNR change (fg-bal - E0). Same scales.',
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, 'posthoc_val_three_way.png'), dpi=90)
    plt.close(fig)

    rc.write_json(os.path.join(fc.RESULTS, 'metrics', 'summary_fgbal.json'), rep)
    print('FGBAL EVAL DONE')


if __name__ == '__main__':
    main()
