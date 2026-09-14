"""Qualitative panels (validation only) and the window-level recovery/control
comparison, both defined in qualitative_cases.json BEFORE training.

Panel columns, identical display scales and crop per row:
  noisy | clean target | E0 | refined | signed correction | E0 - target | refined - target
Intensities: gamma 0.5 on [0, 1] for all four image columns. Correction: +-0.15.
Errors: +-0.30. A second figure shows post-hoc full frames (best, median and
worst validation PSNR change), labelled post hoc.
"""

import csv
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                               # noqa: E402
import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402

import refiner_common as rc                                   # noqa: E402
from refiner_arch import ResidualRefinerUNet                  # noqa: E402

GAMMA, DMAX, EMAX = 0.5, 0.15, 0.30


def win_stats(g, p, n, ys, xs, size):
    sl = (slice(ys, ys + size), slice(xs, xs + size))
    gw, pw = g[sl], p[sl]
    m = gw > rc.FG_THRESHOLD
    ret = float((pw * m).sum() / max((gw * m).sum(), 1e-12))
    mse = float(((gw - pw) ** 2).mean())
    return {'retention': ret, 'psnr': 10 * np.log10(1 / mse) if mse > 0 else float('inf')}


def main():
    rc.strict_fp32()
    with open(rc.CASES_JSON) as f:
        cases = json.load(f)
    with open(os.path.join(rc.EXP_DIR, 'train_summary.json')) as f:
        summ = json.load(f)
    ck = torch.load(summ['selected_checkpoint'], map_location='cpu', weights_only=False)
    net = ResidualRefinerUNet()
    net.load_state_dict(ck['params'])
    net.eval()
    ids = rc.split_ids('val')
    pos = {i: k for k, i in enumerate(ids)}
    y0, _, _ = rc.load_cache('val', ids)
    xu, gu = rc.load_split_uint16('val', ids)

    def run(i):
        k = pos[i]
        x = torch.from_numpy(rc.to_unit(xu[k]))[None, None]
        yb = torch.from_numpy(np.ascontiguousarray(y0[k]))[None, None]
        with torch.no_grad():
            y, d = net(x, yb)
        u = lambda a: a.astype(np.float64) / 65535.
        return (u(xu[k]), u(gu[k]), u(rc.quantize_np(y0[k])),
                u(rc.quantize_np(y[0, 0].numpy())), d[0, 0].numpy().astype(np.float64))

    figdir = os.path.join(rc.RESULTS, 'figures')
    os.makedirs(figdir, exist_ok=True)
    out = {'selected_update': ck['update'], 'cases': [], 'population': {}}

    # --- pre-declared cases -------------------------------------------------
    cs = cases['cases']
    cols = ['noisy', 'clean target', 'E0', 'refined', 'correction (delta)',
            'E0 - target', 'refined - target']
    fig, ax = plt.subplots(len(cs), 7, figsize=(19, 2.9 * len(cs)))
    for r, c in enumerate(cs):
        n, g, a, b, d = run(c['id'])
        y, x, s = c['crop_yx_size']
        wy, wx, ws = c['window_yx_size']
        sa, sb = win_stats(g, a, n, wy, wx, ws), win_stats(g, b, n, wy, wx, ws)
        out['cases'].append({'role': c['role'], 'id': c['id'], 'e0': sa, 'refined': sb})
        sl = (slice(y, y + s), slice(x, x + s))
        panels = [(n, 'img'), (g, 'img'), (a, 'img'), (b, 'img'), (d, 'delta'),
                  (a - g, 'err'), (b - g, 'err')]
        for cidx, (img, kind) in enumerate(panels):
            A = ax[r, cidx]
            if kind == 'img':
                A.imshow(np.clip(img[sl], 0, 1) ** GAMMA, cmap='inferno', vmin=0, vmax=1)
            elif kind == 'delta':
                A.imshow(img[sl], cmap='RdBu_r', vmin=-DMAX, vmax=DMAX)
            else:
                A.imshow(img[sl], cmap='RdBu_r', vmin=-EMAX, vmax=EMAX)
            A.add_patch(plt.Rectangle((wx - x - .5, wy - y - .5), ws, ws, fill=False,
                                      ec='cyan', lw=0.8))
            A.set_xticks([]); A.set_yticks([])
            if r == 0:
                A.set_title(cols[cidx], fontsize=10)
        ax[r, 0].set_ylabel(f"{c['role']}\n{c['id']}", fontsize=9)
        ax[r, 3].set_xlabel(f"window: retention {sa['retention']:.2f}->{sb['retention']:.2f}, "
                            f"PSNR {sa['psnr']:.2f}->{sb['psnr']:.2f}", fontsize=8)
    fig.suptitle(f'Pre-declared validation cases (fixed before training). Refiner update '
                 f'{ck["update"]}. Images gamma {GAMMA} on [0,1]; correction +-{DMAX}; '
                 f'errors +-{EMAX}; cyan = selection window', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, 'qualitative_val_cases.png'), dpi=110)
    plt.close(fig)

    # --- population: recovery vs control windows, descriptive ----------------
    for grp in ('recovery', 'control'):
        rows = []
        for w in cases['population'][grp]:
            n, g, a, b, d = run(w['id'])
            wy, wx = w['cy'] - 24, w['cx'] - 24
            sa, sb = win_stats(g, a, n, wy, wx, 48), win_stats(g, b, n, wy, wx, 48)
            rows.append((sa['retention'], sb['retention'], sa['psnr'], sb['psnr']))
        R = np.array(rows)
        out['population'][grp] = {
            'n_windows': len(rows),
            'retention_e0_mean': float(R[:, 0].mean()), 'retention_refined_mean': float(R[:, 1].mean()),
            'retention_change_mean': float((R[:, 1] - R[:, 0]).mean()),
            'window_psnr_change_mean': float((R[:, 3] - R[:, 2]).mean()),
            'window_psnr_change_median': float(np.median(R[:, 3] - R[:, 2])),
            'n_window_psnr_improved': int((R[:, 3] > R[:, 2]).sum())}
        print(grp, out['population'][grp])

    # --- post-hoc full frames ------------------------------------------------
    with open(os.path.join(rc.RESULTS, 'metrics', 'full256_val_per_image.csv')) as f:
        pr = list(csv.DictReader(f))
    pr.sort(key=lambda r: float(r['B_psnr_full']) - float(r['A_psnr_full']))
    picks = [('worst', pr[0]), ('worst-2', pr[1]), ('median', pr[len(pr) // 2]),
             ('median+1', pr[len(pr) // 2 + 1]), ('best-2', pr[-2]), ('best', pr[-1])]
    fig, ax = plt.subplots(len(picks), 7, figsize=(19, 2.9 * len(picks)))
    for r, (lab, row) in enumerate(picks):
        i = row['filename'][:-4]
        n, g, a, b, d = run(i)
        dp = float(row['B_psnr_full']) - float(row['A_psnr_full'])
        for cidx, (img, kind) in enumerate([(n, 'img'), (g, 'img'), (a, 'img'), (b, 'img'),
                                            (d, 'delta'), (a - g, 'err'), (b - g, 'err')]):
            A = ax[r, cidx]
            if kind == 'img':
                A.imshow(np.clip(img, 0, 1) ** GAMMA, cmap='inferno', vmin=0, vmax=1)
            else:
                v = DMAX if kind == 'delta' else EMAX
                A.imshow(img, cmap='RdBu_r', vmin=-v, vmax=v)
            A.set_xticks([]); A.set_yticks([])
            if r == 0:
                A.set_title(cols[cidx], fontsize=10)
        ax[r, 0].set_ylabel(f'{lab}\n{i}\n{dp:+.2f} dB', fontsize=9)
        out.setdefault('posthoc', []).append({'label': lab, 'id': i, 'delta_psnr_full': dp})
    fig.suptitle('POST HOC: validation full frames chosen from the results (worst, median, '
                 'best PSNR change). Same display scales.', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, 'posthoc_val_full_frames.png'), dpi=90)
    plt.close(fig)
    rc.write_json(os.path.join(rc.RESULTS, 'metrics', 'qualitative_numbers.json'), out)
    print('PANELS DONE')


if __name__ == '__main__':
    main()
