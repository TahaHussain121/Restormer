"""Evaluate A = frozen E0 against B = E0 + the validation-selected refiner.

Full 256x256 frames only. Both outputs pass through the SAME clamp and uint16
quantisation (predict_phase3's path), are written as PNGs, and are scored by
the UNCHANGED Deraining_Holo/masked_metrics.py (full and foreground PSNR/SSIM,
foreground = gt > 0.01). Background indicators, residual statistics and the
paired comparison are computed here.

Paired statistics follow paired_compare.py: mean of B - A, 95% bootstrap CI of
the mean (5000 resamples, seed 0), two-sided Wilcoxon signed-rank. They are
SAMPLE-LEVEL uncertainty for this one trained run, not training-seed
robustness.

The test split is read only here, after selection is recorded.
"""

import argparse
import csv
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import torch
from scipy import stats

import refiner_common as rc
from refiner_arch import ResidualRefinerUNet

HIGHER = ['psnr_full', 'psnr_mask', 'ssim_full', 'ssim_mask']
LOWER = ['bg_mae', 'bg_rmse', 'bg_mean_pred', 'bg_frac_pred_gt_0p05',
         'bg_frac_pred_gt_0p10']


def paired(a, b, higher_is_better, n_boot=5000, seed=0):
    d = np.asarray(b) - np.asarray(a)
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(d, len(d)).mean() for _ in range(n_boot)])
    better = (d > 0) if higher_is_better else (d < 0)
    worse = (d < 0) if higher_is_better else (d > 0)
    p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return {'mean_a': float(np.mean(a)), 'mean_b': float(np.mean(b)),
            'mean_delta_b_minus_a': float(d.mean()),
            'ci95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            'n': int(len(d)), 'n_improved': int(better.sum()),
            'n_worsened': int(worse.sum()), 'n_equal': int((d == 0).sum()),
            'wilcoxon_p': p, 'higher_is_better': higher_is_better}


def masked_metrics(pred_dir, gt_dir, out_csv):
    subprocess.run([sys.executable, '-u', os.path.join(rc.REPO, 'Deraining_Holo', 'masked_metrics.py'),
                    '--pred_dir', pred_dir, '--gt_dir', gt_dir, '--csv', out_csv],
                   check=True)
    with open(out_csv) as f:
        return {r['filename'][:-4]: r for r in csv.DictReader(f)}


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', nargs='+', default=['val', 'test'])
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    rc.strict_fp32()

    with open(os.path.join(rc.EXP_DIR, 'train_summary.json')) as f:
        summ = json.load(f)
    ck = torch.load(summ['selected_checkpoint'], map_location='cpu', weights_only=False)
    if ck['update'] != summ['selected_update']:
        raise SystemExit('selected checkpoint does not match the training summary')
    net = ResidualRefinerUNet().to(args.device)
    net.load_state_dict(ck['params'], strict=True)
    net.eval()
    print(f'refiner: update {ck["update"]} (validation-selected), val {summ["selected_val"]:.4f}')

    report = {'experiment': rc.EXP, 'selected_update': ck['update'],
              'uncertainty': 'sample-level (images), single trained run; NOT '
                             'training-seed robustness',
              'splits': {}, 'device': rc.device_record(args.device), 'created': rc.now()}
    for split in args.splits:
        ids = rc.split_ids(split)
        y0, _, prov = rc.load_cache(split, ids)
        xu, gu = rc.load_split_uint16(split, ids)
        base = os.path.join(rc.RESULTS, 'predictions', f'full256_{split}')
        dA, dB = os.path.join(base, 'e0'), os.path.join(base, 'refined')
        os.makedirs(dA, exist_ok=True)
        os.makedirs(dB, exist_ok=True)
        extra = {}
        with torch.no_grad():
            for s in range(0, len(ids), 16):
                x = torch.from_numpy(rc.to_unit(xu[s:s + 16]))[:, None].to(args.device)
                yb = torch.from_numpy(np.ascontiguousarray(y0[s:s + 16]))[:, None].to(args.device)
                y, d = net(x, yb)
                y, d = y[:, 0].cpu().numpy(), d[:, 0].cpu().numpy()
                for j in range(y.shape[0]):
                    k, i = s + j, ids[s + j]
                    qa, qb = rc.quantize_np(y0[k]), rc.quantize_np(y[j])
                    cv2.imwrite(os.path.join(dA, f'{i}.png'), qa)
                    cv2.imwrite(os.path.join(dB, f'{i}.png'), qb)
                    g = gu[k].astype(np.float64) / 65535.
                    fg = g > rc.FG_THRESHOLD
                    dd = d[j].astype(np.float64)
                    xx = xu[k].astype(np.float64) / 65535.
                    yy0 = y0[k].astype(np.float64)
                    ba, bb = rc.background_metrics(gu[k], qa), rc.background_metrics(gu[k], qb)
                    chg = (qb.astype(np.float64) - qa.astype(np.float64)) / 65535.
                    extra[i] = {
                        **{f'A_{m}': v for m, v in ba.items()},
                        **{f'B_{m}': v for m, v in bb.items()},
                        'delta_mean_abs_fg': float(np.abs(dd[fg]).mean()),
                        'delta_mean_abs_bg': float(np.abs(dd[~fg]).mean()),
                        'delta_max_abs': float(np.abs(dd).max()),
                        'bg_mean_positive_change': float(np.clip(chg[~fg], 0, None).mean()),
                        # >0: in the background the correction moves TOWARDS the noisy frame
                        'corr_delta_vs_noisy_minus_e0_bg': corr(dd[~fg], (xx - yy0)[~fg]),
                        # >0: the correction points along the true residual (diagnostic only)
                        'corr_delta_vs_true_residual_all': corr(dd.ravel(), (g - yy0).ravel()),
                        'corr_delta_vs_true_residual_fg': corr(dd[fg], (g - yy0)[fg]),
                    }
        gt_dir = os.path.join(rc.DATASET, f'{split}_clean')
        mdir = os.path.join(rc.RESULTS, 'metrics')
        os.makedirs(mdir, exist_ok=True)
        mA = masked_metrics(dA, gt_dir, os.path.join(mdir, f'masked_full256_{split}_e0.csv'))
        mB = masked_metrics(dB, gt_dir, os.path.join(mdir, f'masked_full256_{split}_refined.csv'))

        rows = []
        for i in ids:
            r = {'filename': f'{i}.png', 'mask_frac': float(mA[i]['mask_frac'])}
            for m in HIGHER:
                r[f'A_{m}'], r[f'B_{m}'] = float(mA[i][m]), float(mB[i][m])
            r.update(extra[i])
            rows.append(r)
        with open(os.path.join(mdir, f'full256_{split}_per_image.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

        comp = {}
        for m in HIGHER + LOWER:
            comp[m] = paired([r[f'A_{m}'] for r in rows], [r[f'B_{m}'] for r in rows],
                             m in HIGHER)
        descr = {k: {'mean': float(np.nanmean([r[k] for r in rows])),
                     'median': float(np.nanmedian([r[k] for r in rows]))}
                 for k in ('delta_mean_abs_fg', 'delta_mean_abs_bg', 'delta_max_abs',
                           'bg_mean_positive_change', 'corr_delta_vs_noisy_minus_e0_bg',
                           'corr_delta_vs_true_residual_all', 'corr_delta_vs_true_residual_fg')}
        with open(rc.E0_REF_CSV.format(split=split)) as f:
            refp = {r['filename']: float(r['psnr_full']) for r in csv.DictReader(f)}
        repro = float(max(abs(r['A_psnr_full'] - refp[r['filename']]) for r in rows))
        report['splits'][split] = {
            'n': len(rows), 'paired_b_minus_a': comp, 'residual_and_background': descr,
            'A_reproduces_E0_reference_max_abs_psnr_diff_db': repro,
            'cache_reference_check': prov['reference_check'],
            'meets_practical_target_0p10_full': comp['psnr_full']['mean_delta_b_minus_a'] >= 0.10,
        }
        c = comp['psnr_full']
        print(f'\n[{split}] n={len(rows)}  E0 {c["mean_a"]:.4f}  refined {c["mean_b"]:.4f}  '
              f'delta {c["mean_delta_b_minus_a"]:+.4f}  CI [{c["ci95"][0]:+.4f}, '
              f'{c["ci95"][1]:+.4f}]  better {c["n_improved"]} / worse {c["n_worsened"]}  '
              f'p={c["wilcoxon_p"]:.2e}   (A vs recorded E0: max {repro:.1e} dB)')
        for m in HIGHER[1:] + LOWER:
            c = comp[m]
            print(f'    {m:22s} A {c["mean_a"]:.5f}  B {c["mean_b"]:.5f}  '
                  f'd {c["mean_delta_b_minus_a"]:+.5f}  CI [{c["ci95"][0]:+.5f}, '
                  f'{c["ci95"][1]:+.5f}]  better {c["n_improved"]}/worse {c["n_worsened"]}  '
                  f'p={c["wilcoxon_p"]:.1e}')
        for k, v in descr.items():
            print(f'    {k:34s} mean {v["mean"]:+.5f}  median {v["median"]:+.5f}')
    rc.write_json(os.path.join(rc.RESULTS, 'metrics', 'summary_full256.json'), report)
    print('EVAL DONE')


if __name__ == '__main__':
    main()
