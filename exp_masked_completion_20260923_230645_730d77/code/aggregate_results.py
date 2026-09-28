"""Aggregate the per-window / per-image metric CSVs into the report tables.

Written after the evaluation because `evaluate_completion.py`'s own `windows`
summary block keyed only by (group, variant) and therefore averaged the methods
together -- a bug in that summary table, not in the data. The per-window CSV it
wrote is complete, so the tables here are recomputed from it. The paired
comparisons in evaluation_summary.json were keyed per method and are correct;
they are recomputed here as well so every number in the report comes from one
place.

Read-only over the CSVs; writes only inside the experiment root.
"""
import argparse, csv, json, os, sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mc_common as mc                                        # noqa: E402
import metrics_lib as ml                                      # noqa: E402

METRICS = ['psnr', 'mae', 'fill_ratio', 'frac_above_0p02', 'frac_above_0p05',
           'overshoot_frac', 'mean_positive_excess', 'target_corr', 'detail_corr',
           'grad_ncc', 'iou_0p05', 'hit_rate_0p05', 'mean_signed_change_vs_base',
           'max_abs_change_outside_mask', 'false_add_frac', 'n_mask_px']
ORDER = ['E0', 'REF', 'FGB', 'FILL-RING', 'FILL-DIFF', 'FILL-TELEA', 'COMP', 'COMP-SHUF']


def fl(v):
    if v in ('', None):
        return None
    try:
        f = float(v)
    except ValueError:
        return None
    return f if np.isfinite(f) else None


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def table(rows, keyf):
    out = defaultdict(list)
    for r in rows:
        out[keyf(r)].append(r)
    res = {}
    for k, rs in out.items():
        d = {'n_rows': len(rs)}
        for m in METRICS:
            v = [fl(r.get(m)) for r in rs]
            v = [x for x in v if x is not None]
            if v:
                d[m] = float(np.mean(v))
                d[m + '__n_defined'] = len(v)
                d[m + '__n_total'] = len(rs)
        res['|'.join(k)] = d
    return res


def paired_all(rows, keyf, ref, tests):
    by = defaultdict(dict)
    for r in rows:
        by[(keyf(r), r['method'])][r['id']] = r
    out = {}
    for (k, meth) in list(by):
        if meth not in tests:
            continue
        a = by.get((k, ref), {})
        common = sorted(set(a) & set(by[(k, meth)]))
        blk = {'n_paired': len(common), 'reference': ref}
        for m in METRICS[:-2]:
            aa = [fl(a[i].get(m)) for i in common]
            bb = [fl(by[(k, meth)][i].get(m)) for i in common]
            aa = [np.nan if x is None else x for x in aa]
            bb = [np.nan if x is None else x for x in bb]
            blk[m] = ml.paired(aa, bb, m in ml.HIGHER)
        out[f'{"|".join(k)}|{meth}_vs_{ref}'] = blk
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    tag = ('_' + a.tag) if a.tag else ''
    M = os.path.join(mc.RESULTS, 'metrics')
    syn = read(os.path.join(M, f'synthetic_per_image{tag}.csv'))
    win = read(os.path.join(M, f'windows_per_window{tag}.csv'))
    syn = [r for r in syn if r['method'] != 'EMPTY_MASK']

    out = {
        'note': 'means over windows/images where the metric is defined; '
                'n_defined vs n_total is reported for every metric',
        'synthetic_by_method': table(syn, lambda r: (r['method'],)),
        'synthetic_paired_vs_INPUT-DAMAGED': paired_all(
            syn, lambda r: ('synthetic',), 'INPUT-DAMAGED',
            ['COMP', 'FILL-RING', 'FILL-DIFF', 'FILL-TELEA']),
        'synthetic_paired_vs_FILL-DIFF': paired_all(
            syn, lambda r: ('synthetic',), 'FILL-DIFF', ['COMP']),
        'windows_by_group_variant_method': table(
            win, lambda r: (r['group'], r['variant'], r['method'])),
        'windows_paired_vs_E0': paired_all(
            win, lambda r: (r['group'], r['variant']), 'E0',
            ['COMP', 'COMP-SHUF', 'FILL-RING', 'FILL-DIFF', 'FILL-TELEA', 'REF', 'FGB']),
        'windows_paired_vs_FILL-DIFF': paired_all(
            win, lambda r: (r['group'], r['variant']), 'FILL-DIFF', ['COMP']),
        'windows_paired_vs_FILL-RING': paired_all(
            win, lambda r: (r['group'], r['variant']), 'FILL-RING', ['COMP']),
        'windows_paired_COMP_vs_COMP-SHUF': paired_all(
            win, lambda r: (r['group'], r['variant']), 'COMP', ['COMP-SHUF']),
        'created': mc.now(),
    }
    mc.write_json_exclusive(os.path.join(mc.RESULTS, f'aggregate_tables{tag}.json'), out)

    lines = []
    cols = ['psnr', 'mae', 'fill_ratio', 'frac_above_0p02', 'target_corr', 'detail_corr',
            'grad_ncc', 'overshoot_frac', 'max_abs_change_outside_mask']
    lines.append('SYNTHETIC COMPLETION TASK (339 validation images, fixed masks)')
    lines.append(f'{"method":13s}' + ''.join(f'{c[:12]:>13s}' for c in cols))
    for m in ['INPUT-DAMAGED'] + ORDER:
        d = out['synthetic_by_method'].get(m)
        if d:
            lines.append(f'{m:13s}' + ''.join(
                (f'{d[c]:>13.4f}' if c in d else f'{"-":>13s}') for c in cols))
    for variant in ('oracle', 'expanded', 'expanded_ring', 'background'):
        for grp in ('recovery', 'control'):
            lines.append('')
            lines.append(f'{grp.upper()} windows / mask = {variant}')
            lines.append(f'{"method":13s}' + ''.join(f'{c[:12]:>13s}' for c in cols))
            for m in ORDER:
                d = out['windows_by_group_variant_method'].get(f'{grp}|{variant}|{m}')
                if d:
                    lines.append(f'{m:13s}' + ''.join(
                        (f'{d[c]:>13.4f}' if c in d else f'{"-":>13s}') for c in cols))
    txt = '\n'.join(lines)
    with mc.open_exclusive(os.path.join(mc.RESULTS, f'tables{tag}.txt')) as f:
        f.write(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
