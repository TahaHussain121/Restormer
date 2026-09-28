"""Print the specific numbers the report quotes, from aggregate_tables*.json."""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mc_common as mc                                        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='')
a = ap.parse_args()
tag = ('_' + a.tag) if a.tag else ''
d = json.load(open(os.path.join(mc.RESULTS, f'aggregate_tables{tag}.json')))
T, W = d['synthetic_by_method'], d['windows_by_group_variant_method']


def g(dd, k, m):
    v = dd.get(k, {}).get(m)
    return f'{v:8.4f}' if isinstance(v, float) else '       -'


print('### SYNTHETIC (339 val images, fixed masks)')
for m in ('INPUT-DAMAGED', 'FILL-RING', 'FILL-TELEA', 'FILL-DIFF', 'COMP'):
    print(f'{m:14s} psnr {g(T, m, "psnr")} mae {g(T, m, "mae")} fill {g(T, m, "fill_ratio")} '
          f'corr {g(T, m, "target_corr")} detail {g(T, m, "detail_corr")} '
          f'gradncc {g(T, m, "grad_ncc")} over {g(T, m, "overshoot_frac")} '
          f'(psnr defined {T[m].get("psnr__n_defined")}/{T[m].get("psnr__n_total")})')
for blk, lbl in (('synthetic_paired_vs_INPUT-DAMAGED', 'vs damaged input'),
                 ('synthetic_paired_vs_FILL-DIFF', 'vs FILL-DIFF')):
    for k, v in d[blk].items():
        print(f'  PAIRED {k} ({lbl}), n={v["n_paired"]}')
        for m in ('psnr', 'target_corr', 'detail_corr', 'grad_ncc', 'fill_ratio'):
            p = v[m]
            if p.get('n'):
                print(f'    {m:12s} {p["mean_a"]:8.4f} -> {p["mean_b"]:8.4f}  '
                      f'delta {p["mean_delta"]:+8.4f} CI [{p["ci95"][0]:+.4f},{p["ci95"][1]:+.4f}] '
                      f'better/worse {p["n_better"]}/{p["n_worse"]} (n={p["n"]})')

print('\n### WINDOWS (validation, oracle-assisted)')
for variant in ('oracle', 'expanded', 'expanded_ring', 'background'):
    for grp in ('recovery', 'control'):
        print(f'\n-- {grp} / {variant}')
        for m in ('E0', 'REF', 'FGB', 'FILL-RING', 'FILL-DIFF', 'FILL-TELEA', 'COMP', 'COMP-SHUF'):
            k = f'{grp}|{variant}|{m}'
            if k not in W:
                continue
            print(f'   {m:10s} n {W[k]["n_rows"]:4d} psnr {g(W, k, "psnr")} '
                  f'fill {g(W, k, "fill_ratio")} >0.02 {g(W, k, "frac_above_0p02")} '
                  f'corr {g(W, k, "target_corr")} detail {g(W, k, "detail_corr")} '
                  f'grad {g(W, k, "grad_ncc")} over {g(W, k, "overshoot_frac")} '
                  f'dpix {g(W, k, "mean_signed_change_vs_base")} '
                  f'outmax {g(W, k, "max_abs_change_outside_mask")} '
                  f'falseadd {g(W, k, "false_add_frac")} '
                  f'corr_def {W[k].get("target_corr__n_defined")}/{W[k].get("target_corr__n_total")}')

print('\n### PAIRED, key comparisons')
for blk in ('windows_paired_vs_E0', 'windows_paired_vs_FILL-DIFF',
            'windows_paired_vs_FILL-RING', 'windows_paired_COMP_vs_COMP-SHUF'):
    for k, v in d[blk].items():
        if '|oracle|' not in k and '|background|' not in k and '|expanded_ring|' not in k:
            continue
        if not any(s in k for s in ('COMP_vs', 'COMP-SHUF_vs')):
            continue
        print(f'  {blk} :: {k}  n={v["n_paired"]}')
        for m in ('psnr', 'target_corr', 'detail_corr', 'grad_ncc', 'fill_ratio',
                  'frac_above_0p02', 'overshoot_frac'):
            p = v.get(m, {})
            if p.get('n'):
                print(f'    {m:15s} {p["mean_a"]:8.4f} -> {p["mean_b"]:8.4f} delta {p["mean_delta"]:+8.4f} '
                      f'CI [{p["ci95"][0]:+.4f},{p["ci95"][1]:+.4f}] b/w {p["n_better"]}/{p["n_worse"]} n={p["n"]}')
