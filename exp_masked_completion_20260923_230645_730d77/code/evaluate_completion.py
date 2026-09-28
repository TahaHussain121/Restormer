"""Evaluate the validation-selected completion checkpoint.

PART A -- the fixed SYNTHETIC completion task (can the model do the job it was
trained for?), against the damaged input and the non-learned fills.

PART B -- the ORACLE-ASSISTED actual-failure diagnostic on the EXISTING
validation recovery/control windows, whose membership and definitions are
reused unchanged from refiner_e0/qualitative_cases.json. The oracle mask rule
(target > 0.05 and E0 < 0.02) was fixed in docs/PROTOCOL.md before any
prediction of this experiment was inspected.

The test split is not read. Everything written stays inside the experiment root.
"""
import argparse, csv, json, os, sys, time

import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mc_common as mc                                       # noqa: E402
import fills                                                  # noqa: E402
import metrics_lib as ml                                      # noqa: E402
from completion_arch import MaskedCompletionUNet, count_params  # noqa: E402

K3 = np.ones((3, 3), np.uint8)
BG_SEED, BG_MARGIN = 7, 6
EXPAND_ITERS = 3


def unit(u16):
    return u16.astype(np.float64) / 65535.


def q_unit(x):
    """float -> the study's uint16 convention -> back to [0,1] float64."""
    return mc.quantize_np(np.asarray(x, np.float32)).astype(np.float64) / 65535.


def background_mask(gt, n_px, seed):
    """A compact mask of n_px pixels on deep background (target <= 0.01, at
    least BG_MARGIN px from any foreground pixel), placed deterministically."""
    fg = (gt > mc.FG_THRESHOLD).astype(np.uint8)
    far = cv2.dilate(fg, np.ones((2 * BG_MARGIN + 1,) * 2, np.uint8)) == 0
    idx = np.flatnonzero(far)
    m = np.zeros(gt.shape, bool)
    if idx.size == 0 or n_px == 0:
        return m, 0
    rng = np.random.default_rng(seed)
    c = int(rng.choice(idx))
    cy, cx = divmod(c, gt.shape[1])
    ys, xs = np.divmod(idx, gt.shape[1])
    d = (ys - cy) ** 2 + (xs - cx) ** 2
    take = idx[np.argsort(d, kind='stable')[:min(n_px, idx.size)]]
    m.flat[take] = True
    return m, int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--checkpoint', default=None,
                    help='POST-HOC SENSITIVITY ONLY: evaluate a different saved '
                         'checkpoint. The primary result always uses the checkpoint '
                         'recorded in results/checkpoint_selection.json.')
    ap.add_argument('--tag', default='',
                    help='suffix for the output files of a sensitivity run')
    a = ap.parse_args()
    TAG = ('_' + a.tag) if a.tag else ''
    dev = a.device if torch.cuda.is_available() else 'cpu'
    mc.strict_fp32()
    t0 = time.time()

    with open(os.path.join(mc.RESULTS, 'checkpoint_selection.json')) as f:
        sel = json.load(f)
    ckpt_path = a.checkpoint or sel['selected_checkpoint']
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if a.checkpoint is None and ck['update'] != sel['selected_update']:
        raise SystemExit('checkpoint does not match the recorded selection')
    if a.checkpoint is not None and not a.tag:
        raise SystemExit('a non-selected checkpoint needs --tag (it is a sensitivity run)')
    net = MaskedCompletionUNet().to(dev)
    net.load_state_dict(ck['params'], strict=True)
    net.eval()
    print(f'checkpoint: {ckpt_path} (update {ck["update"]}, {count_params(net)} params)'
          + ('  [POST-HOC SENSITIVITY RUN]' if a.checkpoint else '  [pre-registered selection]'),
          flush=True)

    ids = mc.split_ids('val')
    N = len(ids)
    cache, cids, prov = mc.load_cache('val', ids, mmap=True)
    x_u, g_u = mc.load_split_uint16('val', ids)

    def run_net(x_np, yin_np, m_np):
        """x, yin, m: [B,H,W] float32 -> quantised prediction in [0,1]."""
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(x_np, np.float32))[:, None].to(dev)
            y = torch.from_numpy(np.ascontiguousarray(yin_np, np.float32))[:, None].to(dev)
            m = torch.from_numpy(np.ascontiguousarray(m_np, np.float32))[:, None].to(dev)
            out, _ = net(x, y, m)
            return q_unit(out[:, 0].cpu().numpy())

    # ================= PART A: the fixed synthetic completion task ============
    va = np.load(os.path.join(mc.CACHE, 'val_synth_masks.npz'))
    if list(va['ids']) != ids:
        raise SystemExit('validation mask ids differ')
    dmg, msk = va['damaged'].astype(np.float32), va['mask'].astype(bool)
    rowsA, B = [], 16
    for s in range(0, N, B):
        sl = slice(s, min(s + B, N))
        comp = run_net(mc.to_unit(x_u[sl]), dmg[sl], msk[sl].astype(np.float32))
        for k in range(comp.shape[0]):
            i = s + k
            g = unit(g_u[i]); m = msk[i]
            if not m.any():
                rowsA.append({'id': ids[i], 'method': 'EMPTY_MASK', 'n_mask_px': 0})
                continue
            base = dmg[i].astype(np.float64)
            preds = {'INPUT-DAMAGED': q_unit(base), 'COMP': comp[k],
                     'FILL-RING': q_unit(fills.ring_fill(base, m)),
                     'FILL-DIFF': q_unit(fills.diffusion_fill(base, m)),
                     'FILL-TELEA': q_unit(fills.telea_fill(base, m))}
            for name, p in preds.items():
                r = ml.window_metrics(p, g, m, base=q_unit(base))
                rowsA.append({'id': ids[i], 'method': name, **r})
        print(f'  synthetic {min(s + B, N)}/{N}', flush=True) if s % 80 == 0 else None

    # ================= PART B: the oracle-assisted actual-failure diagnostic ==
    with open(mc.CASES_JSON) as f:
        cases = json.load(f)
    pop = cases['population']
    fixed_case_ids = {c['id']: c['role'] for c in cases['cases']}
    W = cases['params']['window']
    e0q_dir = mc.E0_REF_PRED.format(split='val')
    idx_of = {i: k for k, i in enumerate(cids)}

    rowsB, skipped = [], []
    save_dir = os.path.join(mc.RESULTS, 'qualitative_cases')
    for group in ('recovery', 'control'):
        entries = pop[group]
        for n_done, e in enumerate(entries):
            i = e['id']
            k = idx_of[i]
            g = unit(g_u[k])
            xk = mc.to_unit(x_u[k]).astype(np.float64)
            x_shuf = mc.to_unit(x_u[(k + 1) % N]).astype(np.float64)
            e0raw = np.asarray(cache[k], np.float32)
            e0q = q_unit(e0raw)
            y0, x0 = e['cy'] - W // 2, e['cx'] - W // 2
            win = np.zeros(g.shape, bool)
            win[y0:y0 + W, x0:x0 + W] = True

            oracle = (g > mc.MISS_GT) & (e0q < mc.MISS_E0) & win
            if not oracle.any():
                skipped.append({'group': group, 'id': i, 'reason': 'empty oracle mask'})
                continue
            expanded = (cv2.dilate(oracle.astype(np.uint8), K3,
                                   iterations=EXPAND_ITERS) > 0) & win
            bg_mask, n_bg = background_mask(g, int(oracle.sum()),
                                            BG_SEED + (int(i) if i.isdigit() else 0))
            variants = {'oracle': oracle, 'expanded': expanded}
            if n_bg > 0:
                variants['background'] = bg_mask

            ref = unit(mc.load_uint16(os.path.join(mc.REFINER_E0_VAL_PRED, f'{i}.png')))
            fgb = unit(mc.load_uint16(os.path.join(mc.FGBAL_VAL_PRED, f'{i}.png')))
            pre = {'hp_gt': g - ml.box5(g), 'grad_gt': ml.grad_mag(g)}

            for vname, m in variants.items():
                mf = m.astype(np.float32)
                comp = run_net(xk[None].astype(np.float32), e0raw[None], mf[None])[0]
                shuf = run_net(x_shuf[None].astype(np.float32), e0raw[None], mf[None])[0]
                preds = {
                    'E0': e0q, 'REF': ref, 'FGB': fgb,
                    'FILL-RING': q_unit(fills.ring_fill(e0q, m)),
                    'FILL-DIFF': q_unit(fills.diffusion_fill(e0q, m)),
                    'FILL-TELEA': q_unit(fills.telea_fill(e0q, m)),
                    'COMP': comp, 'COMP-SHUF': shuf}
                for name, p in preds.items():
                    mask_restricted = name not in ('E0', 'REF', 'FGB')
                    r = ml.window_metrics(p, g, m, base=e0q if mask_restricted else None,
                                          pre=pre)
                    rowsB.append({'group': group, 'id': i, 'variant': vname,
                                  'method': name, 'fixed_case': fixed_case_ids.get(i, ''),
                                  'gt_mean_window': e.get('gt_mean'),
                                  'noisy_support_corr': e.get('noisy_support_corr'),
                                  'e0_retention': e.get('e0_retention'), **r})
                if vname == 'expanded':
                    ring = expanded & (~oracle)
                    if ring.any():
                        for name, p in preds.items():
                            rr = ml.window_metrics(p, g, ring,
                                                   base=e0q if name not in ('E0', 'REF', 'FGB')
                                                   else None, pre=pre)
                            rr['false_add_frac'] = float(
                                ((p[ring] > 0.05) & (g[ring] <= 0.05)).mean())
                            rowsB.append({'group': group, 'id': i, 'variant': 'expanded_ring',
                                          'method': name, 'fixed_case': fixed_case_ids.get(i, ''),
                                          **rr})
            if n_done % 50 == 0:
                print(f'  {group} {n_done}/{len(entries)}', flush=True)

    # --- the EIGHT fixed cases, each in ITS OWN recorded window -----------------
    # 154 validation images carry both a recovery and a control window, so the
    # case images are written once here, from the window recorded in the case
    # entry, rather than inside the population loops.
    case_records = []
    if not TAG:
        for c in cases['cases']:
            i = c['id']; k = idx_of[i]
            g = unit(g_u[k]); xk = mc.to_unit(x_u[k]).astype(np.float64)
            e0raw = np.asarray(cache[k], np.float32); e0q = q_unit(e0raw)
            wy, wx, ws = c['window_yx_size']
            win = np.zeros(g.shape, bool); win[wy:wy + ws, wx:wx + ws] = True
            m = (g > mc.MISS_GT) & (e0q < mc.MISS_E0) & win
            comp = run_net(xk[None].astype(np.float32), e0raw[None],
                           m.astype(np.float32)[None])[0]
            ring = q_unit(fills.ring_fill(e0q, m))
            diff = q_unit(fills.diffusion_fill(e0q, m))
            for nm, arr in (('noisy', xk), ('target', g), ('e0', e0q),
                            ('mask', m.astype(np.float64)), ('comp', comp),
                            ('fill_ring', ring), ('fill_diff', diff)):
                mc.imwrite_guarded(os.path.join(save_dir, f'{i}_{nm}.png'),
                                   mc.quantize_np(arr.astype(np.float32)))
            case_records.append({'id': i, 'role': c['role'],
                                 'window_yx_size': c['window_yx_size'],
                                 'n_mask_px': int(m.sum()),
                                 'comp_vs_e0': ml.window_metrics(comp, g, m, base=e0q)
                                 if m.any() else None,
                                 'e0': ml.window_metrics(e0q, g, m) if m.any() else None,
                                 'fill_ring': ml.window_metrics(ring, g, m, base=e0q)
                                 if m.any() else None})

    # ---------------------------------------------------------------- write out
    def dump(rows, path, fields):
        p = mc.assert_inside_root(path)
        mc.ensure_dir(os.path.dirname(p))
        with open(p, 'x', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            for r in rows:
                w.writerow(r)

    fA = ['id', 'method', 'n_mask_px', 'mae', 'psnr', 'fill_ratio', 'frac_above_0p02',
          'frac_above_0p05', 'overshoot_frac', 'mean_positive_excess',
          'mean_signed_change_vs_base', 'target_corr', 'detail_corr', 'grad_ncc',
          'iou_0p05', 'hit_rate_0p05', 'max_abs_change_outside_mask']
    fB = ['group', 'id', 'variant', 'method', 'fixed_case', 'gt_mean_window',
          'noisy_support_corr', 'e0_retention', 'false_add_frac'] + fA[2:]
    dump(rowsA, os.path.join(mc.RESULTS, 'metrics', f'synthetic_per_image{TAG}.csv'), fA)
    dump(rowsB, os.path.join(mc.RESULTS, 'metrics', f'windows_per_window{TAG}.csv'), fB)

    # aggregate -------------------------------------------------------------
    METRICS = [m for m in fA[2:] if m not in ('mean_signed_change_vs_base',)]

    def agg(rows, keyf):
        out = {}
        for r in rows:
            out.setdefault(keyf(r), []).append(r)
        summary = {}
        for key, rs in out.items():
            d = {'n_windows': len(rs)}
            for mname in METRICS + ['mean_signed_change_vs_base', 'false_add_frac']:
                v = [r[mname] for r in rs if r.get(mname) is not None
                     and np.isfinite(r.get(mname, np.nan))]
                if v:
                    d[mname] = float(np.mean(v))
                    d[f'{mname}__n_defined'] = len(v)
            summary['|'.join(key) if isinstance(key, tuple) else key] = d
        return summary

    def paired_block(rows, keyf, ref_method, test_methods):
        by = {}
        for r in rows:
            by.setdefault((keyf(r), r['method']), {})[r['id']] = r
        out = {}
        for (key, meth) in list(by):
            if meth not in test_methods:
                continue
            ref = by.get((key, ref_method), {})
            common = sorted(set(ref) & set(by[(key, meth)]))
            block = {'n_paired': len(common), 'reference': ref_method}
            for mname in METRICS:
                aa = [ref[i].get(mname) for i in common]
                bb = [by[(key, meth)][i].get(mname) for i in common]
                aa = [np.nan if v is None else v for v in aa]
                bb = [np.nan if v is None else v for v in bb]
                block[mname] = ml.paired(aa, bb, mname in ml.HIGHER)
            out[f'{"|".join(key) if isinstance(key, tuple) else key}|{meth}_vs_{ref_method}'] = block
        return out

    keyB = lambda r: (r['group'], r['variant'])
    summary = {
        'experiment': mc.EXP, 'root': mc.ROOT,
        'checkpoint': ckpt_path, 'checkpoint_update': ck['update'],
        'is_preregistered_selection': a.checkpoint is None,
        'selected_checkpoint': sel['selected_checkpoint'],
        'selected_update': sel['selected_update'],
        'oracle_assisted': True,
        'mask_rule': 'target > %.2f and E0 < %.2f, inside the fixed 48x48 window'
                     % (mc.MISS_GT, mc.MISS_E0),
        'expand_iters': EXPAND_ITERS, 'background_mask_seed': BG_SEED,
        'n_recovery_windows_available': len(pop['recovery']),
        'n_control_windows_available': len(pop['control']),
        'skipped_empty_mask': skipped,
        'synthetic': agg(rowsA, lambda r: r['method']),
        'synthetic_paired_vs_INPUT': paired_block(
            rowsA, lambda r: 'synthetic', 'INPUT-DAMAGED',
            ['COMP', 'FILL-RING', 'FILL-DIFF', 'FILL-TELEA']),
        'synthetic_paired_vs_FILL-DIFF': paired_block(
            rowsA, lambda r: 'synthetic', 'FILL-DIFF', ['COMP']),
        'windows': agg(rowsB, keyB),
        'windows_paired_vs_E0': paired_block(rowsB, keyB, 'E0',
                                             ['COMP', 'COMP-SHUF', 'FILL-RING',
                                              'FILL-DIFF', 'FILL-TELEA', 'REF', 'FGB']),
        'windows_paired_vs_FILL-RING': paired_block(rowsB, keyB, 'FILL-RING', ['COMP']),
        'windows_paired_vs_FILL-DIFF': paired_block(rowsB, keyB, 'FILL-DIFF', ['COMP']),
        'fixed_cases': case_records,
        'windows_paired_COMP_vs_COMP-SHUF': paired_block(rowsB, keyB, 'COMP', ['COMP-SHUF']),
        'device': mc.device_record(dev), 'seconds': round(time.time() - t0, 1),
        'created': mc.now(),
    }
    mc.write_json_exclusive(os.path.join(mc.RESULTS, f'evaluation_summary{TAG}.json'), summary)
    print(f'\nwrote evaluation_summary{TAG}.json in {summary["seconds"]}s; '
          f'{len(skipped)} windows skipped for an empty oracle mask')


if __name__ == '__main__':
    main()
