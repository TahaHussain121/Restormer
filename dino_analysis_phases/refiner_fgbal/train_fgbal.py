"""Train the foreground-balanced refiner: refiner_e0's run with the loss, the
selection rule, and the stopping count changed as documented in fgbal_common.

Outputs (experiments/<EXP>/): models/refiner_u<update>.pth at every
validation (u00000 is the identity = E0), refiner_selected.pth, val_log.csv,
train_log.csv, selection.json (both selection rules), train_summary.json.
"""

import argparse
import csv
import os
import shutil
import time

import numpy as np
import torch

import fgbal_common as fc
import refiner_common as rc
from refiner_arch import ResidualRefinerUNet, count_params, EXPECTED_PARAMS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    dev = args.device

    cfg, _, diff = fc.load_configs()
    if cfg['name'] != fc.EXP:
        raise SystemExit('config name mismatch')
    marker = os.path.join(fc.EXP_DIR, 'TRAINING_STARTED')
    if os.path.exists(marker):
        raise SystemExit(f'{fc.EXP} already started; it runs once. Refusing.')
    os.makedirs(fc.MODEL_DIR, exist_ok=True)
    shutil.copy(fc.CONFIG, os.path.join(fc.EXP_DIR, 'refiner_fgbal.yml'))

    rc.strict_fp32()
    torch.backends.cudnn.benchmark = False
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])

    ids_tr, ids_va = rc.split_ids('train'), rc.split_ids('val')
    ytr, _, prov_tr = rc.load_cache('train', ids_tr)
    yva, _, prov_va = rc.load_cache('val', ids_va)
    xtr_u, gtr_u = rc.load_split_uint16('train', ids_tr)
    xva_u, gva_u = rc.load_split_uint16('val', ids_va)
    ntr, B = len(ids_tr), cfg['batch_size']
    per_epoch = ntr // B if cfg['drop_last'] else -(-ntr // B)

    xva = torch.from_numpy(rc.to_unit(xva_u))[:, None].to(dev)
    y0va = torch.from_numpy(yva)[:, None].to(dev)
    gva_lv = torch.from_numpy(gva_u.astype(np.float32))[:, None].to(dev)
    fg_va = torch.from_numpy(gva_u.astype(np.float64) / 65535. > rc.FG_THRESHOLD)[:, None].to(dev)

    torch.manual_seed(cfg['seed'])                       # same init as refiner_e0
    net = ResidualRefinerUNet()
    old0 = torch.load(os.path.join(fc.OLD_EXP_DIR, 'models', 'refiner_step0.pth'),
                      map_location='cpu', weights_only=False)['params']
    init_diff = max(float((net.state_dict()[k] - old0[k]).abs().max()) for k in old0)
    init_exact = sum(torch.equal(net.state_dict()[k], old0[k]) for k in old0)
    print(f'init vs refiner_e0 step 0: {init_exact}/{len(old0)} tensors bit-identical, '
          f'max abs diff {init_diff:.2e}', flush=True)
    if init_diff > 1e-5:
        raise SystemExit('initialisation does not reproduce refiner_e0; refusing to train')
    net = net.to(dev)
    assert count_params(net) == EXPECTED_PARAMS
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], betas=tuple(cfg['betas']),
                            weight_decay=cfg['weight_decay'])
    gen = torch.Generator().manual_seed(cfg['seed'])

    def validate():
        net.eval()
        acc = {k: [] for k in ('pf', 'pm', 'bmae', 'brmse', 'fmae', 'ad', 'sfg', 'rl')}
        with torch.no_grad():
            for s in range(0, len(ids_va), B):
                x, y0, gl, fg = xva[s:s + B], y0va[s:s + B], gva_lv[s:s + B], fg_va[s:s + B]
                y, d = net(x, y0)
                q = rc.quantize_torch(y)
                acc['pf'].append(rc.psnr_levels_torch(gl, q))
                acc['pm'].append(rc.psnr_levels_torch(gl, q, fg))
                err = (q.double() - gl.double()) / 65535.
                bg, fgd = (~fg).double(), fg.double()
                nb, nf = bg.flatten(1).sum(1), fgd.flatten(1).sum(1)
                acc['bmae'].append((err.abs() * bg).flatten(1).sum(1) / nb)
                acc['brmse'].append(torch.sqrt((err ** 2 * bg).flatten(1).sum(1) / nb))
                acc['fmae'].append((err.abs() * fgd).flatten(1).sum(1) / nf)
                acc['ad'].append(d.abs().flatten(1).mean(1))
                acc['sfg'].append((d.double() * fgd).flatten(1).sum(1) / nf)
                acc['rl'].append(fc.region_l1(y, gl / 65535.)[0].reshape(1))
        net.train()
        mean = lambda v: torch.cat(v).double().mean().item()
        return {'psnr_full': mean(acc['pf']), 'psnr_mask': mean(acc['pm']),
                'bg_mae': mean(acc['bmae']), 'bg_rmse': mean(acc['brmse']),
                'fg_mae': mean(acc['fmae']), 'mean_abs_delta': mean(acc['ad']),
                'mean_signed_delta_fg': mean(acc['sfg']), 'region_l1_batchmean': mean(acc['rl'])}

    def save(t, val):
        p = os.path.join(fc.MODEL_DIR, f'refiner_u{t:05d}.pth')
        torch.save({'params': net.state_dict(), 'update': t, 'val': val, 'config': cfg}, p)
        return p

    tlog = open(os.path.join(fc.EXP_DIR, 'train_log.csv'), 'w', newline='')
    tw = csv.writer(tlog)
    tw.writerow(['update', 'lr', 'loss_mean_100', 'fg_term_mean_100', 'bg_term_mean_100',
                 'grad_norm_preclip_mean_100', 'frac_clipped_100'])
    fields = ['update', 'psnr_full', 'psnr_mask', 'bg_mae', 'bg_rmse', 'fg_mae', 'mean_abs_delta',
              'mean_signed_delta_fg', 'region_l1_batchmean', 'eligible', 'new_best',
              'best_update', 'nonimproving_streak', 'seconds']
    vlog = open(os.path.join(fc.EXP_DIR, 'val_log.csv'), 'w', newline='')
    vw = csv.DictWriter(vlog, fieldnames=fields)
    vw.writeheader()

    t0 = time.time()
    v0 = validate()
    save(0, v0)
    with open(marker, 'w') as f:
        f.write(f'{rc.now()} job {os.environ.get("SLURM_JOB_ID")}\n')
    ref = prov_va['reference_check']['mean_psnr_full_reference']
    print(f'step 0 (identity = E0): {v0}', flush=True)
    if abs(v0['psnr_full'] - ref) > 1e-6:
        raise SystemExit('identity validation does not reproduce the E0 reference')
    e0 = dict(v0)
    rows = [{'update': 0, **v0, 'eligible': True, 'new_best': '', 'best_update': 0,
             'nonimproving_streak': 0, 'seconds': 0.0}]
    vw.writerow(rows[-1])
    vlog.flush()

    best, best_t, streak, t, stop = v0['psnr_mask'], 0, 0, 0, None
    L, LF, LB, GN, CL = [], [], [], [], []
    torch.cuda.reset_peak_memory_stats()
    while stop is None:
        perm = torch.randperm(ntr, generator=gen).numpy()
        for b in range(per_epoch):
            idx = perm[b * B:(b + 1) * B]
            lr = rc.cosine_lr(t, cfg['lr'], cfg['lr_min'], cfg['max_updates'])
            for gr in opt.param_groups:
                gr['lr'] = lr
            x = torch.from_numpy(rc.to_unit(xtr_u[idx]))[:, None].to(dev, non_blocking=True)
            g = torch.from_numpy(rc.to_unit(gtr_u[idx]))[:, None].to(dev, non_blocking=True)
            y0 = torch.from_numpy(ytr[idx])[:, None].to(dev, non_blocking=True)
            y, _ = net(x, y0)
            loss, parts = fc.region_l1(y, g)                 # unbounded Y0 + delta
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg['grad_clip_norm'])
            if not (torch.isfinite(loss) and torch.isfinite(gn)):
                with open(os.path.join(fc.EXP_DIR, 'NONFINITE'), 'w') as f:
                    f.write(f'update {t + 1}: loss {float(loss)} grad {float(gn)}\n')
                raise SystemExit(f'non-finite loss/gradient at update {t + 1}')
            opt.step()
            t += 1
            L.append(float(loss)); GN.append(float(gn)); CL.append(float(gn) > cfg['grad_clip_norm'])
            LF.append(parts['fg'] if parts['fg'] is not None else np.nan)
            LB.append(parts['bg'] if parts['bg'] is not None else np.nan)
            if t % 100 == 0:
                tw.writerow([t, lr, np.mean(L), np.nanmean(LF), np.nanmean(LB), np.mean(GN), np.mean(CL)])
                tlog.flush()
                print(f'update {t:5d}  lr {lr:.3e}  loss {np.mean(L):.6f}  fg {np.nanmean(LF):.6f}  '
                      f'bg {np.nanmean(LB):.6f}  grad {np.mean(GN):.3e}', flush=True)
                L, LF, LB, GN, CL = [], [], [], [], []
            if t % cfg['val_every'] == 0:
                v = validate()
                save(t, v)
                eligible = v['bg_mae'] <= e0['bg_mae'] and v['bg_rmse'] <= e0['bg_rmse']
                improved = eligible and v['psnr_mask'] > best
                if improved:
                    best, best_t, streak = v['psnr_mask'], t, 0
                else:
                    streak += 1
                rows.append({'update': t, **v, 'eligible': eligible, 'new_best': improved,
                             'best_update': best_t, 'nonimproving_streak': streak,
                             'seconds': round(time.time() - t0, 1)})
                vw.writerow(rows[-1])
                vlog.flush()
                print(f'  VAL {t:5d} full {v["psnr_full"]:.4f} fg {v["psnr_mask"]:.4f} '
                      f'bgMAE {v["bg_mae"]:.5f} bgRMSE {v["bg_rmse"]:.5f} sFG {v["mean_signed_delta_fg"]:+.5f} '
                      f'eligible {eligible} best {best:.4f}@{best_t} streak {streak}', flush=True)
                if t >= cfg['min_updates'] and streak >= cfg['patience_checks']:
                    stop = (f'early stop: {streak} consecutive checks without a new eligible '
                            f'foreground-PSNR best, at update {t}')
                    break
            if t >= cfg['max_updates']:
                stop = f'hard maximum {cfg["max_updates"]} updates'
                break
    wall = time.time() - t0

    sel_src = os.path.join(fc.MODEL_DIR, f'refiner_u{best_t:05d}.pth')
    shutil.copy(sel_src, os.path.join(fc.MODEL_DIR, 'refiner_selected.pth'))
    old_rule = max(rows, key=lambda r: r['psnr_full'])
    selection = {
        'experiment': fc.EXP,
        'new_rule': {'rule': cfg['selection_metric'] + '; ' + cfg['eligibility'],
                     'selected_update': best_t, 'e0_fallback': best_t == 0,
                     'val': next(r for r in rows if r['update'] == best_t),
                     'n_eligible_checks': sum(bool(r['eligible']) for r in rows[1:]),
                     'n_checks': len(rows) - 1},
        'original_rule_on_this_run': {'rule': 'max validation full256 PSNR (refiner_e0)',
                                      'selected_update': old_rule['update'], 'val': old_rule},
        'e0_reference_val': e0,
        'checks': rows,
    }
    rc.write_json(fc.SELECTION_JSON, selection)
    summary = {'experiment': fc.EXP, 'updates_done': t, 'stop_reason': stop,
               'epochs_equivalent': t / per_epoch, 'selected_update': best_t,
               'e0_fallback': best_t == 0, 'trainable_params': count_params(net),
               'wall_seconds_incl_validation': wall,
               'peak_gpu_mem_gib': torch.cuda.max_memory_allocated() / 2 ** 30,
               'config_diff_vs_refiner_e0': diff, 'device': rc.device_record(dev),
               'cache_sha256': {'train': prov_tr['output']['sha256'],
                                'val': prov_va['output']['sha256']},
               'created': rc.now()}
    rc.write_json(os.path.join(fc.EXP_DIR, 'train_summary.json'), summary)
    with open(os.path.join(fc.EXP_DIR, 'TRAINING_DONE'), 'w') as f:
        f.write(f'{rc.now()}\n')
    print(f'\nDONE: {stop}. new rule -> update {best_t}'
          f'{" (E0 FALLBACK)" if best_t == 0 else ""}; original rule -> update '
          f'{old_rule["update"]}; {t} updates, {wall / 60:.1f} min')


if __name__ == '__main__':
    main()
