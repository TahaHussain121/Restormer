"""Train ONLY the residual refiner on cached, frozen E0 full-frame outputs.

E0 is not even instantiated here: its outputs come from the validated cache,
so no E0 parameter can reach an optimizer. Every setting comes from
configs/refiner.yml; nothing is tuned.

Outputs (experiments/<EXP>/):  models/refiner_step0.pth (identity),
refiner_best.pth (validation-selected), refiner_final.pth; train_log.csv,
val_log.csv, train_summary.json.
"""

import argparse
import csv
import os
import shutil
import time

import numpy as np
import torch
import yaml

import refiner_common as rc
from refiner_arch import ResidualRefinerUNet, count_params, EXPECTED_PARAMS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    dev = args.device

    with open(rc.CONFIG) as f:
        cfg = yaml.safe_load(f)
    if cfg['name'] != rc.EXP:
        raise SystemExit('config name mismatch')
    marker = os.path.join(rc.EXP_DIR, 'TRAINING_STARTED')
    if os.path.exists(marker) or os.path.exists(os.path.join(rc.MODEL_DIR, 'refiner_final.pth')):
        raise SystemExit(f'{rc.EXP} has already been trained or started; '
                         f'this experiment runs once. Refusing.')
    os.makedirs(rc.MODEL_DIR, exist_ok=True)
    shutil.copy(rc.CONFIG, os.path.join(rc.EXP_DIR, 'refiner.yml'))

    rc.strict_fp32()
    torch.backends.cudnn.benchmark = False
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])

    # --- data -----------------------------------------------------------------
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

    # --- model ----------------------------------------------------------------
    torch.manual_seed(cfg['seed'])
    net = ResidualRefinerUNet().to(dev)
    assert count_params(net) == EXPECTED_PARAMS
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], betas=tuple(cfg['betas']),
                            weight_decay=cfg['weight_decay'])
    gen = torch.Generator().manual_seed(cfg['seed'])

    def validate():
        net.eval()
        pf, pm, bgm, ad, l1 = [], [], [], [], []
        with torch.no_grad():
            for s in range(0, len(ids_va), B):
                x, y0, gl, fg = xva[s:s + B], y0va[s:s + B], gva_lv[s:s + B], fg_va[s:s + B]
                y, d = net(x, y0)
                q = rc.quantize_torch(y)
                pf.append(rc.psnr_levels_torch(gl, q))
                pm.append(rc.psnr_levels_torch(gl, q, fg))
                err = ((q.double() - gl.double()) / 65535.).abs()
                bg = (~fg).double()
                bgm.append((err * bg).flatten(1).sum(1) / bg.flatten(1).sum(1))
                ad.append(d.abs().flatten(1).mean(1))
                l1.append((y - gl / 65535.).abs().flatten(1).mean(1))
        net.train()
        cat = lambda v: torch.cat(v).double().mean().item()
        return {'psnr_full': cat(pf), 'psnr_mask': cat(pm), 'bg_mae': cat(bgm),
                'mean_abs_delta': cat(ad), 'l1_unbounded': cat(l1)}

    def save(path, t, val):
        torch.save({'params': net.state_dict(), 'update': t, 'val': val,
                    'config': cfg}, path)

    tlog = open(os.path.join(rc.EXP_DIR, 'train_log.csv'), 'w', newline='')
    vlog = open(os.path.join(rc.EXP_DIR, 'val_log.csv'), 'w', newline='')
    tw = csv.writer(tlog)
    tw.writerow(['update', 'lr', 'loss_mean_100', 'grad_norm_preclip_mean_100',
                 'frac_clipped_100', 'mean_abs_delta_last'])
    vw = csv.DictWriter(vlog, fieldnames=['update', 'psnr_full', 'psnr_mask', 'bg_mae',
                                          'mean_abs_delta', 'l1_unbounded', 'improved',
                                          'best_update', 'nonimproving_streak', 'seconds'])
    vw.writeheader()

    t_start = time.time()
    v0 = validate()
    save(os.path.join(rc.MODEL_DIR, 'refiner_step0.pth'), 0, v0)
    shutil.copy(os.path.join(rc.MODEL_DIR, 'refiner_step0.pth'),
                os.path.join(rc.MODEL_DIR, 'refiner_best.pth'))
    with open(marker, 'w') as f:
        f.write(f'{rc.now()} job {os.environ.get("SLURM_JOB_ID")}\n')
    ref_mean = prov_va['reference_check']['mean_psnr_full_reference']
    print(f'step 0 (identity): val psnr_full {v0["psnr_full"]:.6f} '
          f'(E0 reference {ref_mean:.6f})', flush=True)
    if abs(v0['psnr_full'] - ref_mean) > 1e-6:
        raise SystemExit('identity validation does not reproduce the E0 reference')
    vw.writerow({'update': 0, **v0, 'improved': '', 'best_update': 0,
                 'nonimproving_streak': 0, 'seconds': 0.0})
    vlog.flush()

    best, best_t, streak = v0['psnr_full'], 0, 0
    t, stop_reason = 0, None
    losses, gns, clipped = [], [], []
    torch.cuda.reset_peak_memory_stats()
    while stop_reason is None:
        perm = torch.randperm(ntr, generator=gen).numpy()
        for b in range(per_epoch):
            idx = perm[b * B:(b + 1) * B]
            lr = rc.cosine_lr(t, cfg['lr'], cfg['lr_min'], cfg['max_updates'])
            for gr in opt.param_groups:
                gr['lr'] = lr
            x = torch.from_numpy(rc.to_unit(xtr_u[idx]))[:, None].to(dev, non_blocking=True)
            g = torch.from_numpy(rc.to_unit(gtr_u[idx]))[:, None].to(dev, non_blocking=True)
            y0 = torch.from_numpy(ytr[idx])[:, None].to(dev, non_blocking=True)
            y, d = net(x, y0)
            loss = (y - g).abs().mean()                    # unbounded Y0 + delta
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg['grad_clip_norm'])
            if not (torch.isfinite(loss) and torch.isfinite(gn)):
                with open(os.path.join(rc.EXP_DIR, 'NONFINITE'), 'w') as f:
                    f.write(f'update {t + 1}: loss {float(loss)} grad {float(gn)}\n')
                raise SystemExit(f'non-finite loss/gradient at update {t + 1}')
            opt.step()
            t += 1
            losses.append(float(loss))
            gns.append(float(gn))
            clipped.append(float(gn) > cfg['grad_clip_norm'])
            if t % 100 == 0:
                tw.writerow([t, lr, np.mean(losses), np.mean(gns), np.mean(clipped),
                             float(d.detach().abs().mean())])
                tlog.flush()
                print(f'update {t:5d}  lr {lr:.3e}  loss {np.mean(losses):.6f}  '
                      f'grad {np.mean(gns):.3e}  clipped {np.mean(clipped):.2f}', flush=True)
                losses, gns, clipped = [], [], []
            if t % cfg['val_every'] == 0:
                v = validate()
                improved = v['psnr_full'] > best
                if improved:
                    best, best_t, streak = v['psnr_full'], t, 0
                    save(os.path.join(rc.MODEL_DIR, 'refiner_best.pth'), t, v)
                else:
                    streak += 1
                vw.writerow({'update': t, **v, 'improved': improved, 'best_update': best_t,
                             'nonimproving_streak': streak,
                             'seconds': round(time.time() - t_start, 1)})
                vlog.flush()
                print(f'  VAL {t:5d}  psnr_full {v["psnr_full"]:.4f}  psnr_mask '
                      f'{v["psnr_mask"]:.4f}  bg_mae {v["bg_mae"]:.5f}  '
                      f'|delta| {v["mean_abs_delta"]:.5f}  best {best:.4f}@{best_t}  '
                      f'streak {streak}', flush=True)
                if t >= cfg['min_updates'] and streak >= cfg['patience_checks']:
                    stop_reason = (f'early stop: {streak} consecutive validation checks '
                                   f'without improvement, at update {t} (>= {cfg["min_updates"]})')
                    break
            if t >= cfg['max_updates']:
                stop_reason = f'hard maximum {cfg["max_updates"]} updates'
                break

    vfinal = validate()
    save(os.path.join(rc.MODEL_DIR, 'refiner_final.pth'), t, vfinal)
    wall = time.time() - t_start
    summary = {
        'experiment': rc.EXP, 'updates_done': t, 'stop_reason': stop_reason,
        'epochs_equivalent': t / per_epoch, 'batches_per_epoch': per_epoch,
        'selected_update': best_t, 'selected_val': best,
        'identity_selected': best_t == 0,
        'step0_val': v0, 'final_val': vfinal,
        'selected_checkpoint': os.path.join(rc.MODEL_DIR, 'refiner_best.pth'),
        'trainable_params': count_params(net),
        'wall_seconds_incl_validation': wall,
        'peak_gpu_mem_gib': torch.cuda.max_memory_allocated() / 2 ** 30,
        'device': rc.device_record(dev), 'config': cfg,
        'cache_sha256': {'train': prov_tr['output']['sha256'],
                         'val': prov_va['output']['sha256']},
        'created': rc.now(),
    }
    rc.write_json(os.path.join(rc.EXP_DIR, 'train_summary.json'), summary)
    with open(os.path.join(rc.EXP_DIR, 'TRAINING_DONE'), 'w') as f:
        f.write(f'{rc.now()}\n')
    print(f'\nDONE: {stop_reason}. selected update {best_t} (val {best:.4f}); '
          f'{t} updates, {wall / 60:.1f} min')


if __name__ == '__main__':
    main()
