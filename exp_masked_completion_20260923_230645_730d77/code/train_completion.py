"""Train ONLY the masked-completion U-Net on synthetically damaged CLEAN
training targets. One fixed configuration, one seed, bounded budget.

The frozen E0 is never instantiated here, so no E0 parameter can reach the
optimizer; E0 appears only later, at the diagnostic, through its existing
read-only output cache.

Every write goes through mc_common's root guard. Checkpoints are saved under
unique step-specific names (ckpt_update_XXXXXX.pth) with exclusive creation;
nothing is ever overwritten and there is no best.pth / latest.pth.
"""
import argparse, csv, os, sys, time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mc_common as mc                                     # noqa: E402
import corruption                                           # noqa: E402
from completion_arch import MaskedCompletionUNet, count_params, EXPECTED_PARAMS  # noqa: E402


def region_l1(err_abs, region):
    """Per-image mean of |err| over a per-image region, averaged over the images
    whose region is non-empty. Returns (value, n_images_with_region)."""
    r = region.flatten(1)
    n = r.sum(1)
    have = n > 0
    if not bool(have.any()):
        return None, 0
    s = (err_abs.flatten(1) * r).sum(1)
    return (s[have] / n[have]).mean(), int(have.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max-updates', type=int, default=None, help='smoke use only')
    a = ap.parse_args()
    dev = a.device

    with open(mc.CONFIG) as f:
        cfg = yaml.safe_load(f)
    if cfg['name'] != mc.EXP:
        raise SystemExit('config name mismatch')
    if a.max_updates:
        cfg['max_updates'] = a.max_updates

    started = os.path.join(mc.ROOT, 'TRAINING_STARTED')
    if os.path.exists(started):
        raise SystemExit('this experiment root has already been trained; refusing')

    # ---- requirement 10: every planned destination resolves inside the root
    planned = [started, os.path.join(mc.ROOT, 'TRAINING_DONE'),
               os.path.join(mc.ROOT, 'results', 'train_log.csv'),
               os.path.join(mc.ROOT, 'results', 'val_log.csv'),
               os.path.join(mc.ROOT, 'results', 'train_summary.json'),
               os.path.join(mc.ROOT, 'results', 'checkpoint_selection.json'),
               os.path.join(mc.CKPT_DIR, 'ckpt_update_000000.pth')]
    checked = mc.verify_write_paths(planned)
    print('write-path check passed for %d destinations' % len(checked), flush=True)

    mc.strict_fp32()
    torch.backends.cudnn.benchmark = False
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])

    # ---- data (read-only) ---------------------------------------------------
    ids_tr, ids_va = mc.split_ids('train'), mc.split_ids('val')
    overlap = set(ids_tr) & set(ids_va)
    if overlap:
        raise SystemExit(f'train/val overlap: {sorted(overlap)[:5]}')
    xtr_u, gtr_u = mc.load_split_uint16('train', ids_tr)
    xva_u, gva_u = mc.load_split_uint16('val', ids_va)

    va = np.load(os.path.join(mc.CACHE, 'val_synth_masks.npz'))
    if list(va['ids']) != list(ids_va):
        raise SystemExit('validation mask file ids differ from the split')
    xva = torch.from_numpy(mc.to_unit(xva_u))[:, None].to(dev)
    gva = torch.from_numpy(mc.to_unit(gva_u))[:, None].to(dev)
    gva_lv = torch.from_numpy(gva_u.astype(np.float32))[:, None].to(dev)
    dva = torch.from_numpy(va['damaged'].astype(np.float32))[:, None].to(dev)
    mva = torch.from_numpy(va['mask'].astype(np.float32))[:, None].to(dev)

    ntr, B = len(ids_tr), cfg['batch_size']
    per_epoch = ntr // B

    # ---- model --------------------------------------------------------------
    torch.manual_seed(cfg['seed'])
    net = MaskedCompletionUNet().to(dev)
    if count_params(net) != EXPECTED_PARAMS:
        raise SystemExit(f'parameter count {count_params(net)} != {EXPECTED_PARAMS}')
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], betas=tuple(cfg['betas']),
                            weight_decay=cfg['weight_decay'])
    opt_ids = {id(p) for gr in opt.param_groups for p in gr['params']}
    if opt_ids != {id(p) for p in net.parameters()}:
        raise SystemExit('optimizer parameter set != completion network parameters')
    gen = torch.Generator().manual_seed(cfg['seed'])
    rng = np.random.default_rng(cfg['seed'])          # corruption stream

    def validate():
        """The FIXED synthetic completion task. Selection metric = mean
        per-image in-mask PSNR under the uint16 convention."""
        net.eval()
        pm, pfg, mae, out_max, n_eff = [], [], [], 0.0, 0
        with torch.no_grad():
            for s in range(0, len(ids_va), B):
                x, yin, m = xva[s:s + B], dva[s:s + B], mva[s:s + B]
                g, gl = gva[s:s + B], gva_lv[s:s + B]
                y, _ = net(x, yin, m)
                out_max = max(out_max, float(((y - yin).abs() * (1 - m)).max()))
                q = mc.quantize_torch(y)
                fgm = m * (g > mc.FG_THRESHOLD).float()
                has = m.flatten(1).sum(1) > 0
                n_eff += int(has.sum())
                p = mc.psnr_levels_torch(gl, q, m)
                pm.append(p[has])
                hf = fgm.flatten(1).sum(1) > 0
                pfg.append(mc.psnr_levels_torch(gl, q, fgm)[hf])
                e = ((q.double() - gl.double()) / 65535.).abs()
                md = m.double()
                mae.append(((e * md).flatten(1).sum(1) / md.flatten(1).sum(1).clamp(min=1))[has])
        net.train()
        cat = lambda v: torch.cat(v).double().mean().item()
        return {'psnr_in_mask': cat(pm), 'psnr_in_mask_fg': cat(pfg),
                'mae_in_mask': cat(mae), 'n_images_with_mask': n_eff,
                'max_abs_change_outside_mask': out_max}

    # ---- logs ---------------------------------------------------------------
    mc.ensure_dir(mc.RESULTS)
    tlog = open(mc.assert_inside_root(os.path.join(mc.RESULTS, 'train_log.csv')), 'w', newline='')
    vlog = open(mc.assert_inside_root(os.path.join(mc.RESULTS, 'val_log.csv')), 'w', newline='')
    tw = csv.writer(tlog)
    tw.writerow(['update', 'lr', 'loss_mean_100', 'loss_mask_mean_100', 'loss_maskfg_mean_100',
                 'grad_norm_preclip_mean_100', 'frac_clipped_100', 'mean_abs_delta_last',
                 'mask_frac_last'])
    vfields = ['update', 'psnr_in_mask', 'psnr_in_mask_fg', 'mae_in_mask',
               'n_images_with_mask', 'max_abs_change_outside_mask', 'improved',
               'best_update', 'nonimproving_streak', 'checkpoint', 'seconds']
    vw = csv.DictWriter(vlog, fieldnames=vfields)
    vw.writeheader()

    def ckpt_path(t):
        return os.path.join(mc.CKPT_DIR, f'ckpt_update_{t:06d}.pth')

    def save(t, val):
        p = ckpt_path(t)
        mc.save_checkpoint_exclusive(p, {'params': net.state_dict(), 'update': t,
                                         'val': val, 'config': cfg,
                                         'params_count': count_params(net)})
        return p

    t_start = time.time()
    v0 = validate()
    p0 = save(0, v0)
    with open(mc.assert_inside_root(started), 'x') as f:
        f.write(f'{mc.now()} job {os.environ.get("SLURM_JOB_ID")}\n')
    if v0['max_abs_change_outside_mask'] != 0.0:
        raise SystemExit('step 0: the model changed pixels outside the mask')
    print(f'step 0 (identity): in-mask PSNR {v0["psnr_in_mask"]:.4f} on '
          f'{v0["n_images_with_mask"]} val images', flush=True)
    vw.writerow({'update': 0, **v0, 'improved': '', 'best_update': 0,
                 'nonimproving_streak': 0, 'checkpoint': os.path.basename(p0),
                 'seconds': 0.0})
    vlog.flush()

    best, best_t, best_ck, streak = v0['psnr_in_mask'], 0, p0, 0
    t, stop_reason = 0, None
    losses, lm, lf, gns, clipped = [], [], [], [], []
    if dev.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    while stop_reason is None:
        perm = torch.randperm(ntr, generator=gen).numpy()
        for b in range(per_epoch):
            idx = perm[b * B:(b + 1) * B]
            lr = mc.cosine_lr(t, cfg['lr'], cfg['lr_min'], cfg['max_updates'])
            for gr in opt.param_groups:
                gr['lr'] = lr
            g_np = mc.to_unit(gtr_u[idx])
            d_np, m_np, _ = corruption.corrupt_batch(g_np, rng)
            x = torch.from_numpy(mc.to_unit(xtr_u[idx]))[:, None].to(dev)
            g = torch.from_numpy(g_np)[:, None].to(dev)
            yin = torch.from_numpy(d_np)[:, None].to(dev)
            m = torch.from_numpy(m_np)[:, None].to(dev)

            y, _d = net(x, yin, m)
            err = (y - g).abs()
            l_mask, n_m = region_l1(err, m)
            l_fg, n_f = region_l1(err, m * (g > mc.FG_THRESHOLD).float())
            terms = [(0.5, l_mask), (0.5, l_fg)]
            loss = sum(w * v for w, v in terms if v is not None)
            if not torch.is_tensor(loss):
                continue                                   # no region in this batch
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg['grad_clip_norm'])
            if not (torch.isfinite(loss) and torch.isfinite(gn)):
                with open(mc.assert_inside_root(os.path.join(mc.ROOT, 'NONFINITE')), 'w') as f:
                    f.write(f'update {t + 1}: loss {float(loss)} grad {float(gn)}\n')
                raise SystemExit(f'non-finite loss/gradient at update {t + 1}')
            opt.step()
            t += 1
            losses.append(float(loss))
            lm.append(float(l_mask) if l_mask is not None else np.nan)
            lf.append(float(l_fg) if l_fg is not None else np.nan)
            gns.append(float(gn))
            clipped.append(float(gn) > cfg['grad_clip_norm'])
            if t % 100 == 0:
                tw.writerow([t, lr, np.mean(losses), np.nanmean(lm), np.nanmean(lf),
                             np.mean(gns), np.mean(clipped),
                             float(_d.detach().abs().mean()), float(m.mean())])
                tlog.flush()
                print(f'update {t:5d}  lr {lr:.3e}  loss {np.mean(losses):.6f}  '
                      f'mask {np.nanmean(lm):.6f}  maskfg {np.nanmean(lf):.6f}  '
                      f'grad {np.mean(gns):.3e}', flush=True)
                losses, lm, lf, gns, clipped = [], [], [], [], []
            if t % cfg['val_every'] == 0:
                v = validate()
                p = save(t, v)
                improved = v['psnr_in_mask'] > best
                if improved:
                    best, best_t, best_ck, streak = v['psnr_in_mask'], t, p, 0
                else:
                    streak += 1
                vw.writerow({'update': t, **v, 'improved': improved, 'best_update': best_t,
                             'nonimproving_streak': streak, 'checkpoint': os.path.basename(p),
                             'seconds': round(time.time() - t_start, 1)})
                vlog.flush()
                print(f'  VAL {t:5d}  in-mask PSNR {v["psnr_in_mask"]:.4f}  '
                      f'in-mask-fg {v["psnr_in_mask_fg"]:.4f}  MAE {v["mae_in_mask"]:.5f}  '
                      f'best {best:.4f}@{best_t}  streak {streak}', flush=True)
                if t >= cfg['min_updates'] and streak >= cfg['patience_checks']:
                    stop_reason = (f'early stop: {streak} consecutive validation checks without '
                                   f'improvement, at update {t} (>= {cfg["min_updates"]})')
                    break
            if t >= cfg['max_updates']:
                stop_reason = f'hard maximum {cfg["max_updates"]} updates'
                break

    vfin = validate()
    pfin = save(t, vfin) if t % cfg['val_every'] != 0 else None
    wall = time.time() - t_start
    summary = {'experiment': mc.EXP, 'root': mc.ROOT, 'updates_done': t,
               'stop_reason': stop_reason, 'epochs_equivalent': t / per_epoch,
               'batches_per_epoch': per_epoch, 'trainable_params': count_params(net),
               'step0_val': v0, 'final_val': vfin,
               'wall_seconds_incl_validation': wall,
               'peak_gpu_mem_gib': (torch.cuda.max_memory_allocated() / 2 ** 30
                                    if dev.startswith('cuda') else None),
               'device': mc.device_record(dev), 'config': cfg,
               'corruption_recipe': corruption.recipe_record(),
               'val_task': os.path.join(mc.CACHE, 'val_synth_masks.npz'),
               'n_train': ntr, 'n_val': len(ids_va),
               'final_extra_checkpoint': pfin, 'created': mc.now()}
    mc.write_json_exclusive(os.path.join(mc.RESULTS, 'train_summary.json'), summary)
    selection = {'rule': cfg['selection_metric'],
                 'selected_update': best_t, 'selected_checkpoint': best_ck,
                 'selected_value_psnr_in_mask': best,
                 'step0_value_psnr_in_mask': v0['psnr_in_mask'],
                 'identity_selected': best_t == 0,
                 'candidates': 'every saved ckpt_update_*.pth, one per validation check',
                 'selection_used_actual_E0_failure_results': False,
                 'created': mc.now()}
    mc.write_json_exclusive(os.path.join(mc.RESULTS, 'checkpoint_selection.json'), selection)
    with open(mc.assert_inside_root(os.path.join(mc.ROOT, 'TRAINING_DONE')), 'x') as f:
        f.write(f'{mc.now()}\n')
    print(f'\nDONE: {stop_reason}. selected update {best_t} '
          f'(in-mask PSNR {best:.4f}); {t} updates, {wall / 60:.1f} min')


if __name__ == '__main__':
    main()
