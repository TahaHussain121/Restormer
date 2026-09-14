"""Pre-training smoke checks on REAL data, run on GPU after the cache exists.

Every check prints PASS/FAIL and the measured value; the script exits non-zero
on any failure and writes results/smoke/smoke_results.json. Training is not
submitted unless this passes and has been read.
"""

import copy
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr

import refiner_common as rc
from refiner_arch import ResidualRefinerUNet, count_params, EXPECTED_PARAMS

RESULTS = []
DEV = 'cuda'


def check(name, ok, detail):
    RESULTS.append({'check': name, 'pass': bool(ok), 'detail': detail})
    print(f'  [{"PASS" if ok else "FAIL"}] {name}: {detail}', flush=True)


def e0_digest(net):
    h = hashlib.md5()
    for k, v in net.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def module_grad_norms(net):
    out = {}
    for name in ('e1', 'e2', 'bottleneck', 'd2', 'd1', 'out'):
        gs = [p.grad for p in getattr(net, name).parameters()]
        out[name] = (0.0 if any(g is None for g in gs) else
                     float(torch.sqrt(sum((g.double() ** 2).sum() for g in gs))))
    return out


def batch(split_u16, cache, idx):
    xu, gu = split_u16
    x = torch.from_numpy(rc.to_unit(xu[idx]))[:, None].to(DEV)
    g = torch.from_numpy(rc.to_unit(gu[idx]))[:, None].to(DEV)
    y0 = torch.from_numpy(np.ascontiguousarray(cache[idx]))[:, None].to(DEV)
    return x, y0, g


def main():
    rc.strict_fp32()
    torch.backends.cudnn.benchmark = False
    info = rc.resolve_e0_checkpoint()
    print(f'E0 {info["checkpoint"]} (iter {info["best_iter"]}, md5 ok)')

    # --- architecture -------------------------------------------------------
    torch.manual_seed(100)
    net = ResidualRefinerUNet().to(DEV)
    n = count_params(net)
    per = {k: sum(p.numel() for p in getattr(net, k).parameters())
           for k in ('e1', 'e2', 'bottleneck', 'd2', 'd1', 'out')}
    check('parameter count', n == EXPECTED_PARAMS, f'{n:,} ({per})')
    check('output conv zero weight and bias',
          float(net.out.weight.abs().max()) == 0 and float(net.out.bias.abs().max()) == 0,
          'max |w|, |b| = 0')
    kinds = sorted({type(m).__name__ for m in net.modules()})
    check('only Conv2d/ReLU/AvgPool2d/Sequential', set(kinds) <= {
        'ResidualRefinerUNet', 'Sequential', 'Conv2d', 'ReLU', 'AvgPool2d'}, kinds)
    check('TF32 disabled', not torch.backends.cuda.matmul.allow_tf32 and
          not torch.backends.cudnn.allow_tf32, rc.device_record(DEV))

    # --- caches, pairing, normalisation --------------------------------------
    ids_tr, ids_va = rc.split_ids('train'), rc.split_ids('val')
    ytr, _, prov_tr = rc.load_cache('train', ids_tr, mmap=True)
    yva, _, prov_va = rc.load_cache('val', ids_va)
    check('cache sizes', ytr.shape == (6101, 256, 256) and yva.shape == (339, 256, 256),
          f'train {ytr.shape}, val {yva.shape}, float32')
    ref = prov_va['reference_check']
    check('val cache reproduces E0 recorded predictions (bit-exact expected on v100)',
          ref['n_identical_images'] == ref['n_images'],
          f"{ref['n_identical_images']}/{ref['n_images']} identical, max level diff "
          f"{ref['max_abs_level_diff_uint16']}, max per-image PSNR diff "
          f"{ref['max_abs_per_image_psnr_diff_db']:.2e} dB, mean "
          f"{ref['mean_psnr_full_from_cache']:.6f} vs {ref['mean_psnr_full_reference']:.6f}")
    check('val reproduction within tolerance (per-image PSNR < 1e-3 dB)',
          ref['max_abs_per_image_psnr_diff_db'] < 1e-3,
          f"{ref['max_abs_per_image_psnr_diff_db']:.2e}")

    from basicsr.utils import imfrombytes_uint16
    worst = 0.0
    for split, i in (('train', ids_tr[0]), ('train', ids_tr[4000]), ('val', ids_va[7])):
        for sub in ('verynoisy', 'clean'):
            p = os.path.join(rc.DATASET, f'{split}_{sub}', f'{i}.png')
            with open(p, 'rb') as f:
                a = imfrombytes_uint16(f.read())[:, :, 0]
            b = rc.to_unit(rc.load_uint16(p))
            worst = max(worst, float(np.abs(a - b).max()))
    check('normalisation identical to the training loader (imfrombytes_uint16)',
          worst == 0.0, f'max abs diff {worst}')

    tr_u16 = rc.load_split_uint16('train', ids_tr)
    va_u16 = rc.load_split_uint16('val', ids_va)
    check('pairing: noisy and target differ (not the same file)',
          not np.array_equal(tr_u16[0][0], tr_u16[1][0]), 'train[0] noisy != target')
    check('value ranges', all(float(a.max()) <= 65535 for a in tr_u16 + va_u16),
          f'train X max {tr_u16[0].max()}, gt max {tr_u16[1].max()}')

    e0, _ = rc.build_frozen_e0(DEV)
    d_before = e0_digest(e0)
    worst, idxs = 0.0, [0, 1234, 6100]
    with torch.no_grad():
        for k in idxs:
            t = torch.from_numpy(rc.to_unit(tr_u16[0][k]))[None, None].to(DEV)
            live = e0(t)[0, 0].cpu().numpy()
            worst = max(worst, float(np.abs(live - ytr[k]).max()))
    check('train cache == live E0 at the same ids (pairing of cache rows)',
          worst == 0.0, f'ids {[ids_tr[k] for k in idxs]}, max abs diff {worst}')

    # --- input identity, shapes, zero init ----------------------------------
    idx = np.arange(16) * 381
    x, y0, g = batch(tr_u16, ytr, idx)
    cap = {}
    h = net.e1[0].register_forward_hook(lambda m, i, o: cap.__setitem__('inp', i[0].detach().clone()))
    with torch.no_grad():
        y, delta = net(x, y0)
    h.remove()
    inp = cap['inp']
    check('refiner receives exactly [X, Y0]', inp.shape[1] == 2 and torch.equal(inp[:, :1], x)
          and torch.equal(inp[:, 1:], y0), f'first conv input {tuple(inp.shape)}, ch0==X, ch1==Y0')
    check('shapes', tuple(delta.shape) == tuple(y.shape) == (16, 1, 256, 256),
          f'delta {tuple(delta.shape)}, Y {tuple(y.shape)}')
    check('initial residual exactly zero', float(delta.abs().max()) == 0.0,
          f'max |delta| {float(delta.abs().max())}')
    check('initial output equals Y0 exactly', torch.equal(y, y0), 'torch.equal')

    # --- metric path ----------------------------------------------------------
    q_np = np.stack([rc.quantize_np(yva[k]) for k in range(len(ids_va))])
    q_t = rc.quantize_torch(torch.from_numpy(yva)).numpy().astype(np.uint16)
    check('torch quantisation == numpy (predict_phase3) quantisation, full val',
          np.array_equal(q_np, q_t), f'{int((q_np != q_t).sum())} differing pixels')
    gtv = va_u16[1]
    ps_np = np.array([rc.psnr_u16(gtv[k], q_np[k]) for k in range(len(ids_va))])
    ps_sk = np.array([skimage_psnr(gtv[k].astype(np.float64) / 65535.,
                                   q_np[k].astype(np.float64) / 65535., data_range=1.0)
                      for k in range(len(ids_va))])
    ps_t = rc.psnr_levels_torch(torch.from_numpy(gtv.astype(np.float32))[:, None],
                                torch.from_numpy(q_t.astype(np.float32))[:, None]).numpy()
    check('PSNR: own == skimage == torch path', max(np.abs(ps_np - ps_sk).max(),
          np.abs(ps_t - ps_sk).max()) < 1e-9,
          f'max diffs {np.abs(ps_np - ps_sk).max():.1e}, {np.abs(ps_t - ps_sk).max():.1e}')
    check('identity (step-0) val mean equals E0 reference val mean',
          abs(ps_sk.mean() - ref['mean_psnr_full_reference']) < 1e-6,
          f'{ps_sk.mean():.6f} vs {ref["mean_psnr_full_reference"]:.6f}')

    # --- learning on real data, with the LIVE frozen E0 ----------------------
    torch.manual_seed(100)
    lnet = ResidualRefinerUNet().to(DEV)
    opt = torch.optim.AdamW(lnet.parameters(), lr=1e-4, betas=(0.9, 0.999),
                            weight_decay=1e-4)
    opt_ids = {id(p) for gr in opt.param_groups for p in gr['params']}
    check('no E0 parameter in the optimizer',
          not any(id(p) in opt_ids for p in e0.parameters()),
          f'{len(opt_ids)} optimizer tensors, all refiner')
    steps, rng = [], np.random.default_rng(0)
    for s in range(1, 7):
        ib = rng.choice(len(ids_tr), 4, replace=False)
        x = torch.from_numpy(rc.to_unit(tr_u16[0][ib]))[:, None].to(DEV)
        g = torch.from_numpy(rc.to_unit(tr_u16[1][ib]))[:, None].to(DEV)
        with torch.no_grad():
            y0 = e0(x)
        y, delta = lnet(x, y0)
        loss = (y - g).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = module_grad_norms(lnet)
        tot = torch.nn.utils.clip_grad_norm_(lnet.parameters(), 1.0)
        opt.step()
        steps.append({'step': s, 'loss': float(loss), 'grad_total_preclip': float(tot),
                      'max_abs_delta': float(delta.detach().abs().max()), 'grads': gn})
        print(f'    step {s}: loss {float(loss):.6f}  total grad {float(tot):.3e}  '
              f'max|delta| {float(delta.detach().abs().max()):.2e}  {gn}')
    g1, g2 = steps[0]['grads'], steps[1]['grads']
    check('step 1: final conv receives gradient', g1['out'] > 0, f"{g1['out']:.3e}")
    check('step 1: earlier layers exactly zero (expected: zero-init output conv)',
          all(g1[k] == 0 for k in ('e1', 'e2', 'bottleneck', 'd2', 'd1')),
          {k: g1[k] for k in ('e1', 'e2', 'bottleneck', 'd2', 'd1')})
    check('step 2+: every earlier layer receives gradient',
          all(st['grads'][k] > 0 for st in steps[1:] for k in g2),
          {k: f'{g2[k]:.2e}' for k in g2})
    check('loss and gradients finite', all(np.isfinite(st['loss']) and np.isfinite(st['grad_total_preclip'])
                                           for st in steps), [round(st['loss'], 6) for st in steps])
    check('residual leaves zero after learning starts', steps[-1]['max_abs_delta'] > 0,
          f"max|delta| at step 6 forward {steps[-1]['max_abs_delta']:.2e}")
    check('E0: eval mode, requires_grad False, no .grad, parameters unchanged',
          (not e0.training) and all((not p.requires_grad) and p.grad is None
                                     for p in e0.parameters())
          and e0_digest(e0) == d_before, f'digest {d_before}')

    # --- batch 16 on the cached path: memory and speed -----------------------
    del e0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    tnet = copy.deepcopy(lnet)
    topt = torch.optim.AdamW(tnet.parameters(), lr=1e-4, weight_decay=1e-4)
    t0 = None
    for s in range(25):
        if s == 5:
            torch.cuda.synchronize()
            t0 = time.time()
        ib = rng.choice(len(ids_tr), 16, replace=False)
        x, y0, g = batch(tr_u16, ytr, ib)
        y, _ = tnet(x, y0)
        loss = (y - g).abs().mean()
        topt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tnet.parameters(), 1.0)
        topt.step()
    torch.cuda.synchronize()
    ms = (time.time() - t0) / 20 * 1000
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    check('batch 16 fits (physical, no accumulation)', peak < 8.0,
          f'peak {peak:.2f} GiB allocated, {ms:.1f} ms/update incl. host loading, '
          f'{torch.cuda.get_device_name(0)}')

    ok = all(r['pass'] for r in RESULTS)
    rc.write_json(os.path.join(rc.RESULTS, 'smoke', 'smoke_results.json'),
                  {'experiment': rc.EXP, 'all_pass': ok,
                   'n_pass': sum(r['pass'] for r in RESULTS), 'n': len(RESULTS),
                   'checks': RESULTS, 'learning_steps': steps,
                   'device': rc.device_record(DEV), 'created': rc.now()})
    print(f'\nSMOKE {"PASS" if ok else "FAIL"}: '
          f'{sum(r["pass"] for r in RESULTS)}/{len(RESULTS)}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
