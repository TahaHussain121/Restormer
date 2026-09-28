"""Real-data smoke checks. Run on a GPU node BEFORE the training run.

Every check uses real dataset frames and the real frozen-E0 cache; nothing is
simulated. Results are written to results/smoke_checks.json inside the root.
The script never creates the training markers and never writes a checkpoint
under the training names.
"""
import hashlib, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mc_common as mc                                      # noqa: E402
import corruption                                            # noqa: E402
from completion_arch import MaskedCompletionUNet, count_params, EXPECTED_PARAMS  # noqa: E402

CHECKS = []


def check(name, ok, detail=None):
    CHECKS.append({'check': name, 'pass': bool(ok), 'detail': detail})
    print(('PASS  ' if ok else 'FAIL  ') + name + (f'   {detail}' if detail else ''), flush=True)
    return bool(ok)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    mc.strict_fp32()
    torch.backends.cudnn.benchmark = False

    # existing files that this run reads: hashed before and after the smoke run
    watched = [mc.CASES_JSON,
               os.path.join(mc.REF_E0_DIR, 'refiner_arch.py'),
               os.path.join(mc.REF_E0_DIR, 'refiner_common.py'),
               mc.cache_paths('val')['prov'], mc.cache_paths('train')['prov'],
               mc.E0_BEST_JSON, mc.E0_KEPT_JSON,
               os.path.join(mc.DATASET, 'val_clean', mc.split_ids('val')[0] + '.png')]
    before = {p: (sha256(p), os.stat(p).st_mtime_ns) for p in watched}

    # 1. isolation: writing outside the root is refused ------------------------
    refused = []
    for bad in ('/tmp/should_not_write.txt',
                os.path.join(mc.REPO, 'dino_analysis_phases', 'refiner_e0', 'X.txt'),
                os.path.join(mc.ROOT, '..', 'escape.txt')):
        try:
            mc.assert_inside_root(bad)
            refused.append((bad, 'ACCEPTED'))
        except mc.OutsideRootError:
            refused.append((bad, 'refused'))
    check('write paths outside the root are refused',
          all(r[1] == 'refused' for r in refused), refused)
    check('the root contains no symlink component',
          not any(os.path.islink(os.path.join(mc.ROOT, d)) for d in os.listdir(mc.ROOT)))

    # 2. splits ----------------------------------------------------------------
    ids_tr, ids_va = mc.split_ids('train'), mc.split_ids('val')
    check('train / validation membership is disjoint and unchanged',
          not (set(ids_tr) & set(ids_va)) and (len(ids_tr), len(ids_va)) == (6101, 339),
          {'n_train': len(ids_tr), 'n_val': len(ids_va)})

    # 3. the frozen E0: present only as a read-only cache -----------------------
    info = mc.resolve_e0_checkpoint()
    arr, cids, prov = mc.load_cache('val', ids_va, mmap=True)
    check('E0 checkpoint resolves from existing provenance and its md5 matches',
          info['md5'] == prov['e0']['md5'],
          {'iter': info['best_iter'], 'md5': info['md5']})
    check('E0 cache opened read-only (memmap mode r)', arr.flags.writeable is False,
          {'shape': list(arr.shape), 'dtype': str(arr.dtype)})
    src = open(os.path.join(mc.CODE, 'train_completion.py')).read()
    check('training code never builds or imports E0',
          ('build_model' not in src) and ('predict_phase3' not in src)
          and ('load_cache' not in src))

    # 4. model, channels, optimiser --------------------------------------------
    torch.manual_seed(100)
    net = MaskedCompletionUNet().to(dev)
    check('parameter count is the declared 118,273',
          count_params(net) == EXPECTED_PARAMS, count_params(net))
    check('first convolution takes exactly 3 input channels',
          net.e1[0].in_channels == 3 and net.e1[0].weight.shape[1] == 3)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    opt_ids = {id(p) for g in opt.param_groups for p in g['params']}
    check('the optimizer holds ONLY the completion network parameters and no E0 tensor',
          opt_ids == {id(p) for p in net.parameters()} and len(opt_ids) == 22,
          {'n_tensors': len(opt_ids)})

    # 5. real data through the real preprocessing ------------------------------
    ids = ids_tr[:16]
    x_u = np.stack([mc.load_uint16(os.path.join(mc.DATASET, 'train_verynoisy', f'{i}.png'))
                    for i in ids])
    g_u = np.stack([mc.load_uint16(os.path.join(mc.DATASET, 'train_clean', f'{i}.png'))
                    for i in ids])
    g_np = mc.to_unit(g_u)
    check('preprocessing is uint16 / 65535 float32 in [0,1], 256x256',
          g_np.dtype == np.float32 and g_np.shape[1:] == (256, 256)
          and 0.0 <= g_np.min() and g_np.max() <= 1.0)
    rng = np.random.default_rng(100)
    d_np, m_np, infos = corruption.corrupt_batch(g_np, rng)
    outside_equal = float(np.abs((d_np - g_np) * (1 - m_np)).max())
    check('the damaged image equals the clean target EXACTLY outside the mask',
          outside_equal == 0.0, outside_equal)
    dmg_regions = [r for i in infos for r in i['regions'] if not r['background_region']]
    bg_regions = [r for i in infos for r in i['regions'] if r['background_region']]
    inside_lower = float(((d_np - g_np) * m_np).max())
    check('damage only removes intensity (no brightening inside the mask)',
          inside_lower <= 1e-7, inside_lower)
    check('the recipe produced compact holes, bands AND background-only masks',
          {r['kind'] for r in dmg_regions} >= {'ellipse', 'band'} and len(bg_regions) > 0,
          {'kinds': sorted({r['kind'] for r in dmg_regions}),
           'n_damaged_regions': len(dmg_regions), 'n_background_regions': len(bg_regions),
           'modes': sorted({r['mode'] for r in dmg_regions})})

    x = torch.from_numpy(mc.to_unit(x_u))[:, None].to(dev)
    g = torch.from_numpy(g_np)[:, None].to(dev)
    yin = torch.from_numpy(d_np)[:, None].to(dev)
    m = torch.from_numpy(m_np)[:, None].to(dev)

    # 6. the input tensor is exactly [X, Y_in, M] -------------------------------
    seen = {}
    h = net.e1[0].register_forward_pre_hook(lambda mod, inp: seen.update(inp=inp[0].detach()))
    y0, d0 = net(x, yin, m)
    h.remove()
    inp = seen['inp']
    check('the network input is exactly concat(X, Y_in, M) in that order',
          inp.shape[1] == 3 and torch.equal(inp[:, 0:1], x)
          and torch.equal(inp[:, 1:2], yin) and torch.equal(inp[:, 2:3], m))
    check('ground truth is NOT among the inputs',
          not torch.equal(inp[:, 1:2], g) and not any(
              torch.equal(inp[:, c:c + 1], g) for c in range(3)))

    # 7. identity at initialisation and exact preservation outside the mask -----
    check('at initialisation delta == 0 and Y_out == Y_in exactly',
          float(d0.abs().max()) == 0.0 and torch.equal(y0, yin))
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * 0.05)              # a deliberately non-trivial net
    with torch.no_grad():
        y1, d1 = net(x, yin, m)
    out_change = float(((y1 - yin).abs() * (1 - m)).max())
    check('with non-zero weights, pixels outside the mask are bit-identical to the input',
          out_change == 0.0 and torch.equal(y1 * (1 - m), yin * (1 - m)), out_change)
    q_in, q_out = mc.quantize_np(yin[:, 0].cpu().numpy()), mc.quantize_np(y1[:, 0].cpu().numpy())
    mm = m[:, 0].cpu().numpy() > 0
    check('preservation survives the uint16 serialisation convention',
          int(np.abs(q_in.astype(np.int64) - q_out.astype(np.int64))[~mm].max()) == 0)
    tmp_png = os.path.join(mc.ROOT, 'results', 'smoke', 'roundtrip.png')
    mc.imwrite_guarded(tmp_png, q_out[0])
    back = mc.load_uint16(tmp_png)
    check('PNG round-trip is lossless and preserves the outside-mask pixels',
          np.array_equal(back, q_out[0]))

    # 8. empty mask -------------------------------------------------------------
    zero = torch.zeros_like(m)
    with torch.no_grad():
        y2, d2 = net(x, yin, zero)
    check('an EMPTY mask leaves the image exactly unchanged',
          torch.equal(y2, yin) and float(d2.abs().max()) == 0.0)
    import train_completion as tc
    v, n = tc.region_l1((y2 - g).abs(), zero)
    check('the masked loss handles an all-empty region without NaN',
          v is None and n == 0, {'value': v, 'n_images': n})
    half = m.clone()
    half[: len(ids) // 2] = 0
    vh, nh = tc.region_l1((y1 - g).abs(), half)
    check('with some empty masks the loss averages only over images that have one',
          vh is not None and nh == len(ids) - len(ids) // 2 and torch.isfinite(vh),
          {'n_images_with_region': nh, 'value': float(vh)})

    # 9. finite losses and gradients on real data -------------------------------
    torch.manual_seed(100)
    net2 = MaskedCompletionUNet().to(dev)
    opt2 = torch.optim.AdamW(net2.parameters(), lr=1e-4, weight_decay=1e-4)
    grads, steps = [], []
    for step in range(3):
        y, _ = net2(x, yin, m)
        err = (y - g).abs()
        lm, _ = tc.region_l1(err, m)
        lf, _ = tc.region_l1(err, m * (g > mc.FG_THRESHOLD).float())
        loss = 0.5 * lm + 0.5 * lf
        opt2.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net2.parameters(), 1.0)
        n_with_grad = sum(1 for p in net2.parameters()
                          if p.grad is not None and float(p.grad.abs().max()) > 0)
        opt2.step()
        steps.append({'step': step + 1, 'loss': float(loss), 'grad_norm': float(gn),
                      'tensors_with_nonzero_grad': n_with_grad})
        grads.append(n_with_grad)
    check('losses and gradients are finite for three real training steps',
          all(np.isfinite(s['loss']) and np.isfinite(s['grad_norm']) for s in steps), steps)
    check('gradient reaches the zero-initialised output layer first, then every layer',
          grads[0] == 2 and grads[-1] == 22, grads)
    with torch.no_grad():
        y3, _ = net2(x, yin, m)
    check('three steps changed the prediction only inside the mask',
          float(((y3 - yin).abs() * (1 - m)).max()) == 0.0)

    # 10. the oracle-mask rule on real validation data --------------------------
    gv = mc.to_unit(mc.load_uint16(os.path.join(mc.DATASET, 'val_clean', f'{ids_va[0]}.png')))
    e0raw = np.asarray(arr[cids.index(ids_va[0])], dtype=np.float32)
    e0q = mc.quantize_np(e0raw).astype(np.float64) / 65535.
    ref = mc.load_uint16(os.path.join(mc.E0_REF_PRED.format(split='val'), f'{ids_va[0]}.png'))
    check('the cache quantises to E0\'s recorded prediction bit-exactly',
          np.array_equal(mc.quantize_np(e0raw), ref))
    orc = (gv > mc.MISS_GT) & (e0q < mc.MISS_E0)
    check('the documented oracle-mask rule (target > 0.05 and E0 < 0.02) is non-degenerate',
          0 < orc.mean() < 0.5, {'frac': float(orc.mean())})

    # 11. no existing file changed ---------------------------------------------
    after = {p: (sha256(p), os.stat(p).st_mtime_ns) for p in watched}
    changed = [p for p in watched if before[p] != after[p]]
    check('no watched existing file changed during the smoke run', not changed, changed)

    out = {'created': mc.now(), 'device': mc.device_record(dev),
           'n_checks': len(CHECKS), 'n_passed': sum(c['pass'] for c in CHECKS),
           'checks': CHECKS, 'train_steps': steps}
    mc.write_json_exclusive(os.path.join(mc.RESULTS, 'smoke_checks.json'), out)
    print(f'\n{out["n_passed"]}/{out["n_checks"]} checks passed')
    if out['n_passed'] != out['n_checks']:
        raise SystemExit('SMOKE FAILED')


if __name__ == '__main__':
    main()
