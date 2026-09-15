"""Small targeted checks before training (CPU, real data where it matters).

Only what this follow-up changes or relies on: config identity with
refiner_e0, cache reuse and provenance, identical initialisation, the region
loss (including empty regions), learning under the new loss, frozen E0, and the
permitted inference inputs. The architecture and data path were verified by
refiner_e0's 26/26 smoke and are reused unchanged.
"""

import inspect
import os
import sys

import numpy as np
import torch

import fgbal_common as fc
import refiner_common as rc
from refiner_arch import ResidualRefinerUNet, count_params, EXPECTED_PARAMS

OUT = []


def check(name, ok, detail):
    OUT.append({'check': name, 'pass': bool(ok), 'detail': str(detail)})
    print(f'  [{"PASS" if ok else "FAIL"}] {name}: {detail}', flush=True)


def main():
    torch.set_num_threads(4)
    new, old, diff = fc.load_configs()
    check('config identical to refiner_e0 except loss/selection/name', True, diff)

    info = rc.resolve_e0_checkpoint()
    check('E0 checkpoint unchanged (md5 = KEPT_CHECKPOINTS)', info['md5'] == info['md5_recorded'],
          f"iter {info['best_iter']} md5 {info['md5']}")

    ids_tr, ids_va = rc.split_ids('train'), rc.split_ids('val')
    ytr, _, ptr = rc.load_cache('train', ids_tr, mmap=True)
    yva, _, pva = rc.load_cache('val', ids_va, mmap=True)
    sha = {s: rc.file_digest(rc.cache_paths(s)['npy'], 'sha256') for s in ('train', 'val')}
    check('reused caches match their provenance (sha256, E0 md5, id order)',
          sha['train'] == ptr['output']['sha256'] and sha['val'] == pva['output']['sha256'],
          f"train {sha['train'][:16]}, val {sha['val'][:16]}; ids {len(ids_tr)}/{len(ids_va)}")
    check('dataset membership and preprocessing identical to refiner_e0',
          ptr['preprocessing'] == pva['preprocessing'] and ptr['n_images'] == 6101
          and pva['n_images'] == 339, ptr['preprocessing'][:80] + ' ...')

    torch.manual_seed(new['seed'])
    net = ResidualRefinerUNet()
    old0 = torch.load(os.path.join(fc.OLD_EXP_DIR, 'models', 'refiner_step0.pth'),
                      map_location='cpu', weights_only=False)['params']
    # Same seed and procedure. Bit-identity is only expected on the same CPU
    # type: refiner_e0 drew its init on a v100 node's CPU, this check runs on
    # the login node, and the normal sampler's vectorised maths can differ in
    # the last float32 bits. The training job repeats this on the GPU node.
    diffs = {k: float((net.state_dict()[k] - old0[k]).abs().max()) for k in old0}
    n_exact = sum(v == 0 for v in diffs.values())
    check('fresh refiner == refiner_e0 initial weights to float32 rounding (seed 100), '
          'not its trained weights', max(diffs.values()) <= 1e-6 and count_params(net) == EXPECTED_PARAMS,
          f'{count_params(net):,} params; {n_exact}/{len(old0)} tensors bit-identical, max abs diff '
          f'{max(diffs.values()):.2e} (for scale: refiner_e0 at update 6000 differs from its step 0 by a '
          f'per-tensor max of 3.2e-05 to 0.19, median 4.9e-02, measured 2026-09-15)')

    # --- region loss -----------------------------------------------------------
    g = torch.zeros(4, 1, 8, 8)
    g[0, 0, :4] = 0.5                          # half fg
    g[1, 0, 0, 0] = 0.3                        # one fg pixel
    # g[2] all background; g[3] all foreground
    g[3] = 0.2
    y = g + torch.linspace(-0.1, 0.1, 256).reshape(4, 1, 8, 8)
    loss, parts = fc.region_l1(y, g)
    e = (y - g).abs().numpy()
    m = (g > 0.01).numpy()
    ref_fg = np.mean([e[i][m[i]].mean() for i in range(4) if m[i].any()])
    ref_bg = np.mean([e[i][~m[i]].mean() for i in range(4) if (~m[i]).any()])
    check('region loss == per-image region means, averaged over images with the region',
          abs(float(loss) - (0.5 * ref_fg + 0.5 * ref_bg)) < 1e-7 and parts['n_fg_images'] == 3
          and parts['n_bg_images'] == 3, f'{float(loss):.7f} vs {0.5 * ref_fg + 0.5 * ref_bg:.7f}; {parts}')
    check('no dilution: the one-pixel image weighs as much as the half-fg image in the fg term',
          abs(parts['fg'] - np.mean([e[0][m[0]].mean(), e[1][m[1]].mean(), e[3][m[3]].mean()])) < 1e-7,
          f"fg term {parts['fg']:.6f}")
    for label, gg in (('all-background batch', torch.zeros(2, 1, 8, 8)),
                      ('all-foreground batch', torch.full((2, 1, 8, 8), 0.5))):
        yy = (gg + 0.05).requires_grad_(True)
        l2, p2 = fc.region_l1(yy, gg)
        l2.backward()
        check(f'empty region, {label}: finite loss and gradient, empty term omitted',
              torch.isfinite(l2) and torch.isfinite(yy.grad).all() and abs(float(l2) - 0.025) < 1e-7,
              f'loss {float(l2):.4f} (= 0.5 x 0.05), {p2}')

    # --- permitted inputs -------------------------------------------------------
    params = list(inspect.signature(net.forward).parameters)
    check('inference takes only (X, Y0); the mask exists only inside the loss',
          params == ['x', 'y0'], f'forward{tuple(params)}')

    # --- zero init reproduces E0; learning under the new loss --------------------
    xu, gu = rc.load_split_uint16('val', ids_va[:8])
    x = torch.from_numpy(rc.to_unit(xu))[:, None]
    y0 = torch.from_numpy(np.ascontiguousarray(yva[:8]))[:, None]
    with torch.no_grad():
        yh, d = net(x, y0)
    check('zero initialisation reproduces the cached E0 output exactly',
          float(d.abs().max()) == 0 and torch.equal(yh, y0), 'max|delta| 0, Y == Y0 (8 val frames)')

    opt = torch.optim.AdamW(net.parameters(), lr=new['lr'], betas=tuple(new['betas']),
                            weight_decay=new['weight_decay'])
    n_opt = sum(p.numel() for gr in opt.param_groups for p in gr['params'])
    check('optimizer holds only the refiner (E0 is never built in training)',
          n_opt == EXPECTED_PARAMS, f'{n_opt:,} parameters in the optimizer')
    xtu, gtu = rc.load_split_uint16('train', [ids_tr[k] for k in (0, 1500, 3000, 4500)])
    xt = torch.from_numpy(rc.to_unit(xtu))[:, None]
    gt = torch.from_numpy(rc.to_unit(gtu))[:, None]
    y0t = torch.from_numpy(np.ascontiguousarray(ytr[[0, 1500, 3000, 4500]]))[:, None]
    gn = []
    for s in range(3):
        yh, _ = net(xt, y0t)
        loss, parts = fc.region_l1(yh, gt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn.append({k: float(sum((p.grad.double() ** 2).sum() for p in getattr(net, k).parameters()) ** .5)
                   for k in ('e1', 'bottleneck', 'd1', 'out')})
        torch.nn.utils.clip_grad_norm_(net.parameters(), new['grad_clip_norm'])
        opt.step()
        print(f'    step {s + 1}: loss {float(loss):.6f} {parts} grads {gn[-1]}')
    check('refiner learns: output conv at step 1, every layer from step 2, finite',
          gn[0]['out'] > 0 and gn[0]['e1'] == 0 and all(v > 0 for v in gn[1].values())
          and all(np.isfinite(list(g_.values())).all() for g_ in gn), gn[:2])

    ok = all(o['pass'] for o in OUT)
    rc.write_json(os.path.join(fc.RESULTS, 'smoke', 'smoke_fgbal.json'),
                  {'experiment': fc.EXP, 'all_pass': ok, 'checks': OUT, 'created': rc.now()})
    print(f'\nSMOKE {"PASS" if ok else "FAIL"}: {sum(o["pass"] for o in OUT)}/{len(OUT)}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
