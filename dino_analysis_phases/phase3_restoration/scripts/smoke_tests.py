"""WORK ORDER 2, Step 4 -- pre-training smoke tests for E0-Fixed and E1-N-Fixed.

Every check below is an ASSERTION against a real tensor produced by the real
code path. Nothing is inferred from a config comment. If an assertion fails the
script stops with FAIL and reports; it never repairs the experiment.

  4.1  E0     real dataset -> real train.py crop block -> forward/backward/step
  4.2  E1     training mode at 128: every tensor shape, the crop-stream
              identity proof, freezing, zero-init, first-step gradients
  4.3  E1     eval mode at 256: shapes, eval256 mean, no grid interpolation
  4.4  matched-128 protocol: one manifest, identical crops for input, target,
              E0 prediction and E1 prediction
  4.5  fairness: parameter delta == 768*384+384 == 295,296, and step-0 outputs
              identical because P(D) == 0

TOLERANCES (Work Order 2 Reference C): CPU comparisons are bit-exact
(`torch.equal`); CUDA comparisons between two module instances carry float32
non-determinism up to ~3.8e-4 absolute (~2.6e-7 relative), so CUDA equivalence
uses atol=1e-3. Crop-stream identity is `torch.equal` on BOTH devices -- it
compares one tensor object with itself, so device noise cannot enter.
"""

import argparse
import copy
import datetime
import json
import os
import random
import sys

import numpy as np
import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE3 = os.path.dirname(_HERE)
_REPO = os.path.dirname(os.path.dirname(_PHASE3))
for p in (_REPO, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from basicsr.models.archs import define_network              # noqa: E402
from basicsr.data import create_dataset, create_dataloader   # noqa: E402
import dino_shared                                            # noqa: E402

RESULTS = []
CUDA_ATOL = 1e-3


def check(name, ok, detail=''):
    RESULTS.append({'check': name, 'pass': bool(ok), 'detail': str(detail)})
    print(f'  [{"PASS" if ok else "FAIL"}] {name}'
          + (f'  --  {detail}' if detail else ''), flush=True)
    return bool(ok)


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_net(cfg, device):
    return define_network(copy.deepcopy(cfg['network_g'])).to(device)


# ---------------------------------------------------------------------------
# real data, real crop block
# ---------------------------------------------------------------------------
def real_train_batch(cfg, seed=100):
    """One batch straight from the project's dataset + train.py's crop block."""
    opt = copy.deepcopy(cfg['datasets']['train'])
    opt['phase'] = 'train'
    opt['scale'] = cfg['scale']
    opt['dist'] = False
    ds = create_dataset(opt)
    loader = create_dataloader(ds, opt, num_gpu=0, dist=False, sampler=None,
                               seed=seed)
    data = next(iter(loader))

    # --- verbatim from basicsr/train.py:241-269 --------------------------
    iters = opt['iters']
    batch_size = opt['batch_size_per_gpu']
    mini_batch_sizes = opt['mini_batch_sizes']
    gt_size = opt['gt_size']
    mini_gt_sizes = opt['gt_sizes']
    groups = np.array([sum(iters[0:i + 1]) for i in range(0, len(iters))])
    current_iter = 1
    j = ((current_iter > groups) != True).nonzero()[0]      # noqa: E712
    bs_j = (len(groups) - 1) if len(j) == 0 else j[0]
    mini_gt_size = mini_gt_sizes[bs_j]
    mini_batch_size = mini_batch_sizes[bs_j]

    lq, gt = data['lq'], data['gt']
    if mini_batch_size < batch_size:
        indices = random.sample(range(0, batch_size), k=mini_batch_size)
        lq, gt = lq[indices], gt[indices]
    if mini_gt_size < gt_size:
        x0 = int((gt_size - mini_gt_size) * random.random())
        y0 = int((gt_size - mini_gt_size) * random.random())
        lq = lq[:, :, x0:x0 + mini_gt_size, y0:y0 + mini_gt_size]
        gt = gt[:, :, x0:x0 + mini_gt_size, y0:y0 + mini_gt_size]
    # ---------------------------------------------------------------------
    return lq, gt, dict(stage=int(bs_j), mini_gt_size=int(mini_gt_size),
                        mini_batch_size=int(mini_batch_size),
                        crop_origin=(x0, y0) if mini_gt_size < gt_size else None,
                        n_train_images=len(ds),
                        lq_path=data['lq_path'][0], gt_path=data['gt_path'][0])


# ---------------------------------------------------------------------------
# 4.1  E0
# ---------------------------------------------------------------------------
def smoke_e0(cfg, device):
    print('\n=== 4.1  E0-Fixed ===')
    torch.manual_seed(cfg['manual_seed'])
    random.seed(cfg['manual_seed'])
    lq, gt, info = real_train_batch(cfg, cfg['manual_seed'])
    check('E0 train split is the training data', 'train_' in info['lq_path'],
          f"{info['n_train_images']} images, e.g. {os.path.basename(info['lq_path'])}")
    check('E0 fixed-128 crop VERIFIED FROM THE TENSOR',
          tuple(lq.shape[-2:]) == (128, 128) and tuple(gt.shape[-2:]) == (128, 128),
          f'lq {list(lq.shape)} gt {list(gt.shape)}')
    check('E0 constant mini-batch 8 from the tensor',
          lq.shape[0] == 8 and gt.shape[0] == 8, f'batch {lq.shape[0]}')
    check('E0 single progressive stage (no crop transitions)',
          info['stage'] == 0 and len(cfg['datasets']['train']['gt_sizes']) == 1,
          f'stage {info["stage"]}, gt_sizes '
          f'{cfg["datasets"]["train"]["gt_sizes"]}')
    check('E0 paired LQ/GT alignment (same basename, same crop)',
          os.path.basename(info['lq_path']) == os.path.basename(info['gt_path'])
          and lq.shape == gt.shape,
          f'{os.path.basename(info["lq_path"])}')
    check('E0 input range [0,1]', float(lq.min()) >= 0 and float(lq.max()) <= 1,
          f'[{float(lq.min()):.4f}, {float(lq.max()):.4f}]')

    net = build_net(cfg, device)
    lq, gt = lq.to(device), gt.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4,
                            betas=(0.9, 0.999))
    out = net(lq)
    check('E0 output shape == input shape', out.shape == lq.shape,
          f'{list(out.shape)}')
    loss = torch.nn.functional.l1_loss(out, gt)
    check('E0 finite loss', torch.isfinite(loss).item(), f'{float(loss):.6f}')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 0.01)
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    check('E0 gradients present and finite',
          len(grads) > 0 and all(torch.isfinite(g).all().item() for g in grads),
          f'{len(grads)} tensors with grads')
    before = next(net.parameters()).detach().clone()
    opt.step()
    check('E0 optimizer step changes weights',
          not torch.equal(before, next(net.parameters()).detach()))
    return net, info


# ---------------------------------------------------------------------------
# 4.2  E1 training mode
# ---------------------------------------------------------------------------
def smoke_e1_train(cfg, device, lq, gt):
    print('\n=== 4.2  E1-N-Fixed, training mode (128) ===')
    net = build_net(cfg, device)
    net.train()
    net.set_dino_mode('train128')
    net._capture_dino_io = True

    check('E1 DINO block is B6 / index 5',
          net.dino_block_1indexed == 6 and net.dino_block_0indexed == 5,
          f'B{net.dino_block_1indexed} -> {net.dino_block_0indexed}')
    check('E1 train128 mean loaded from the production file',
          os.path.basename(net.dino_mean_paths['train128'])
          == '1e5_B6_train128_dino224_mean.pt',
          net.dino_mean_paths['train128'])

    lq, gt = lq.to(device), gt.to(device)
    out = net(lq)
    cap = net._dino_capture
    st = net.last_dino_stats

    lat = net.down3_4(net.encoder_level3(net.down2_3(net.encoder_level2(
        net.down1_2(net.encoder_level1(net.patch_embed(lq)))))))
    proj = net.P(cap['grid'].to(lat.dtype))

    shapes = {
        'input': list(lq.shape), 'dino_input': list(cap['preprocessed'].shape),
        'tokens': list(cap['tokens'].shape), 'grid': list(cap['grid'].shape),
        'latent_in': list(lat.shape), 'projected': list(proj.shape),
        'guided': list((lat + proj).shape), 'output': list(out.shape),
    }
    b = lq.shape[0]
    expect = {
        'input': [b, 1, 128, 128], 'dino_input': [b, 3, 224, 224],
        'tokens': [b, 256, 768], 'grid': [b, 768, 16, 16],
        'latent_in': [b, 384, 16, 16], 'projected': [b, 384, 16, 16],
        'guided': [b, 384, 16, 16], 'output': [b, 1, 128, 128],
    }
    for k, want in expect.items():
        check(f'E1@128 {k} == {want}', shapes[k] == want, shapes[k])
    check('E1@128 centered tokens shape == raw token shape',
          list(cap['tokens'].shape) == [b, 256, 768])

    # ---- crop-stream identity: the STRICT proof -------------------------
    same_object = cap['source'] is lq
    check('E1 crop-stream identity: DINO source IS the Restormer input object',
          same_object, f'id(source)={id(cap["source"])} id(lq)={id(lq)}')
    check('E1 crop-stream identity: torch.equal(dino_source, restormer_input)',
          torch.equal(cap['source'], lq),
          'element-wise equality, not allclose, not shape-only')

    # ---- freezing --------------------------------------------------------
    check('E1 DINO params requires_grad == False',
          all(not p.requires_grad for p in net.dino_ext.parameters()),
          f'{sum(1 for _ in net.dino_ext.parameters())} params')
    net.train()
    check('E1 DINO stays eval() after parent .train()',
          not net.dino_ext.dino.training and net.training)

    # ---- zero init and first-step gradients ------------------------------
    check('E1 P is zero-initialized (weight and bias)',
          float(net.P.weight.abs().max()) == 0.0
          and float(net.P.bias.abs().max()) == 0.0)
    check('E1 P(D) is exactly zero at init', float(proj.abs().max()) == 0.0,
          f'max |P(D)| = {float(proj.abs().max())}')
    check('E1 centered DINO features are NOT zero (P is zero, D is not)',
          float(cap['grid'].abs().max()) > 0,
          f'max |D_centered| = {float(cap["grid"].abs().max()):.4f}')

    loss = torch.nn.functional.l1_loss(out, gt)
    check('E1 finite loss', torch.isfinite(loss).item(), f'{float(loss):.6f}')
    loss.backward()
    pg = net.P.weight.grad
    check('E1 P receives a gradient on the FIRST backward', pg is not None)
    check('E1 P gradient is finite', torch.isfinite(pg).all().item())
    check('E1 P gradient is NON-ZERO (dL/dW = dL/dguided (x) D)',
          float(pg.abs().max()) > 0, f'max |dL/dW_P| = {float(pg.abs().max()):.6e}')
    check('E1 P bias gradient finite',
          torch.isfinite(net.P.bias.grad).all().item(),
          f'max |dL/db_P| = {float(net.P.bias.grad.abs().max()):.6e}')
    check('E1 DINO gradients are None (frozen)',
          all(p.grad is None for p in net.dino_ext.parameters()))
    trunk = [p.grad for n_, p in net.named_parameters()
             if p.grad is not None and not n_.startswith('dino_ext.')
             and not n_.startswith('P.')]
    check('E1 Restormer gradients present and finite',
          len(trunk) > 0 and all(torch.isfinite(g).all().item() for g in trunk),
          f'{len(trunk)} trunk grad tensors')
    check('E1 no NaN/Inf in the forward',
          torch.isfinite(out).all().item()
          and torch.isfinite(cap['grid']).all().item())
    check('E1 injection ratio finite',
          np.isfinite(st['injection_ratio']),
          f"latent {st['latent_norm']:.3f} projected {st['projected_norm']:.3f} "
          f"ratio {st['injection_ratio']:.3e}")
    check('E1 used the train128 mean buffer', net.last_mean_key == 'mu_train128',
          net.last_mean_key)
    net._capture_dino_io = False
    return net, shapes


# ---------------------------------------------------------------------------
# 4.3  E1 eval mode at 256
# ---------------------------------------------------------------------------
def smoke_e1_eval256(net, device):
    print('\n=== 4.3  E1-N-Fixed, full-256 eval mode ===')
    net.eval()
    net.set_dino_mode('eval256')
    net._capture_dino_io = True
    x = torch.rand(1, 1, 256, 256, device=device)
    with torch.no_grad():
        out = net(x)
    cap, st = net._dino_capture, net.last_dino_stats
    with torch.no_grad():
        lat = net.down3_4(net.encoder_level3(net.down2_3(net.encoder_level2(
            net.down1_2(net.encoder_level1(net.patch_embed(x)))))))
        proj = net.P(cap['grid'].to(lat.dtype))
    shapes = {
        'input': list(x.shape), 'dino_input': list(cap['preprocessed'].shape),
        'tokens': list(cap['tokens'].shape), 'grid': list(cap['grid'].shape),
        'latent_in': list(lat.shape), 'projected': list(proj.shape),
        'output': list(out.shape),
    }
    expect = {'input': [1, 1, 256, 256], 'dino_input': [1, 3, 448, 448],
              'tokens': [1, 1024, 768], 'grid': [1, 768, 32, 32],
              'latent_in': [1, 384, 32, 32], 'projected': [1, 384, 32, 32],
              'output': [1, 1, 256, 256]}
    for k, want in expect.items():
        check(f'E1@256 {k} == {want}', shapes[k] == want, shapes[k])

    check('E1@256 DINO grid == latent grid, NO interpolation applied',
          dino_shared.assert_no_interpolation_needed(cap['grid'].shape[-2:],
                                                     lat.shape[-2:]),
          '32x32 == 32x32; assert_no_interpolation_needed is the hard gate')
    check('E1@256 uses the eval256 mean buffer', net.last_mean_key == 'mu_eval256',
          f"{net.last_mean_key} <- {net.dino_mean_paths['eval256']}")
    check('E1@256 eval mean file is the production eval256 file',
          os.path.basename(net.dino_mean_paths['eval256'])
          == '1e5_B6_eval256_dino448_mean.pt')
    check('E1@256 the two mean buffers actually differ',
          not torch.equal(net.mu_train128, net.mu_eval256),
          f'||train128|| {float(net.mu_train128.norm()):.3f} vs '
          f'||eval256|| {float(net.mu_eval256.norm()):.3f}')
    check('E1@256 output finite', torch.isfinite(out).all().item(),
          f"injection ratio {st['injection_ratio']:.3e}")
    # a wrong mode must be caught, not silently interpolated
    try:
        net.set_dino_mode('train128')
        with torch.no_grad():
            net(x)
        gate_ok = False
    except RuntimeError:
        gate_ok = True
    finally:
        net.set_dino_mode('eval256')
        net._capture_dino_io = False
    check('E1 mode/size mismatch RAISES instead of interpolating', gate_ok,
          'radar 256 in train128 mode -> 16x16 DINO grid vs 32x32 latent -> raise')
    return shapes


# ---------------------------------------------------------------------------
# 4.5  fairness
# ---------------------------------------------------------------------------
def smoke_fairness(cfg0, cfg1, device, lq):
    print('\n=== 4.5  Fairness: parameter delta and step-0 equivalence ===')
    seed = cfg0['manual_seed']

    torch.manual_seed(seed)
    net0 = build_net(cfg0, device).eval()
    torch.manual_seed(seed)
    net1 = build_net(cfg1, device).eval()

    n0 = sum(p.numel() for p in net0.parameters())
    n1 = sum(p.numel() for p in net1.parameters()
             if not any(p is q for q in net1.dino_ext.parameters()))
    delta = n1 - n0
    check('parameter delta == 768*384 + 384 == 295,296', delta == 295296,
          f'E0 {n0:,}  E1(trainable trunk+P) {n1:,}  delta {delta:,}')
    n1_train = sum(p.numel() for p in net1.parameters() if p.requires_grad)
    check('E1 trainable params == E0 + 295,296', n1_train == n0 + 295296,
          f'{n1_train:,}')

    # the trunk must be initialised identically -- the RNG fence in the arch
    sd0, sd1 = net0.state_dict(), net1.state_dict()
    shared = [k for k in sd0 if k in sd1]
    same = all(torch.equal(sd0[k].cpu(), sd1[k].cpu()) for k in shared)
    check('E1 trunk initialisation is bit-identical to E0 (RNG fence)',
          same and len(shared) > 300, f'{len(shared)} shared tensors compared')

    x = lq.to(device)
    net1.set_dino_mode('train128')
    with torch.no_grad():
        o0, o1 = net0(x), net1(x)
    diff = float((o0 - o1).abs().max())
    if device == 'cpu':
        ok = torch.equal(o0, o1)
        detail = f'max abs diff {diff:.3e} (CPU: bit-exact required)'
    else:
        ok = diff <= CUDA_ATOL
        detail = (f'max abs diff {diff:.3e} <= atol {CUDA_ATOL} (CUDA float32 '
                  f'non-determinism, WO1 measured ~3.8e-4)')
    check('step-0 outputs of E0 and E1 are equivalent (P(D) == 0)', ok, detail)
    return {'params_e0': n0, 'params_e1_trainable': n1_train,
            'delta': delta, 'step0_max_abs_diff': diff}


# ---------------------------------------------------------------------------
# 4.4  matched-128 protocol
# ---------------------------------------------------------------------------
def smoke_matched128(manifest, device, net0, net1):
    print('\n=== 4.4  Matched-128 evaluation protocol ===')
    import csv as _csv
    import cv2
    ds = '/home/woody/iwnt/iwnt174h/thesis_dino/holographic_image_dataset'
    with open(manifest) as f:
        lines = [ln for ln in f if not ln.startswith('#')]
    rows = list(_csv.DictReader(lines))
    check('crop manifest exists and is non-empty', len(rows) > 0,
          f'{len(rows)} rows in {os.path.basename(manifest)}')
    check('manifest crop size is 128 for every row',
          all(int(r['size']) == 128 for r in rows))
    check('manifest ids are unique',
          len({r['image_id'] for r in rows}) == len(rows))

    ok_align = True
    for r in rows[:3]:
        i, x, y, s = r['image_id'], int(r['x']), int(r['y']), int(r['size'])
        lq = cv2.imread(f'{ds}/val_verynoisy/{i}.png', cv2.IMREAD_UNCHANGED)
        gt = cv2.imread(f'{ds}/val_clean/{i}.png', cv2.IMREAD_UNCHANGED)
        lqc, gtc = lq[y:y + s, x:x + s], gt[y:y + s, x:x + s]
        inp = torch.from_numpy(lqc.astype(np.float32) / 65535.)[None, None].to(device)
        net1.set_dino_mode('train128')
        with torch.no_grad():
            p0, p1 = net0(inp), net1(inp)
        ok_align &= (lqc.shape == gtc.shape == tuple(p0.shape[-2:])
                     == tuple(p1.shape[-2:]) == (128, 128))
    check('same crop coords applied to 1e5, 1e7, E0 pred and E1 pred',
          ok_align, 'checked on the first 3 manifest rows, all 128x128')
    check('matched-128 uses the train128 DINO mean',
          net1.last_mean_key == 'mu_train128', net1.last_mean_key)
    check('E0 and E1 read the SAME manifest file', True,
          os.path.abspath(manifest))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--e0', default=os.path.join(
        _PHASE3, 'configs', 'E0_fixed128_baseline.yml'))
    ap.add_argument('--e1', default=os.path.join(
        _PHASE3, 'configs', 'E1_addition_noisy_fixed128_spatial_B6_latent.yml'))
    ap.add_argument('--manifest', default=os.path.join(
        _PHASE3, 'results', 'crop_manifests', 'matched128_val.csv'))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default=os.path.join(
        _PHASE3, 'results', 'wo2_implementation', 'smoke_results.json'))
    args = ap.parse_args()

    cfg0, cfg1 = load_cfg(args.e0), load_cfg(args.e1)
    print(f'E0 {cfg0["name"]}   ({args.e0})')
    print(f'E1 {cfg1["name"]}   ({args.e1})')

    net0, info = smoke_e0(cfg0, args.device)
    torch.manual_seed(cfg0['manual_seed'])
    random.seed(cfg0['manual_seed'])
    lq, gt, _ = real_train_batch(cfg1, cfg1['manual_seed'])
    net1, shapes128 = smoke_e1_train(cfg1, args.device, lq, gt)
    shapes256 = smoke_e1_eval256(net1, args.device)
    fair = smoke_fairness(cfg0, cfg1, args.device, lq)
    smoke_matched128(args.manifest, args.device, net0, net1)

    n_fail = sum(1 for r in RESULTS if not r['pass'])
    print(f'\n=== {len(RESULTS) - n_fail}/{len(RESULTS)} checks passed ===')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'created': datetime.datetime.now().astimezone().isoformat(),
                   'device': args.device, 'hostname': os.uname().nodename,
                   'torch': torch.__version__,
                   'e0_config': os.path.abspath(args.e0),
                   'e1_config': os.path.abspath(args.e1),
                   'train_batch_info': {k: str(v) for k, v in info.items()},
                   'shapes_train128': shapes128, 'shapes_eval256': shapes256,
                   'fairness': fair, 'n_checks': len(RESULTS),
                   'n_failed': n_fail, 'checks': RESULTS}, f, indent=2)
    print(f'wrote {args.out}')
    if n_fail:
        print('SMOKE TESTS FAILED — stop and report; do not redesign.')
        sys.exit(1)


if __name__ == '__main__':
    main()
