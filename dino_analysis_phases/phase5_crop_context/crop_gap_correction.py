"""Can the crop-versus-full DINO gap be closed by a small learned correction?

Phase 5 (DEVLOG Step 34) measured that DINO describes a 128 crop differently
from the same region inside the full 256 frame, worst at the crop borders, and
that centring does not remove it. This script asks the follow-up question in
feature space, with no training arm: is the gap a SIMPLE transform of the crop
features, one a tiny module could undo before the prior reaches the projection?

THE FREE LABELS. For each training scene:
    full   render 256 -> DINO 448 -> 32x32 tokens        (what evaluation sees)
    crop   token-aligned random 128 crop -> DINO 224 -> 16x16 tokens
                                                          (what training sees)
    target the 16x16 block of the FULL grid at the crop's position
Both are centred with the regime mean the arms use (train128 for the crop,
eval256 for the full frame), so "crop" and "target" are exactly the tensors an
arm receives in its two regimes.

THE CANDIDATES, fit on 80% of the scenes (4,880) and scored on the 1,221 HELD OUT:
    A0  nothing                       the gap as it stands
    A1  per-channel scale + shift     1,536 numbers; is it just statistics?
    A2  linear map + bias             a 768x768 1x1 conv, closed-form ridge
    A2p A2 plus a per-POSITION bias   can a fixed border offset explain it?
    A3  A2, then two 3x3 convs        residual on top of A2; sees neighbours and
                                      the zero-padded edge, so it can treat border
                                      tokens differently

THE SCORES, per candidate, on held-out scenes, split border / centre:
    cos        mean cosine(corrected crop token, true full token)
    rel_l2     ||corrected - true|| / ||true||
    r2         1 - MSE / Var(true)          (fraction of variance explained)
    place_top1 fraction of tokens whose corrected vector is closer to ITS OWN
               full token than to any other position of the same block -- the
               honest check that the correction did not just pull everything
               toward the mean

READING IT. If A2 or A3 closes more than about half the gap on held-out scenes
AND keeps place_top1 high, a one-factor arm (addition-render + the frozen
correction in the train128 regime only) is worth its 39 hours. If not, the
drift is not a simple transform and no arm is built. DEVLOG Step 36's tiling
null already says the mismatch costs little PSNR, so even a good correction may
be a PSNR null; this script decides whether that test is worth running.

Train split only. The fitted A2 map is saved so an arm can load it verbatim.
"""

import argparse
import datetime
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
_P3 = os.path.join(_REPO, 'dino_analysis_phases', 'phase3_restoration')
for p in (_REPO, os.path.join(_P3, 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

import dino_shared                                            # noqa: E402

DATASET = '/home/woody/iwnt/iwnt174h/thesis_dino/holographic_image_dataset'
CROP, TOK = 128, 8                    # 8 radar px per token in both regimes
G_CROP, G_FULL = 16, 32
LAYERS1 = [3, 6, 9, 12]
FIT_LAYER = 6
BORDER = 2                            # ring width, in tokens, counted as border


def load_render(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(path)
    return img[:, :, 0].astype(np.float32) / 255.


def load_mean(name):
    obj = torch.load(os.path.join(_P3, 'means', name), map_location='cpu',
                     weights_only=False)
    return obj['mean'].reshape(-1).float()


def border_mask(g=G_CROP, w=BORDER):
    m = torch.zeros(g, g, dtype=torch.bool)
    m[:w, :] = m[-w:, :] = m[:, :w] = m[:, -w:] = True
    return m


@torch.no_grad()
def tokens(ext, batch01, size, device):
    x = dino_shared.preprocess(batch01.to(device), size)
    b0 = tuple(dino_shared.b1_to_b0(b) for b in LAYERS1)
    feats = ext.dino.get_intermediate_layers(
        x, n=b0, reshape=False, return_class_token=False, norm=True)
    g = size // dino_shared.PATCH_SIZE
    return {b1: f.reshape(f.shape[0], g, g, -1).permute(0, 3, 1, 2)
            for b1, f in zip(sorted(LAYERS1), feats)}       # [B,768,g,g]


# ----------------------------------------------------------------- scoring
def score(pred, true, bmask):
    """pred, true: [N,768,16,16] float32 (CPU ok). Returns dict of metrics."""
    N = pred.shape[0]
    p = pred.permute(0, 2, 3, 1).reshape(N, -1, 768)         # [N,256,768]
    t = true.permute(0, 2, 3, 1).reshape(N, -1, 768)
    bm = bmask.reshape(-1)
    cos = F.cosine_similarity(p, t, dim=-1)                  # [N,256]
    rel = (p - t).norm(dim=-1) / t.norm(dim=-1).clamp_min(1e-6)
    mse = ((p - t) ** 2).sum(-1)                             # [N,256]
    var = ((t - t.mean(dim=(0, 1), keepdim=True)) ** 2).sum(-1)
    # place_top1: within each scene, is the corrected token closest to its own
    # full token among the 256 positions of the block?
    pn = F.normalize(p, dim=-1)
    tn = F.normalize(t, dim=-1)
    top1 = torch.empty(N, p.shape[1])
    for i in range(N):
        sim = pn[i] @ tn[i].T                               # [256,256]
        top1[i] = (sim.argmax(dim=1) == torch.arange(sim.shape[0])).float()
    out = {}
    for name, sel in (('all', torch.ones_like(bm)), ('border', bm),
                      ('centre', ~bm)):
        out[name] = {
            'cos': float(cos[:, sel].mean()),
            'rel_l2': float(rel[:, sel].mean()),
            'r2': float(1 - mse[:, sel].sum() / var[:, sel].sum()),
            'place_top1': float(top1[:, sel].mean()),
        }
    return out


# -------------------------------------------------------------- candidates
def fit_affine(X, Y):
    """Per-channel y = a x + b. X, Y: [M,768]. Streaming sums, no copies."""
    xm, ym = X.mean(0), Y.mean(0)
    sxy = torch.zeros(X.shape[1], dtype=torch.float64)
    sxx = torch.zeros_like(sxy)
    for i in range(0, X.shape[0], 1 << 18):
        xc = (X[i:i + (1 << 18)] - xm).double()
        yc = (Y[i:i + (1 << 18)] - ym).double()
        sxy += (xc * yc).sum(0)
        sxx += (xc * xc).sum(0)
    a = (sxy / sxx.clamp_min(1e-8)).float()
    b = ym - a * xm
    return a, b


def fit_linear(X, Y, lam=1e-3, device='cpu'):
    """Ridge: Y ~ X W + b, closed form. Returns W [768,768], b [768]."""
    xm = X.mean(0).to(device, torch.float64)
    ym = Y.mean(0).to(device, torch.float64)
    D = X.shape[1]
    XtX = torch.zeros(D, D, device=device, dtype=torch.float64)
    XtY = torch.zeros(D, Y.shape[1], device=device, dtype=torch.float64)
    for i in range(0, X.shape[0], 1 << 17):
        xc = X[i:i + (1 << 17)].to(device, torch.float64) - xm
        yc = Y[i:i + (1 << 17)].to(device, torch.float64) - ym
        XtX += xc.T @ xc
        XtY += xc.T @ yc
    reg = lam * XtX.diagonal().mean() * torch.eye(D, device=device,
                                                  dtype=XtX.dtype)
    W = torch.linalg.solve(XtX + reg, XtY)
    b = ym - xm @ W
    return W.float().cpu(), b.float().cpu()


def apply_linear(C, W, b):
    """C [N,768,16,16] -> same shape."""
    N = C.shape[0]
    x = C.permute(0, 2, 3, 1).reshape(-1, 768)
    y = x @ W + b
    return y.reshape(N, G_CROP, G_CROP, 768).permute(0, 3, 1, 2)


class ConvCorrection(nn.Module):
    """Residual two-layer 3x3 conv net. Zero padding makes the edge visible,
    so border tokens can be treated differently from centre tokens."""

    def __init__(self, dim=768, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, dim, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


def fit_conv(Ctr, Ttr, Cva, Tva, device, epochs=30, bs=64, lr=1e-3, log=print):
    net = ConvCorrection().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    N = Ctr.shape[0]
    best, best_state = float('inf'), None
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(N)
        tot = 0.
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            c = Ctr[idx].to(device, torch.float32)
            t = Ttr[idx].to(device, torch.float32)
            loss = F.mse_loss(net(c), t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        net.eval()
        with torch.no_grad():
            va = 0.
            for i in range(0, Cva.shape[0], bs):
                c = Cva[i:i + bs].to(device, torch.float32)
                t = Tva[i:i + bs].to(device, torch.float32)
                va += float(F.mse_loss(net(c), t)) * c.shape[0]
            va /= Cva.shape[0]
        if va < best:
            best, best_state = va, {k: v.detach().cpu().clone()
                                    for k, v in net.state_dict().items()}
        if ep % 5 == 0 or ep == epochs - 1:
            log(f'    conv epoch {ep:3d}  train mse {tot / N:.4f}  held-out mse {va:.4f}')
    net.load_state_dict(best_state)
    return net.eval()


@torch.no_grad()
def apply_conv(net, C, device, bs=64):
    out = []
    for i in range(0, C.shape[0], bs):
        out.append(net(C[i:i + bs].to(device, torch.float32)).cpu())
    return torch.cat(out)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--n-fit', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--outdir', default=os.path.join(
        _HERE, 'results', 'crop_gap_correction'))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    log_lines = []

    def log(s):
        print(s, flush=True)
        log_lines.append(s)

    rdir = os.path.join(DATASET, f'{args.split}_renders_blackbg')
    ids = sorted(os.path.splitext(f)[0] for f in os.listdir(rdir)
                 if f.endswith('.png'))
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    if args.limit:
        ids = ids[:args.limit]
    N = len(ids)
    n_fit = min(args.n_fit, max(1, int(0.8 * N)))
    log(f'{N} scenes; fit on {n_fit}, held out {N - n_fit}; layers {LAYERS1}')

    dev = args.device
    ext = dino_shared.build_dino(device=dev, verbose=True)
    mu_c = {b: load_mean(f'render_B{b}_train128_dino224_mean.pt') for b in LAYERS1}
    mu_f = {b: load_mean(f'render_B{b}_eval256_dino448_mean.pt') for b in LAYERS1}
    bm = border_mask()

    # ---- extraction: raw A0 stats for every layer, stored tensors for B6 ---
    C6 = torch.empty(N, 768, G_CROP, G_CROP, dtype=torch.float16)
    T6 = torch.empty(N, 768, G_CROP, G_CROP, dtype=torch.float16)
    offsets = []
    a0_acc = {b: {'cos': [], 'cos_b': [], 'cos_c': [], 'wrong': []}
              for b in LAYERS1}
    t0 = time.time()
    for s in range(0, N, args.batch):
        chunk = ids[s:s + args.batch]
        fulls, crops, offs = [], [], []
        for i in chunk:
            img = load_render(os.path.join(rdir, f'{i}.png'))
            oy = TOK * rng.randint(0, (img.shape[0] - CROP) // TOK)
            ox = TOK * rng.randint(0, (img.shape[1] - CROP) // TOK)
            fulls.append(img)
            crops.append(img[oy:oy + CROP, ox:ox + CROP])
            offs.append((oy, ox))
        offsets += offs
        tf = tokens(ext, torch.from_numpy(np.stack(fulls))[:, None],
                    dino_shared.DINO_SIZE_EVAL256, dev)
        tc = tokens(ext, torch.from_numpy(np.stack(crops))[:, None],
                    dino_shared.DINO_SIZE_TRAIN128, dev)
        for b in LAYERS1:
            cf = tf[b] - mu_f[b].to(dev).view(1, -1, 1, 1)
            cc = tc[b] - mu_c[b].to(dev).view(1, -1, 1, 1)
            tgt = torch.stack([cf[k, :, oy // TOK:oy // TOK + G_CROP,
                                  ox // TOK:ox // TOK + G_CROP]
                               for k, (oy, ox) in enumerate(offs)])
            cos = F.cosine_similarity(cc, tgt, dim=1)             # [B,16,16]
            a0_acc[b]['cos'].append(cos.mean().item())
            a0_acc[b]['cos_b'].append(cos[:, bm.to(dev)].mean().item())
            a0_acc[b]['cos_c'].append(cos[:, ~bm.to(dev)].mean().item())
            # wrong-place floor: crop token vs the full token one block away
            rolled = torch.roll(tgt, shifts=(G_CROP // 2, G_CROP // 2), dims=(2, 3))
            a0_acc[b]['wrong'].append(
                F.cosine_similarity(cc, rolled, dim=1).mean().item())
            if b == FIT_LAYER:
                C6[s:s + len(chunk)] = cc.half().cpu()
                T6[s:s + len(chunk)] = tgt.half().cpu()
        if (s // args.batch) % 20 == 0:
            log(f'  extracted {min(s + args.batch, N)}/{N}  ({time.time() - t0:.0f}s)')

    log('\n=== A0: the raw gap, all scenes, cosine(crop token, full token at the same place) ===')
    a0 = {}
    for b in LAYERS1:
        a0[b] = {k: float(np.mean(v)) for k, v in a0_acc[b].items()}
        log(f'  B{b:<2d} cos {a0[b]["cos"]:.4f}  border {a0[b]["cos_b"]:.4f}  '
            f'centre {a0[b]["cos_c"]:.4f}  wrong-place floor {a0[b]["wrong"]:.4f}')

    # ---- fits on B6 -----------------------------------------------------
    Ctr, Ttr = C6[:n_fit], T6[:n_fit]
    Cva, Tva = C6[n_fit:].float(), T6[n_fit:].float()
    # built chunk by chunk: a full fp32 copy of Ctr plus its permuted reshape
    # would transiently double the footprint, and the v100 nodes cap RAM per GPU
    Xtr = torch.empty(n_fit * G_CROP * G_CROP, 768)
    Ytr = torch.empty_like(Xtr)
    for i in range(0, n_fit, 256):
        j = min(i + 256, n_fit)
        Xtr[i * 256:j * 256] = Ctr[i:j].float().permute(0, 2, 3, 1).reshape(-1, 768)
        Ytr[i * 256:j * 256] = Ttr[i:j].float().permute(0, 2, 3, 1).reshape(-1, 768)
    log(f'\nfit tensors: {tuple(Xtr.shape)} tokens; held-out scenes {Cva.shape[0]}')

    results = {'A0_nothing': score(Cva, Tva, bm)}
    log('\n=== held-out scores, B6 ===')

    def report(name, r):
        results[name] = r
        log(f'  {name:14s} ' + '  '.join(
            f'{part}: cos {r[part]["cos"]:.4f} rel {r[part]["rel_l2"]:.3f} '
            f'r2 {r[part]["r2"]:+.3f} place {r[part]["place_top1"]:.3f}'
            for part in ('all', 'border', 'centre')))

    report('A0_nothing', results['A0_nothing'])

    a, bb = fit_affine(Xtr, Ytr)
    report('A1_affine', score(Cva * a.view(1, -1, 1, 1) + bb.view(1, -1, 1, 1), Tva, bm))

    W, b = fit_linear(Xtr, Ytr, device=dev)
    del Xtr, Ytr                       # ~3.8 GB each; not needed past this point
    P2 = apply_linear(Cva, W, b)
    report('A2_linear', score(P2, Tva, bm))
    torch.save({'W': W, 'b': b, 'layer': FIT_LAYER, 'n_fit': n_fit,
                'note': 'y = x @ W + b on train128-centred B6 crop tokens -> '
                        'eval256-centred full-frame tokens'},
               os.path.join(args.outdir, 'A2_linear_B6.pt'))

    # per-position bias on top of A2: mean residual at each of the 256 places
    resid = torch.zeros(1, 768, G_CROP, G_CROP)
    for i in range(0, n_fit, 256):
        c = Ctr[i:i + 256].float()
        resid += (Ttr[i:i + 256].float() - apply_linear(c, W, b)).sum(dim=0, keepdim=True)
    resid /= n_fit
    report('A2p_lin+pos', score(P2 + resid, Tva, bm))

    # A3 sits ON TOP of the fitted linear map, so it can only add to A2: the
    # conv sees A2's output and learns the residual, including anything
    # position-dependent that a 1x1 map cannot express.
    log('\n  fitting A3 (A2 linear map, then two 3x3 convs, residual)...')
    Ctr2 = torch.empty_like(Ctr)
    for i in range(0, n_fit, 256):
        Ctr2[i:i + 256] = apply_linear(Ctr[i:i + 256].float(), W, b).half()
    net = fit_conv(Ctr2, Ttr, P2, Tva, dev, epochs=args.epochs, log=log)
    report('A3_lin+conv3x3', score(apply_conv(net, P2, dev), Tva, bm))
    torch.save(net.state_dict(), os.path.join(args.outdir, 'A3_conv_B6.pt'))

    # how much of the gap is closed, in cosine, all / border
    base = results['A0_nothing']
    log('\n=== gap closed (cosine, held-out; 1.0 would mean identical to full) ===')
    for name in ('A1_affine', 'A2_linear', 'A2p_lin+pos', 'A3_lin+conv3x3'):
        r = results[name]
        closed = {part: (r[part]['cos'] - base[part]['cos']) / max(1e-6, 1 - base[part]['cos'])
                  for part in ('all', 'border', 'centre')}
        results[name]['gap_closed_cos'] = closed
        log(f'  {name:14s} all {closed["all"]:.1%}  border {closed["border"]:.1%}  '
            f'centre {closed["centre"]:.1%}')

    with open(os.path.join(args.outdir, 'crop_gap_correction.json'), 'w') as f:
        json.dump({'created': datetime.datetime.now().astimezone().isoformat(),
                   'split': args.split, 'n_scenes': N, 'n_fit': n_fit,
                   'n_heldout': N - n_fit, 'seed': args.seed,
                   'crop': CROP, 'token_px': TOK, 'border_ring': BORDER,
                   'fit_layer': FIT_LAYER, 'epochs_conv': args.epochs,
                   'A0_all_layers': {f'B{b}': v for b, v in a0.items()},
                   'heldout_B6': results, 'log': log_lines}, f, indent=2)
    log(f'\nwrote {args.outdir}')


if __name__ == '__main__':
    main()
