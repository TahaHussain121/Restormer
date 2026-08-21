"""crossattn-render diagnosis, Task B: four inference-only interventions.

READ-ONLY and INFERENCE-ONLY. No training, no config edit, no architecture
edit, no experiment directory written. Checkpoints are opened for reading;
everything else lands under phase4_crossattn_diagnosis/outputs/.

PROTOCOL — deliberately the repo's own matched-128 one, so the numbers can be
checked against evaluations that already exist:

  split          val (339 images)
  crops          results/crop_manifests/matched128_val.csv, the SAME box for
                 radar, render and target, read from disk, seed fixed upstream
  DINO regime    train128 (explicit set_dino_mode, never inferred)
  PSNR           predictions quantised to uint16 exactly as predict_phase3.py
                 does, then skimage PSNR at data_range=1.0 on float64/65535 --
                 the same path masked_metrics.py uses. Masked PSNR uses the
                 same GT>0.01 rule, no dilation, no closing.

SELF-CHECK FIRST. Two conditions in the table are reproductions of numbers the
evaluation chain already published: E0 at 268k and crossattn at 4k on
crop128/val. If this script does not reproduce 19.816 and 18.284, the pipeline
differs from the one that produced every other number in the project and the
table is not usable. It prints PASS/FAIL and says so.

THE INTERVENTIONS, all applied inside the attention module only:

  B1 wrong render   a deterministic derangement supplies a DIFFERENT scene's
                    render while the radar input stays this scene's. Reports how
                    often the argmax key moves, per head.
  B2 temperature    logits / T, T in {1,2,4,8,16,inf}. T=inf is exactly uniform.
  B3 identity       the attention matrix is replaced by I.
  B4 head ablation  named heads have their context zeroed before W_o.

The intervened forward is asserted against the module's own forward under the
null intervention before any number is reported.
"""

import argparse
import csv
import json
import math
import os
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE4 = os.path.dirname(_HERE)
_REPO = os.path.dirname(os.path.dirname(_PHASE4))
_P3 = os.path.join(_REPO, 'dino_analysis_phases', 'phase3_restoration')
for p in (_REPO, os.path.join(_P3, 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault('TORCH_HOME', os.path.join(_REPO, 'torch_hub'))

from basicsr.models.archs import define_network                   # noqa: E402

OUT = os.path.join(_PHASE4, 'outputs')
DATASET = '/home/woody/iwnt/iwnt174h/thesis_dino/holographic_image_dataset'
MANIFEST = os.path.join(_P3, 'results', 'crop_manifests', 'matched128_val.csv')
CFG_XAT = os.path.join(_P3, 'configs',
                       'crossattn_render_fixed128_spatial_B6_latent.yml')
CFG_E0 = os.path.join(_P3, 'configs', 'E0_fixed128_baseline.yml')
CKPT_XAT_LAST = os.path.join(_REPO, 'experiments',
                             'Holo_crossattn_render_fixed128_spatial_B6_latent',
                             'models', 'net_g_178000.pth')
CKPT_XAT_BEST = CKPT_XAT_LAST.replace('178000', '4000')
CKPT_E0 = os.path.join(_REPO, 'experiments', 'Holo_E0_fixed128_baseline',
                       'models', 'net_g_268000.pth')
# published crop128/val figures these two conditions must reproduce
REF_E0, REF_XAT4K = 19.816, 18.284
GRID, HEADS = 16, 6


# ------------------------------------------------------------------ data ----
def read_manifest(path):
    with open(path) as f:
        lines = [l for l in f if not l.startswith('#')]
    return [(r['image_id'], int(r['x']), int(r['y']), int(r['size']))
            for r in csv.DictReader(lines)]


def load_uint16(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(path)
    return img.squeeze()


def load_render01(path):
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise IOError(path)
    # channel 0 of the 3 identical channels, exactly as the stacked dataset does
    return (raw[:, :, 0] if raw.ndim == 3 else raw).astype(np.float32) / 255.


def crops(split, rows):
    """-> list of (image_id, radar01 [128,128], render01, gt_uint16)."""
    out = []
    for image_id, x, y, size in rows:
        lq = load_uint16(os.path.join(DATASET, f'{split}_verynoisy',
                                      f'{image_id}.png'))
        gt = load_uint16(os.path.join(DATASET, f'{split}_clean',
                                      f'{image_id}.png'))
        rn = load_render01(os.path.join(DATASET, f'{split}_renders_blackbg',
                                        f'{image_id}.png'))
        out.append((image_id,
                    lq[y:y + size, x:x + size].astype(np.float32) / 65535.,
                    rn[y:y + size, x:x + size],
                    gt[y:y + size, x:x + size]))
    return out


# ---------------------------------------------------------------- metrics ---
def score(pred_u16, gt_u16):
    pr = pred_u16.astype(np.float64) / 65535.
    gt = gt_u16.astype(np.float64) / 65535.
    mask = gt > 0.01
    mse = np.mean((gt[mask] - pr[mask]) ** 2) if mask.any() else np.nan
    return dict(
        psnr_full=float(psnr_fn(gt, pr, data_range=1.0)),
        psnr_mask=float(10 * math.log10(1.0 / mse)) if mse > 0 else float('nan'),
        ssim_full=float(ssim_fn(gt, pr, data_range=1.0)))


def to_uint16(out):
    return (out.clamp(0, 1)[:, 0].float().cpu().numpy() * 65535.
            ).round().astype(np.uint16)


# ------------------------------------------------------------ the module ----
def patched_forward(m):
    """A forward for DinoCrossAttention that mirrors the module's own maths and
    adds the interventions. Verified against the original under the null."""

    def fwd(F_grid, D_grid, collect_attn_stats=False):
        b, cq, h, w = F_grid.shape
        n = h * w
        q_tok = F_grid.flatten(2).transpose(1, 2)
        kv_tok = D_grid.flatten(2).transpose(1, 2)
        q = m.W_q(m.norm_q(q_tok))
        kv = m.norm_kv(kv_tok)
        k, v = m.W_k(kv), m.W_v(kv)

        def split(t):
            return t.view(b, -1, m.heads, m.head_dim).transpose(1, 2)

        qh, kh, vh = split(q), split(k), split(v)
        logits = (qh @ kh.transpose(-2, -1)) * m.scale

        T = getattr(m, '_probe_T', 1.0)
        if getattr(m, '_probe_identity', False):
            attn = torch.eye(n, device=logits.device, dtype=logits.dtype)
            attn = attn.view(1, 1, n, n).expand(b, m.heads, n, n)
        elif T == float('inf'):
            attn = torch.full_like(logits, 1.0 / n)
        else:
            attn = (logits / T).softmax(dim=-1)

        m._probe_last_attn = attn.detach()
        ctx = attn @ vh
        zero = getattr(m, '_probe_zero_heads', ())
        if zero:
            ctx = ctx.clone()
            ctx[:, list(zero)] = 0.0
        injected = m.W_o(ctx.transpose(1, 2).reshape(b, n, cq))
        return injected.transpose(1, 2).view(b, cq, h, w)

    return fwd


def build(cfg_path, weights, device):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    net = define_network(dict(cfg['network_g']))
    ck = torch.load(weights, map_location='cpu', weights_only=False)
    net.load_state_dict(ck.get('params', ck), strict=True)
    net = net.to(device).eval()
    if hasattr(net, 'set_dino_mode'):
        net.set_dino_mode('train128')
    return net, cfg


def set_probe(net, T=1.0, identity=False, zero_heads=()):
    m = net.xattn
    m._probe_T, m._probe_identity, m._probe_zero_heads = T, identity, tuple(zero_heads)


@torch.no_grad()
def run(net, data, device, batch=8, render_from=None, want_argmax=False):
    """-> (per-image metric dicts, argmax [N,heads,256] or None)."""
    rows, argmax = [], []
    for i in range(0, len(data), batch):
        chunk = data[i:i + batch]
        radar = np.stack([c[1] for c in chunk])
        if render_from is None:
            render = np.stack([c[2] for c in chunk])
        else:
            render = np.stack([render_from[c[0]] for c in chunk])
        x = torch.from_numpy(np.stack([radar, render], 1)).to(device)
        # E0 is plain Restormer (1 channel) and has no inp_stack_channels at
        # all; only the render arms take the stacked [B,2,H,W] tensor.
        if getattr(net, 'inp_stack_channels', 1) == 1:
            x = x[:, :1]
        out = net(x)
        pred = to_uint16(out)
        for j, c in enumerate(chunk):
            rows.append(dict(image_id=c[0], **score(pred[j], c[3])))
        if want_argmax:
            argmax.append(net.xattn._probe_last_attn.argmax(-1).cpu())
    return rows, (torch.cat(argmax) if want_argmax else None)


def mean_of(rows, key):
    return float(np.mean([r[key] for r in rows]))


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--split', default='val')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--batch', type=int, default=8)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)

    rows = read_manifest(MANIFEST)
    if args.limit:
        rows = rows[:args.limit]
    data = crops(args.split, rows)
    print(f'{args.split}: {len(data)} matched-128 crops from '
          f'{os.path.basename(MANIFEST)}')

    # the different-scene mapping: a cyclic shift by one over the manifest
    # order. No fixed points, deterministic, and it reuses the crop box of the
    # image whose render is borrowed -- the render is the only thing swapped.
    ids = [c[0] for c in data]
    wrong = {ids[i]: data[(i + 1) % len(data)][2] for i in range(len(data))}
    print(f'different-scene control: cyclic shift by 1, '
          f'{ids[0]} <- render of {ids[1]}, no fixed points')

    table, extra = [], {}

    # ---- reference rows and the self-check -------------------------------
    net_e0, _ = build(CFG_E0, CKPT_E0, args.device)
    r, _ = run(net_e0, data, args.device, args.batch)
    e0_psnr = mean_of(r, 'psnr_full')
    table.append(('E0-Fixed 268k (no DINO)', e0_psnr, mean_of(r, 'psnr_mask'),
                  mean_of(r, 'ssim_full')))
    del net_e0

    net4, _ = build(CFG_XAT, CKPT_XAT_BEST, args.device)
    net4.xattn.forward = patched_forward(net4.xattn)
    set_probe(net4)
    r, _ = run(net4, data, args.device, args.batch)
    x4_psnr = mean_of(r, 'psnr_full')
    table.append(('crossattn 4k (best val)', x4_psnr, mean_of(r, 'psnr_mask'),
                  mean_of(r, 'ssim_full')))
    del net4

    print('\nSELF-CHECK against the published crop128/val evaluation:')
    ok = True
    for name, got, ref in (('E0 268k', e0_psnr, REF_E0),
                           ('crossattn 4k', x4_psnr, REF_XAT4K)):
        d = got - ref
        good = abs(d) < 0.02
        ok &= good
        print(f'  {name:<16} got {got:.4f}  published {ref:.4f}  '
              f'delta {d:+.4f}  -> {"PASS" if good else "FAIL"}')
    if args.limit:
        print('  (--limit set: this is a SUBSET, so the published full-split '
              'numbers are not the right comparison. Smoke test only.)')
    elif not ok:
        raise SystemExit('pipeline does not reproduce the published numbers; '
                         'not reporting intervention results')

    # ---- the intervened checkpoint ---------------------------------------
    net, _ = build(CFG_XAT, CKPT_XAT_LAST, args.device)
    orig_forward = net.xattn.forward
    probe_fwd = patched_forward(net.xattn)

    # null-intervention equivalence, on one batch, before anything is reported
    with torch.no_grad():
        chunk = data[:args.batch]
        x = torch.from_numpy(np.stack(
            [np.stack([c[1] for c in chunk]),
             np.stack([c[2] for c in chunk])], 1)).to(args.device)
        cap = {}
        hk = net.xattn.register_forward_pre_hook(
            lambda mod, a: cap.update(F=a[0], D=a[1]))
        net(x)
        hk.remove()
        set_probe(net)
        a = orig_forward(cap['F'], cap['D'])
        b = probe_fwd(cap['F'], cap['D'])
        err = float((a - b).abs().max())
        scale = float(a.abs().max())
        rel = err / scale
    # The original takes F.scaled_dot_product_attention on the no-stats path
    # while the probe writes the softmax out explicitly -- mathematically the
    # same, bitwise not, so the tolerance is RELATIVE to the tensor's own scale.
    print(f'\nnull-intervention equivalence: max|orig - patched| = {err:.3e} '
          f'on a tensor of max |value| {scale:.1f}  ->  relative {rel:.2e}  '
          f'{"PASS" if rel < 1e-5 else "FAIL"}')
    if rel >= 1e-5:
        raise SystemExit('patched forward differs from the module; aborting')
    net.xattn.forward = probe_fwd

    set_probe(net)
    ref_rows, am_right = run(net, data, args.device, args.batch, want_argmax=True)
    ref_psnr = mean_of(ref_rows, 'psnr_full')
    table.append(('crossattn 178k UNMODIFIED', ref_psnr,
                  mean_of(ref_rows, 'psnr_mask'), mean_of(ref_rows, 'ssim_full')))

    # ---- B1 ---------------------------------------------------------------
    set_probe(net)
    r1, am_wrong = run(net, data, args.device, args.batch, render_from=wrong,
                       want_argmax=True)
    table.append(('B1  wrong-scene render', mean_of(r1, 'psnr_full'),
                  mean_of(r1, 'psnr_mask'), mean_of(r1, 'ssim_full')))
    changed = (am_right != am_wrong).float()                     # [N,heads,256]
    per_head = changed.mean(dim=(0, 2)).numpy()
    overall = float(changed.mean())
    extra['B1'] = {'per_head_change': per_head.tolist(), 'overall': overall}
    print('\nB1 — fraction of queries whose argmax key MOVES when the render '
          'is from another scene')
    print(f'  {"head":>6}{"changed":>10}{"diag_mass(178k)":>18}')
    diag_ref = [0.017, 0.097, 0.471, 0.081, 0.933, 0.008]
    for h in range(HEADS):
        print(f'  {h:>6}{per_head[h]:>10.3f}{diag_ref[h]:>18.3f}')
    print(f'  {"all":>6}{overall:>10.3f}')

    # ---- B2 ---------------------------------------------------------------
    print('\nB2 — temperature sweep')
    for T in (2.0, 4.0, 8.0, 16.0, float('inf')):
        set_probe(net, T=T)
        r, _ = run(net, data, args.device, args.batch)
        with torch.no_grad():
            a = net.xattn._probe_last_attn
            p = a.clamp_min(1e-12)
            ent = float((-(p * p.log()).sum(-1)).mean())
            dg = float(a.diagonal(dim1=-2, dim2=-1).mean())
        label = 'T=inf (uniform)' if T == float('inf') else f'T={T:g}'
        table.append((f'B2  {label}', mean_of(r, 'psnr_full'),
                      mean_of(r, 'psnr_mask'), mean_of(r, 'ssim_full')))
        extra.setdefault('B2', {})[label] = {'entropy': ent, 'diag_mass': dg}
        print(f'  {label:<16} psnr {mean_of(r, "psnr_full"):7.3f}   '
              f'entropy {ent:6.3f}   diag_mass {dg:.4f}')

    # ---- B3 ---------------------------------------------------------------
    set_probe(net, identity=True)
    r, _ = run(net, data, args.device, args.batch)
    table.append(('B3  attn = identity', mean_of(r, 'psnr_full'),
                  mean_of(r, 'psnr_mask'), mean_of(r, 'ssim_full')))

    # ---- B4 ---------------------------------------------------------------
    for zh in ((0, 5), (0, 1, 3, 5)):
        set_probe(net, zero_heads=zh)
        r, _ = run(net, data, args.device, args.batch)
        table.append((f'B4  zero heads {list(zh)}', mean_of(r, 'psnr_full'),
                      mean_of(r, 'psnr_mask'), mean_of(r, 'ssim_full')))

    # ------------------------------------------------------------ report ---
    print('\n' + '=' * 88)
    print(f'{"condition":<34}{"PSNR":>9}{"masked":>9}{"SSIM":>8}'
          f'{"vs 178k":>10}{"vs E0":>9}')
    print('=' * 88)
    for name, ps, pm, ss in table:
        print(f'{name:<34}{ps:>9.3f}{pm:>9.3f}{ss:>8.4f}'
              f'{ps - ref_psnr:>+10.3f}{ps - e0_psnr:>+9.3f}')
    print('=' * 88)
    print(f'n={len(data)}, crop128/val, uint16 PSNR (data_range 1.0). '
          f'"vs 178k" is against the unmodified checkpoint.')

    with open(os.path.join(OUT, 'interventions.json'), 'w') as f:
        json.dump({'n': len(data), 'protocol': 'crop128/val',
                   'table': [{'condition': n, 'psnr_full': a, 'psnr_mask': b,
                              'ssim_full': c} for n, a, b, c in table],
                   'e0_reference': e0_psnr, 'unmodified_178k': ref_psnr,
                   **extra}, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.6))
    ax = axes[0]
    names = [t[0] for t in table]
    vals = [t[1] for t in table]
    colors = ['#4c4c4c', '#888888', '#c44e52'] + \
             ['#2b7bba'] + ['#dd8452'] * 5 + ['#2ca02c'] + ['#9467bd'] * 2
    ax.barh(range(len(vals)), vals, color=colors[:len(vals)])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9.5)
    ax.invert_yaxis()
    ax.axvline(e0_psnr, color='#4c4c4c', ls='--', lw=1.8,
               label=f'E0 = {e0_psnr:.2f}')
    ax.axvline(ref_psnr, color='#c44e52', ls='--', lw=1.8,
               label=f'unmodified 178k = {ref_psnr:.2f}')
    ax.set_xlim(min(vals) - 1, max(e0_psnr, max(vals)) + 0.6)
    ax.set_xlabel('val PSNR (dB), crop128')
    ax.set_title('Every intervention, against the two reference rows',
                 fontsize=12.5)
    ax.legend(fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    ax = axes[1]
    ax.bar(range(HEADS), per_head, color='#c44e52')
    ax.axhline(overall, color='black', ls='--', lw=1.8,
               label=f'overall {overall:.3f}')
    for h in range(HEADS):
        ax.text(h, per_head[h] + 0.02, f'diag\n{diag_ref[h]:.2f}',
                ha='center', fontsize=9, color='#2b7bba')
    ax.set_xticks(range(HEADS))
    ax.set_xticklabels([f'h{h}' for h in range(HEADS)])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('fraction of queries whose argmax key moves')
    ax.set_title('B1 — does the routing track render CONTENT?\n'
                 '(blue = that head\'s diagonal mass)', fontsize=12.5)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    ax = axes[2]
    b2 = [('T=1', ref_psnr)] + [(k.split()[0], v) for k, v in
                                [(t[0].replace('B2  ', ''), t[1])
                                 for t in table if t[0].startswith('B2')]]
    ax.plot(range(len(b2)), [v for _, v in b2], 'o-', color='#dd8452', lw=2.2,
            ms=8)
    ax.axhline(e0_psnr, color='#4c4c4c', ls='--', lw=1.8, label='E0')
    ax.axhline(ref_psnr, color='#c44e52', ls=':', lw=1.8, label='unmodified')
    ax.set_xticks(range(len(b2)))
    ax.set_xticklabels([k for k, _ in b2])
    ax.set_ylabel('val PSNR (dB)')
    ax.set_title('B2 — re-softening the collapsed attention at inference',
                 fontsize=12.5)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle('crossattn-render — four inference-only interventions on '
                 f'net_g_178000 (crop128/val, n={len(data)})',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(OUT, 'B_interventions.png')
    fig.savefig(p, dpi=125)
    plt.close(fig)
    print('\nwrote', p)
    print('wrote', os.path.join(OUT, 'interventions.json'))


if __name__ == '__main__':
    main()
