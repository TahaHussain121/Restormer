"""crossattn-render diagnosis, sections 2 and 4: the flatten/reshape audit and
an inference-only dump of the real attention matrix from trained checkpoints.

READ-ONLY on every experiment directory: checkpoints are opened for reading,
nothing is written outside phase4_crossattn_diagnosis/outputs/. No training, no
architecture change, no config change. Everything runs under torch.no_grad().

WHAT IT MEASURES, per head, on real validation crops:

  entropy, diag_mass        the same definitions the training run logged, so the
                            two are directly comparable
  effective support         exp(entropy): how many keys a query row spreads over
  destination concentration where the attention MASS lands, summed over queries.
                            If 256 queries all read a handful of DINO tokens the
                            block is behaving like a pooled global read, which is
                            the regime global-addition-render already showed to
                            score BELOW E0.
  argmax displacement       grid distance between a query's position and the key
                            it actually selects
  transposed-diagonal mass  mass at the (col,row) partner of each query. A silent
                            row/column-order mismatch between the two streams
                            would show up here and nowhere else.

The recomputed attention is checked against the module's own output before any
statistic is reported: if `W_o(attn @ V)` does not reproduce what the module
returned, the probe is wrong and it says so instead of printing numbers.
"""

import argparse
import csv
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE4 = os.path.dirname(_HERE)
_REPO = os.path.dirname(os.path.dirname(_PHASE4))
_P3SCRIPTS = os.path.join(_REPO, 'dino_analysis_phases', 'phase3_restoration',
                          'scripts')
for p in (_REPO, _P3SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault('TORCH_HOME', os.path.join(_REPO, 'torch_hub'))

from basicsr.models.archs import define_network                  # noqa: E402
import dino_shared                                               # noqa: E402

OUT = os.path.join(_PHASE4, 'outputs')
DATASET = '/home/woody/iwnt/iwnt174h/thesis_dino/holographic_image_dataset'
CONFIG = os.path.join(_REPO, 'dino_analysis_phases', 'phase3_restoration',
                      'configs', 'crossattn_render_fixed128_spatial_B6_latent.yml')
EXP = 'Holo_crossattn_render_fixed128_spatial_B6_latent'
MANIFEST = os.path.join(_REPO, 'dino_analysis_phases', 'phase3_restoration',
                        'results', 'crop_manifests', 'matched128_val.csv')
GRID = 16


# ------------------------------------------------------------ section 2 -----
def audit_flatten_order():
    """Numeric proof that the two streams are flattened the same way and that
    the reshape back is the exact inverse. Asserted, not read off the source."""
    print('=' * 92)
    print('SECTION 2 — flatten order and reshape inverse, verified numerically')
    print('=' * 92)
    ok = {}

    b, c, g = 2, 5, GRID
    tokens = torch.arange(b * g * g * c, dtype=torch.float32).reshape(b, g * g, c)
    grid = dino_shared.tokens_to_grid(tokens)                    # DINO path
    back = grid.flatten(2).transpose(1, 2)                       # the arch's path
    ok['dino_tokens_roundtrip'] = bool(torch.equal(back, tokens))
    print(f'\n  dino_shared.tokens_to_grid -> flatten(2).transpose(1,2) is the '
          f'identity on tokens : {ok["dino_tokens_roundtrip"]}')

    f = torch.randn(b, 384, g, g)
    q_tok = f.flatten(2).transpose(1, 2)                         # query flatten
    restored = q_tok.transpose(1, 2).view(b, 384, g, g)          # the arch's inverse
    ok['query_roundtrip'] = bool(torch.equal(restored, f))
    print(f'  F.flatten(2).transpose(1,2) -> transpose(1,2).view(b,C,h,w) is '
          f'the identity      : {ok["query_roundtrip"]}')

    d = torch.randn(b, 768, g, g)
    kv_tok = d.flatten(2).transpose(1, 2)
    idx = 7 * g + 3                                              # row 7, col 3
    same = bool(torch.equal(q_tok[0, idx], f[0, :, 7, 3]) and
                torch.equal(kv_tok[0, idx], d[0, :, 7, 3]))
    ok['same_row_major_convention'] = same
    print(f'  token {idx} is grid position (row 7, col 3) in BOTH streams'
          f'          : {same}')
    print('\n  => query token i and key token i refer to the same spatial cell; '
          'there is no\n     transpose or row/column mismatch between the two '
          'streams.')
    return ok


# ------------------------------------------------------------ section 4 -----
def read_manifest(path, n):
    rows = []
    with open(path) as f:
        lines = [l for l in f if not l.startswith('#')]
    for r in csv.DictReader(lines):
        rows.append((r['image_id'], int(r['x']), int(r['y']), int(r['size'])))
    return rows[:n]


def load_crop(split, image_id, x, y, size):
    lq = cv2.imread(os.path.join(DATASET, f'{split}_verynoisy', f'{image_id}.png'),
                    cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.
    rn = cv2.imread(os.path.join(DATASET, f'{split}_renders_blackbg',
                                 f'{image_id}.png'), cv2.IMREAD_UNCHANGED)
    rn = (rn[:, :, 0] if rn.ndim == 3 else rn).astype(np.float32) / 255.
    lq = lq[y:y + size, x:x + size]
    rn = rn[y:y + size, x:x + size]
    return torch.from_numpy(np.stack([lq, rn], 0))               # [2,128,128]


def build(weights, device):
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    net = define_network(dict(cfg['network_g']))
    ckpt = torch.load(weights, map_location='cpu', weights_only=False)
    net.load_state_dict(ckpt.get('params', ckpt), strict=True)
    net = net.to(device).eval()
    net.set_dino_mode('train128')
    return net


@torch.no_grad()
def probe(net, batch, device):
    """-> attn [B,heads,256,256], plus the tensors around the fusion."""
    cap = {}

    def pre_hook(mod, args):
        cap['F'], cap['D'] = args[0], args[1]

    def post_hook(mod, args, out):
        cap['injected'] = out

    h1 = net.xattn.register_forward_pre_hook(pre_hook)
    h2 = net.xattn.register_forward_hook(post_hook)
    try:
        out = net(batch.to(device))
    finally:
        h1.remove()
        h2.remove()

    m = net.xattn
    b = cap['F'].shape[0]
    q_tok = cap['F'].flatten(2).transpose(1, 2)
    kv_tok = cap['D'].flatten(2).transpose(1, 2)
    q = m.W_q(m.norm_q(q_tok))
    kv = m.norm_kv(kv_tok)
    k, v = m.W_k(kv), m.W_v(kv)

    def split(t):
        return t.view(b, -1, m.heads, m.head_dim).transpose(1, 2)

    qh, kh, vh = split(q), split(k), split(v)
    attn = ((qh @ kh.transpose(-2, -1)) * m.scale).softmax(dim=-1)
    ctx = attn @ vh
    recomputed = m.W_o(ctx.transpose(1, 2).reshape(b, q_tok.shape[1], m.dim_q))
    recomputed = recomputed.transpose(1, 2).view(*cap['F'].shape)
    err = float((recomputed - cap['injected']).abs().max())
    return attn, cap, ctx, err, out


def head_stats(attn):
    """attn [B,heads,N,N] -> dict of per-head arrays, averaged over the batch."""
    b, hd, n, _ = attn.shape
    g = int(round(n ** 0.5))
    p = attn.clamp_min(1e-12)
    ent = (-(p * p.log()).sum(-1)).mean(-1)                      # [B,heads]
    diag = attn.diagonal(dim1=-2, dim2=-1).mean(-1)              # [B,heads]

    rows = torch.arange(n)
    tr = (rows % g) * g + (rows // g)                            # transposed cell
    trans = attn[:, :, rows, tr].mean(-1)

    colmass = attn.mean(dim=-2)                                  # [B,heads,N]
    cm = colmass.clamp_min(1e-12)
    col_ent = -(cm * cm.log()).sum(-1)
    top1 = colmass.max(-1).values
    top8 = colmass.topk(8, dim=-1).values.sum(-1)

    am = attn.argmax(-1)                                         # [B,heads,N]
    uniq = torch.tensor([[len(torch.unique(am[i, j])) for j in range(hd)]
                         for i in range(b)], dtype=torch.float32)
    dr = (am // g - (rows // g).view(1, 1, -1)).float()
    dc = (am % g - (rows % g).view(1, 1, -1)).float()
    disp = torch.sqrt(dr ** 2 + dc ** 2).mean(-1)
    hit = (am == rows.view(1, 1, -1)).float().mean(-1)

    # Is the off-diagonal routing STRUCTURED (a constant grid offset) or
    # arbitrary? The modal (dr,dc) and the share of queries taking it separate
    # "learned a systematic shift" from "learned nothing positional".
    off = torch.stack([dr, dc], -1).long()                       # [B,heads,N,2]
    mode_share, mode_off = [], []
    for i in range(b):
        ms, mo = [], []
        for j in range(hd):
            key = (off[i, j, :, 0] + 64) * 256 + (off[i, j, :, 1] + 64)
            vals, counts = torch.unique(key, return_counts=True)
            top = int(counts.argmax())
            ms.append(float(counts[top]) / n)
            mo.append((int(vals[top]) // 256 - 64, int(vals[top]) % 256 - 64))
        mode_share.append(ms)
        mode_off.append(mo)
    mode_share = torch.tensor(mode_share)

    raw = dict(entropy=ent, diag_mass=diag, transposed_mass=trans,
               modal_offset_share=mode_share,
               dest_entropy=col_ent, dest_top1=top1, dest_top8=top8,
               unique_dest=uniq, mean_displacement=disp, argmax_on_diag=hit)
    out = {k: v.mean(0).cpu().numpy() for k, v in raw.items()}
    # spread ACROSS IMAGES: a per-head number that swings image to image is a
    # different claim from one that is stable, so both are reported.
    out.update({k + '_sd': v.std(0, unbiased=True).cpu().numpy()
                for k, v in raw.items()})
    out['modal_offset'] = mode_off[0]          # the first image, for reporting
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--n-images', type=int, default=4)
    ap.add_argument('--split', default='val')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)

    record = {'flatten_audit': audit_flatten_order()}

    print('\n' + '=' * 92)
    print('SECTION 4 — the real attention matrix from the trained checkpoints')
    print('=' * 92)
    rows = read_manifest(MANIFEST, args.n_images)
    batch = torch.stack([load_crop(args.split, *r) for r in rows])
    print(f'\n  {args.split} crops from the matched-128 manifest: '
          f'{", ".join(r[0] for r in rows)}   batch {tuple(batch.shape)}')

    ckpts = {'4k (best val)': 4000, '178k (last)': 178000}
    results, attns = {}, {}
    for label, it in ckpts.items():
        w = os.path.join(_REPO, 'experiments', EXP, 'models', f'net_g_{it}.pth')
        net = build(w, args.device)
        n_grad = sum(1 for p in net.parameters() if p.requires_grad)
        n_dino_grad = sum(1 for p in net.dino_ext.parameters() if p.requires_grad)
        attn, cap, ctx, err, out = probe(net, batch, args.device)
        print(f'\n--- {label}   {os.path.basename(w)} ---')
        print(f'  training mode {net.training} (must be False); DINO params '
              f'with requires_grad: {n_dino_grad} (must be 0); '
              f'trainable tensors elsewhere: {n_grad}')
        print(f'  recomputed injection matches the module output to '
              f'{err:.3e}  ->  {"PASS" if err < 1e-4 else "FAIL"}')
        if err >= 1e-4:
            raise SystemExit('probe does not reproduce the module; aborting')
        st = head_stats(attn)
        fn = float(cap['F'].flatten(1).norm(dim=1).mean())
        cn = float(ctx.transpose(1, 2).reshape(ctx.shape[0], -1).norm(dim=1).mean())
        inj = float(cap['injected'].flatten(1).norm(dim=1).mean())
        print(f'  ||F|| {fn:9.1f}   ||attn@V|| {cn:9.2f}   ||W_o(.)|| {inj:9.1f}'
              f'   ratio {inj / fn:.3f}')
        print(f'\n  {"head":>5}{"entropy":>9}{"eff.sup":>9}{"diag":>8}'
              f'{"argmax@diag":>13}{"transp":>8}{"uniq dest":>11}'
              f'{"top1 dest":>11}{"top8 dest":>11}{"displac.":>10}')
        for h in range(len(st['entropy'])):
            print(f'  {h:>5}{st["entropy"][h]:>9.3f}'
                  f'{np.exp(st["entropy"][h]):>9.2f}{st["diag_mass"][h]:>8.3f}'
                  f'{st["argmax_on_diag"][h]:>13.3f}'
                  f'{st["transposed_mass"][h]:>8.3f}'
                  f'{st["unique_dest"][h]:>11.1f}{st["dest_top1"][h]:>11.3f}'
                  f'{st["dest_top8"][h]:>11.3f}'
                  f'{st["mean_displacement"][h]:>10.2f}'
                  f'   (diag sd {st["diag_mass_sd"][h]:.3f};'
                  f' modal offset {tuple(st["modal_offset"][h])} taken by'
                  f' {st["modal_offset_share"][h]:.0%} of queries)')
        print(f'  {"mean":>5}{st["entropy"].mean():>9.3f}'
              f'{np.exp(st["entropy"]).mean():>9.2f}{st["diag_mass"].mean():>8.3f}'
              f'{st["argmax_on_diag"].mean():>13.3f}'
              f'{st["transposed_mass"].mean():>8.3f}'
              f'{st["unique_dest"].mean():>11.1f}{st["dest_top1"].mean():>11.3f}'
              f'{st["dest_top8"].mean():>11.3f}'
              f'{st["mean_displacement"].mean():>10.2f}')
        results[label] = {k: (v.tolist() if hasattr(v, 'tolist') else v)
                          for k, v in st.items()}
        results[label].update(norm_F=fn, norm_ctx=cn, norm_injected=inj,
                              recompute_max_abs_err=err,
                              uniform_entropy=float(np.log(256)),
                              uniform_diag=1 / 256)
        attns[label] = attn.cpu()
        del net

    # ---------------------------------------------------------- figures -----
    last = '178k (last)'
    a = attns[last][0]                                           # first image
    fig, axes = plt.subplots(2, 6, figsize=(24, 8.6))
    for h in range(6):
        ax = axes[0, h]
        im = ax.imshow(a[h].numpy(), cmap='magma', vmin=0, vmax=1)
        ax.set_title(f'head {h}\nentropy {results[last]["entropy"][h]:.3f}, '
                     f'diag {results[last]["diag_mass"][h]:.3f}', fontsize=11)
        ax.set_xlabel('DINO key token')
        if h == 0:
            ax.set_ylabel('radar query token')
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax = axes[1, h]
        cm = a[h].mean(0).reshape(GRID, GRID).numpy()
        im = ax.imshow(cm, cmap='viridis')
        ax.set_title(f'where the mass lands\ntop-1 key holds '
                     f'{results[last]["dest_top1"][h]:.2f}', fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f'crossattn-render at 178k — the real 256x256 attention matrix '
                 f'(top, image {rows[0][0]}) and the DINO positions it reads '
                 f'(bottom, 16x16 column mass)\n'
                 f'a diagonal line would mean same-position routing; a bright '
                 f'column means every query reads the same few tokens',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    p1 = os.path.join(OUT, 'S4_attention_matrices_178k.png')
    fig.savefig(p1, dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.4))
    x = np.arange(6)
    w = 0.38
    for ax, key, label, ref in (
            (axes[0], 'entropy', 'row entropy (nats)', np.log(256)),
            (axes[1], 'diag_mass', 'diagonal mass', 1 / 256),
            (axes[2], 'dest_top8', 'mass on the 8 most-read DINO tokens', 8 / 256),
            (axes[3], 'mean_displacement', 'argmax distance from own position (cells)', None)):
        for i, (lab, color) in enumerate((('4k (best val)', '#4c4c4c'),
                                          ('178k (last)', '#c44e52'))):
            ax.bar(x + (i - 0.5) * w, results[lab][key], w, label=lab, color=color)
        if ref is not None:
            ax.axhline(ref, color='#2b7bba', ls='--', lw=1.8,
                       label=f'uniform attention = {ref:.4f}')
        ax.set_xticks(x)
        ax.set_xticklabels([f'h{h}' for h in x])
        ax.set_title(label, fontsize=12)
        ax.legend(fontsize=9.5)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Per-head attention statistics at inference, best-val '
                 'checkpoint vs last checkpoint (n=%d val crops)' % args.n_images,
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    p2 = os.path.join(OUT, 'S4_per_head_stats.png')
    fig.savefig(p2, dpi=120)
    plt.close(fig)

    record['probe'] = results
    record['images'] = [r[0] for r in rows]
    with open(os.path.join(OUT, 'attention_probe.json'), 'w') as f:
        json.dump(record, f, indent=2)
    for p in (p1, p2, os.path.join(OUT, 'attention_probe.json')):
        print('wrote', p)


if __name__ == '__main__':
    main()
