"""PHASE 1 — dataset-level spatial DINO consistency across 1e5 / 1e7 / render.

QUESTION
    Across the VALIDATION split, which DINO depth most consistently preserves
    spatial/structural information between Very Noisy Radar (1e5 rays), Clean
    Radar (1e7 rays) and the Render?

    The Phase-0 single-sample run (dino_analysis/visualize_dino_spatial_pca.py,
    sample 0196) suggested intermediate blocks might keep usable structure in the
    1e5 input. One sample cannot settle that. This script measures every valid
    validation triplet and lets the dataset decide. Block 9 is NOT assumed.

WHAT IS REUSED (nothing re-implemented)
    Imported directly from the Phase-0 module:
      build_dataset / DOMAIN_BATCH_KEY   -> the project's pairing + loading
      build_extractor / spatial_tokens   -> the project's offline DINOv2 loader
      batch_to_dino_inputs               -> the project's dino_preprocess path
      compute_layer_means                -> the training-set spatial means
      report_pooled_mean_incompatibility -> the pooled-mean compatibility check
      joint_pca / shared_range / to_map / make_figure  -> the PCA visualisation
      patchwise_cosine                   -> the primary metric
    Phase 1 adds only: validation-set iteration, triplet verification, streaming
    per-sample metrics, aggregation, the by-layer plots and best/worst selection.

METRIC (primary)
    Mean corresponding-patch cosine similarity, computed on the FULL 768-d patch
    features -- never on PCA values. For each spatial position p,
    cos(feature_A[p], feature_B[p]), averaged over all 256 patch positions.
    PCA is used only to draw the best/worst pictures.

DIFFERENT-SCENE CONTROL (added, not in the original spec)
    Each sample is also compared against the PREVIOUS valid sample's features
    (a different object). Phase 0 showed this matters decisively: raw
    1e5-vs-render reached +0.42 while the wrong-object baseline was +0.41, i.e.
    almost all of the "similarity" was a common component shared by every image.
    An absolute cosine cannot be read without it. It costs no extra DINO
    forwards (one sample's features are held in memory) and is reported in
    separate columns / dashed plot lines, so the primary metric stays exactly as
    specified. Disable with --no-control.

CENTERING
    Uses the Phase-0 statistic unchanged: per-(domain, block) spatial means over
    150 TRAIN-split images, averaged over images AND patch positions, cached at
    dino_analysis/dino_spatial_layer_means.pt. Training data only -- validation
    images never enter the statistic, and the evaluated triplet never centres
    itself. The repo's older dino_feat_mean_*.pt are 3072-d POOLED vectors and
    are reported as incompatible and refused.

SCOPE
    Phase 1 only. No Phase 2, no restoration model, no training.
"""

import argparse
import csv
import datetime
import json
import os
import sys

import numpy as np
import torch
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- reuse the Phase-0 module wholesale ------------------------------------
_PHASE1 = os.path.dirname(os.path.abspath(__file__))
_DINO_ANALYSIS = os.path.dirname(_PHASE1)
_REPO = os.path.dirname(_DINO_ANALYSIS)
for _p in (_DINO_ANALYSIS, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import visualize_dino_spatial_pca as base   # noqa: E402

DOMAINS = base.DOMAINS                       # ['1e5', '1e7', 'render']
PAIRS = base.PAIRS                           # [(1e5,1e7), (1e5,render), (1e7,render)]
PAIR_LABEL = {('1e5', '1e7'): '1e5<->1e7',
              ('1e5', 'render'): '1e5<->Render',
              ('1e7', 'render'): '1e7<->Render'}
PAIR_KEY = {p: f'{p[0]}_vs_{p[1]}' for p in PAIRS}
FEATURE_TYPES = ['raw', 'centered']
OUT_DEFAULT = os.path.join(_PHASE1, 'outputs')
DEVLOG = os.path.join(_PHASE1, 'PHASE1_DEVLOG.md')
STATS = ['n', 'mean', 'sem', 'std', 'median', 'p25', 'p75', 'min', 'max']


# ---------------------------------------------------------------------------
# triplet verification
# ---------------------------------------------------------------------------
def verify_triplet(ds, i):
    """Confirm index i is a complete, consistent 1e5/1e7/render triplet.

    Returns (sample_id, paths, batch, reason). reason is None when valid;
    otherwise batch is None and reason says why the sample is skipped.
    """
    gt_path = ds.paths[i]['gt_path']
    lq_path = ds.paths[i]['lq_path']
    render_path = ds._render_path(gt_path)
    paths = {'1e5': lq_path, '1e7': gt_path, 'render': render_path}
    sample_id = os.path.splitext(os.path.basename(gt_path))[0]

    bases = {k: os.path.basename(v) for k, v in paths.items()}
    if len(set(bases.values())) != 1:
        return sample_id, paths, None, f'basename mismatch: {bases}'
    for k, v in paths.items():
        if not os.path.isfile(v):
            return sample_id, paths, None, f'missing {k} file: {v}'

    try:
        batch = ds[i]
    except Exception as e:                                   # noqa: BLE001
        return sample_id, paths, None, f'load error: {type(e).__name__}: {e}'

    # spatial correspondence is only meaningful if the three share a raster
    shapes = {d: tuple(batch[base.DOMAIN_BATCH_KEY[d]].shape[-2:]) for d in DOMAINS}
    if len(set(shapes.values())) != 1:
        return sample_id, paths, None, f'shape mismatch across domains: {shapes}'
    # the returned lq/gt paths must be the ones we verified
    if (batch['gt_path'] != gt_path) or (batch['lq_path'] != lq_path):
        return sample_id, paths, None, 'dataset returned a different path than indexed'
    return sample_id, paths, batch, None


# ---------------------------------------------------------------------------
# feature extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_features(ext, batch, blocks0, img_size, device):
    """One forward for the whole triplet -> {domain: [ [N,C] per block ]}.

    The three domains are stacked into a single batch of 3 so the ordering of
    patch tokens is produced by exactly the same code path for all of them.
    """
    inputs = base.batch_to_dino_inputs(batch, img_size, ext.mean.cpu(), ext.std.cpu())
    stack = torch.cat([inputs[d] for d in DOMAINS], dim=0).to(device)   # [3,3,S,S]
    per_block = base.spatial_tokens(ext, stack, blocks0)                # [3,N,C] each
    feats = {d: [pb[k].numpy() for pb in per_block] for k, d in enumerate(DOMAINS)}
    return inputs, feats


def center(feats, means, mode):
    """Subtract the fixed TRAINING mean. Never derived from this sample."""
    if mode == 'per-domain':
        return {d: [feats[d][li] - means['per_domain'][d][li].numpy()
                    for li in range(len(feats[d]))] for d in DOMAINS}
    return {d: [feats[d][li] - means['global'][li].numpy()
                for li in range(len(feats[d]))] for d in DOMAINS}


# ---------------------------------------------------------------------------
# aggregation (reads back the per-sample CSV -- sanity check 13)
# ---------------------------------------------------------------------------
def summarize(per_sample_csv):
    """Aggregate straight from the SAVED per-sample file, not from memory."""
    rows = list(csv.DictReader(open(per_sample_csv)))
    buckets = {}
    for r in rows:
        for pair in PAIRS:
            for scene, col in (('same', f'similarity_{PAIR_KEY[pair]}'),
                               ('different', f'control_{PAIR_KEY[pair]}')):
                v = r.get(col, '')
                if v in ('', 'nan', None):
                    continue
                key = (r['feature_type'], r['layer_fraction'], int(r['block_index']),
                       PAIR_KEY[pair], scene)
                buckets.setdefault(key, []).append(float(v))

    out = []
    for key, vals in buckets.items():
        ft, frac, blk, pair, scene = key
        a = np.asarray(vals, dtype=np.float64)
        out.append({
            'feature_type': ft, 'layer_fraction': frac, 'block_index': blk,
            'pair': pair, 'scene_match': scene,
            'n': len(a), 'mean': float(a.mean()),
            'sem': float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else float('nan'),
            'std': float(a.std(ddof=1)) if len(a) > 1 else float('nan'),
            'median': float(np.median(a)),
            'p25': float(np.percentile(a, 25)), 'p75': float(np.percentile(a, 75)),
            'min': float(a.min()), 'max': float(a.max()),
        })
    out.sort(key=lambda r: (r['feature_type'], r['block_index'], r['pair'],
                            r['scene_match']))
    return rows, out


def lookup(summary, ft, blk, pair, scene='same', field='mean'):
    for r in summary:
        if (r['feature_type'] == ft and r['block_index'] == blk
                and r['pair'] == pair and r['scene_match'] == scene):
            return r[field]
    return float('nan')


def print_table(summary, ft, blocks1, fracs):
    print(f'\n{ft.upper()} FEATURES   (mean +/- std over valid val triplets; '
          f'[..] = different-scene control mean)')
    head = f'  {"Layer":<10}' + ''.join(f'{PAIR_LABEL[p]:>28}' for p in PAIRS)
    print(head)
    print('  ' + '-' * (len(head) - 2))
    for frac, b in zip(fracs, blocks1):
        cells = ''
        for p in PAIRS:
            m = lookup(summary, ft, b, PAIR_KEY[p], 'same', 'mean')
            s = lookup(summary, ft, b, PAIR_KEY[p], 'same', 'std')
            c = lookup(summary, ft, b, PAIR_KEY[p], 'different', 'mean')
            cells += f'{m:+.4f} +/- {s:.4f} [{c:+.3f}]'.rjust(28)
        print(f'  Block {b:<4}{cells}')


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def plot_by_layer(path, summary, ft, blocks1, fracs, n_valid, dpi, show_control):
    colors = {('1e5', '1e7'): '#0072B2', ('1e5', 'render'): '#D55E00',
              ('1e7', 'render'): '#009E73'}
    x = np.arange(len(blocks1))
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    for p in PAIRS:
        m = [lookup(summary, ft, b, PAIR_KEY[p], 'same', 'mean') for b in blocks1]
        e = [lookup(summary, ft, b, PAIR_KEY[p], 'same', 'sem') for b in blocks1]
        ax.errorbar(x, m, yerr=e, marker='o', ms=5, lw=1.8, capsize=3,
                    color=colors[p], label=PAIR_LABEL[p].replace('<->', ' ↔ '))
        if show_control:
            c = [lookup(summary, ft, b, PAIR_KEY[p], 'different', 'mean')
                 for b in blocks1]
            ax.plot(x, c, marker='.', ms=4, lw=1.0, ls='--', alpha=0.75,
                    color=colors[p])
    ax.set_xticks(x)
    ax.set_xticklabels([f'B{b}\n({f})' for b, f in zip(blocks1, fracs)])
    ax.set_xlabel('DINO transformer block (depth)')
    ax.set_ylabel('Mean corresponding-patch cosine similarity')
    ax.set_title(f'Spatial DINO consistency across the validation set — '
                 f'{ft} features\n'
                 f'n = {n_valid} triplets, error bars = ± 1 SEM'
                 + ('; dashed = different-scene control' if show_control else ''),
                 fontsize=10)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(frameon=False, fontsize=9)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f'  wrote {os.path.relpath(path, _REPO)}')


def pca_figure_for_sample(path, title, inputs, feats, blocks1, fracs, gh, gw,
                          ext, args, seed):
    """Phase-0 style PCA panel: joint PCA per layer, shared range per layer."""
    pc1_maps, pc1_ranges = [], []
    lo, hi = args.percentile
    for li in range(len(blocks1)):
        _, proj = base.joint_pca({d: feats[d][li] for d in DOMAINS}, 3, seed)
        pc1_ranges.append(base.shared_range(proj, 0, lo, hi))
        pc1_maps.append({d: base.to_map(proj[d][:, 0], gh, gw) for d in DOMAINS})
    col_titles = [f'{f}-depth DINO\n(Block {b})' for f, b in zip(fracs, blocks1)]
    base.make_figure(path, title, inputs, pc1_maps, pc1_ranges, col_titles,
                     args.cmap, args.display_size, ext.mean.cpu(), ext.std.cpu(),
                     args.dpi)


# ---------------------------------------------------------------------------
# devlog
# ---------------------------------------------------------------------------
def append_devlog(stamp, summary, blocks1, fracs, ctx):
    """Append a dated Phase-1 entry whose Results come from THIS run."""
    def best(ft, pair):
        vals = [(lookup(summary, ft, b, PAIR_KEY[pair], 'same', 'mean'), b)
                for b in blocks1]
        return max(vals)[1], max(vals)[0]

    def row(ft, pair):
        return ' | '.join(
            f'{lookup(summary, ft, b, PAIR_KEY[pair], "same", "mean"):+.4f} '
            f'± {lookup(summary, ft, b, PAIR_KEY[pair], "same", "sem"):.4f}'
            for b in blocks1)

    b_raw, v_raw = best('raw', ('1e5', '1e7'))
    b_cen, v_cen = best('centered', ('1e5', '1e7'))
    b_r_lr, v_r_lr = best('centered', ('1e5', 'render'))
    b_r_hr, v_r_hr = best('centered', ('1e7', 'render'))
    rank_raw = sorted(blocks1, key=lambda b: -lookup(summary, 'raw', b, '1e5_vs_1e7'))
    rank_cen = sorted(blocks1, key=lambda b: -lookup(summary, 'centered', b, '1e5_vs_1e7'))
    ranking_changed = rank_raw != rank_cen
    final = blocks1[-1]
    inter_better = all(
        lookup(summary, 'centered', b, '1e5_vs_1e7')
        > lookup(summary, 'centered', final, '1e5_vs_1e7')
        for b in blocks1[:-1])
    b9 = 9 if 9 in blocks1 else None
    b9_rank = (rank_cen.index(9) + 1) if b9 else None

    hdr = '| Features | ' + ' | '.join(f'B{b}' for b in blocks1) + ' |'
    sep = '|---' * (len(blocks1) + 1) + '|'
    top_changed = rank_raw[0] != rank_cen[0]

    entry = f"""
## {stamp} — Dataset-Level Spatial DINO Consistency

### Why
The Phase-0 single-sample run (sample 0196) suggested that intermediate DINO
blocks might retain usable spatial structure even from the extremely noisy
1e5-ray radar image. One example cannot support that, so we measured the whole
validation split to see whether the behaviour generalises. Block 9 was not
assumed to be the answer.

### What we did
- iterated the complete {ctx['split']} split and verified every 1e5/1e7/render triplet
- extracted spatial DINO patch tokens (CLS/register removed, nothing pooled)
- evaluated blocks {blocks1} (quarter depths, 1-indexed)
- measured mean corresponding-patch cosine similarity on the full {ctx['embed_dim']}-d features
- repeated for RAW and CENTERED features (fixed training means)
- added a different-scene control (previous valid sample) so the absolute
  cosines are interpretable
- selected and visualised the 3 best and 3 worst samples by 1e5↔1e7

### How
- model: `{ctx['model']}`, frozen, offline local load, strict=True
- checkpoint: `{ctx['checkpoint']}`
- split: {ctx['split']} — {ctx['total']} considered, **{ctx['valid']} valid triplets**, {ctx['skipped']} skipped
- blocks 1-indexed {blocks1} → 0-indexed {ctx['blocks0']}
- DINO input {ctx['img_size']}×{ctx['img_size']}, patch {ctx['patch']} → **{ctx['gh']}×{ctx['gw']} = {ctx['ntok']} patch tokens**, {ctx['embed_dim']}-d
- centering: {ctx['centering_mode']} spatial means over {ctx['n_mean_images']} TRAIN images
  (`{ctx['means_cache']}`); validation never enters the statistic
- primary metric: mean corresponding-patch cosine on full feature vectors (not PCA)
- uncertainty in plots: ± 1 standard error of the mean over the {ctx['valid']} triplets
- seed {ctx['seed']}, device {ctx['device']}

### Results

**1e5 ↔ 1e7 (the primary comparison), mean ± SEM:**

{hdr}
{sep}
| RAW | {row('raw', ('1e5', '1e7'))} |
| CENTERED | {row('centered', ('1e5', '1e7'))} |

**1e5 ↔ Render:**

{hdr}
{sep}
| RAW | {row('raw', ('1e5', 'render'))} |
| CENTERED | {row('centered', ('1e5', 'render'))} |

**1e7 ↔ Render:**

{hdr}
{sep}
| RAW | {row('raw', ('1e7', 'render'))} |
| CENTERED | {row('centered', ('1e7', 'render'))} |

**Different-scene control (wrong object), 1e5↔1e7:**

{hdr}
{sep}
| RAW | {' | '.join(f"{lookup(summary, 'raw', b, '1e5_vs_1e7', 'different'):+.4f}" for b in blocks1)} |
| CENTERED | {' | '.join(f"{lookup(summary, 'centered', b, '1e5_vs_1e7', 'different'):+.4f}" for b in blocks1)} |

**Direct answers to the Phase-1 questions:**
1. Highest mean RAW 1e5↔1e7: **Block {b_raw}** ({v_raw:+.4f})
2. Highest mean CENTERED 1e5↔1e7: **Block {b_cen}** ({v_cen:+.4f})
3. Highest 1e5↔Render (centered): **Block {b_r_lr}** ({v_r_lr:+.4f})
4. Highest 1e7↔Render (centered): **Block {b_r_hr}** ({v_r_hr:+.4f})
5. Does centering change the layer ranking? **{'YES' if ranking_changed else 'NO'}** — raw order {rank_raw}, centered order {rank_cen}
6. Are all intermediate blocks above the final block ({final}) for centered 1e5↔1e7? **{'YES' if inter_better else 'NO'}**
7. Block 9 across the dataset: rank **{b9_rank} of {len(blocks1)}** on centered 1e5↔1e7 ({lookup(summary, 'centered', 9, '1e5_vs_1e7'):+.4f}) — {'the single-sample impression holds' if b9_rank == 1 else 'it is NOT the top block; the single-sample impression did not generalise'}

**Best / worst samples** (block {ctx['bw_block']}, {ctx['bw_feature']} features, 1e5↔1e7):
- best: {ctx['best_str']}
- worst: {ctx['worst_str']}

**Skipped samples:** {ctx['skip_str']}

### Interpretation
- The layer ordering above is what the validation set actually supports; the
  Phase-0 single-sample reading is superseded by it.
- Read the same-scene numbers against the different-scene control on the same
  row. Where the two are close, DINO is reporting a component shared by all
  images rather than this scene's structure.
- Centering {'changes' if top_changed else 'does not change'} which block ranks HIGHEST for 1e5↔1e7
  (raw best B{rank_raw[0]}, centered best B{rank_cen[0]}), and {'does change' if ranking_changed else 'leaves'} the full ordering{'' if ranking_changed else ' unchanged'}.
- {'Every intermediate block beats the final block' if inter_better else 'The intermediate blocks do NOT all beat the final block'} on centered 1e5↔1e7, which is the
  direct evidence on whether noise-stability is an intermediate-layer property.

### Outputs
- `dino_analysis/phase1/outputs/dino_spatial_similarity_val_per_sample.csv`
- `dino_analysis/phase1/outputs/dino_spatial_similarity_val_summary.csv`
- `dino_analysis/phase1/outputs/dino_spatial_similarity_by_layer_raw.png`
- `dino_analysis/phase1/outputs/dino_spatial_similarity_by_layer_centered.png`
- `dino_analysis/phase1/outputs/best_samples/`, `.../worst_samples/`
- `dino_analysis/phase1/outputs/phase1_metadata.json`

### Next Step
Phase 2 — compare DINO(1e5) and DINO(Render) as candidate structural priors
relative to DINO(1e7). Not implemented yet.

---
"""
    with open(DEVLOG, 'a') as f:
        f.write(entry)
    print(f'  appended entry to {DEVLOG}')


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description='Phase 1: dataset-level spatial DINO consistency '
                    '(1e5 / 1e7 / render) over the validation split.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--split', default='val', choices=['val', 'test', 'train'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-samples', type=int, default=None,
                    help='smoke test: analyse only the first N samples. Output '
                         'then goes to outputs/smoke/ so a full run is never '
                         'overwritten (override with --out-dir).')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--opt', default=base.DEFAULT_OPT,
                    help='arm config the DINO model/paths are read from')
    ap.add_argument('--means-cache', default=base.DEFAULT_MEANS)
    ap.add_argument('--compute-layer-means', action='store_true',
                    help='(re)compute the training spatial means instead of '
                         'loading the cache')
    ap.add_argument('--num-mean-samples', type=int, default=150)
    ap.add_argument('--centering-mode', default='per-domain',
                    choices=['per-domain', 'global'])
    ap.add_argument('--no-control', action='store_true',
                    help='skip the different-scene control columns')
    ap.add_argument('--n-best-worst', type=int, default=3)
    ap.add_argument('--best-worst-layer', type=int, default=None,
                    help='1-indexed block for best/worst selection; default = '
                         'the block with the highest mean centered 1e5<->1e7')
    ap.add_argument('--best-worst-feature', default='centered',
                    choices=FEATURE_TYPES)
    ap.add_argument('--percentile', type=float, nargs=2, default=(1.0, 99.0),
                    metavar=('LO', 'HI'))
    ap.add_argument('--display-size', type=int, default=224)
    ap.add_argument('--cmap', default='viridis')
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--no-devlog', action='store_true')
    args = ap.parse_args()

    smoke = args.max_samples is not None
    out_dir = args.out_dir or (os.path.join(OUT_DEFAULT, 'smoke') if smoke
                               else OUT_DEFAULT)
    best_dir = os.path.join(out_dir, 'best_samples')
    worst_dir = os.path.join(out_dir, 'worst_samples')
    for d in (out_dir, best_dir, worst_dir):
        os.makedirs(d, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = datetime.datetime.now().astimezone()

    with open(args.opt) as f:
        cfg = yaml.safe_load(f)
    net = cfg['network_g']

    print('=' * 78)
    print('PHASE 1 — dataset-level spatial DINO consistency')
    print('=' * 78)
    print(f'  started        : {started.isoformat(timespec="seconds")}')
    print(f'  output dir     : {os.path.relpath(out_dir, _REPO)}'
          + ('   (SMOKE TEST — full-run outputs untouched)' if smoke else ''))

    # ---------------- model ----------------
    ext = base.build_extractor(cfg, (0,), args.device)
    n_blocks = len(ext.dino.blocks)
    patch = int(ext.dino.patch_size)
    img_size = int(net['dino_img_size'])
    embed_dim = int(ext.dino.embed_dim)
    n_reg = int(getattr(ext.dino, 'num_register_tokens', 0))
    gh = gw = img_size // patch
    blocks1 = [int(round(n_blocks * k / 4)) for k in (1, 2, 3, 4)]
    blocks0 = [b - 1 for b in blocks1]
    fracs = ['1/4', '2/4', '3/4', '4/4']
    ext.layers = tuple(blocks0)

    print('\n  --- DINO model (identical to the Phase-0 single-sample run) ---')
    print(f'  model name         : {net["dino_model_name"]}')
    print(f'  hub source         : {net["dino_hub_source"]}  (offline)')
    print(f'  checkpoint         : {net["dino_weights"]}')
    print(f'  strict load        : missing={list(ext.load_result.missing_keys)} '
          f'unexpected={list(ext.load_result.unexpected_keys)}')
    print(f'  transformer blocks : {n_blocks}')
    print(f'  embedding dim      : {embed_dim}')
    print(f'  patch size         : {patch}')
    print(f'  DINO input res     : {img_size} x {img_size}')
    print(f'  patch grid         : {gh} x {gw} = {gh * gw} tokens')
    print(f'  selected blocks    : 1-indexed {blocks1} -> 0-indexed {blocks0} '
          f'({", ".join(fracs)} depth)')
    print(f'  special tokens     : CLS + {n_reg} register tokens removed by '
          f'DINOv2 (get_intermediate_layers(return_class_token=False) slices '
          f'out[:, 1+num_register_tokens:]); patch tokens only, no pooling')
    if n_reg != 0:
        raise SystemExit(f'unexpected register tokens ({n_reg}) — token layout '
                         f'must be re-checked before trusting these numbers')

    # ---------------- centering statistics ----------------
    print('\n  --- centering statistics ---')
    pooled_info = base.report_pooled_mean_incompatibility(embed_dim)
    if args.compute_layer_means or not os.path.isfile(args.means_cache):
        if not args.compute_layer_means:
            print(f'  no cache at {args.means_cache} — computing it now')
        per_dom, glob, used = base.compute_layer_means(
            ext, cfg, blocks0, args.num_mean_samples, args.seed, args.device,
            img_size)
        means = {'per_domain': per_dom, 'global': glob, 'meta': {
            'blocks_1indexed': blocks1, 'blocks_0indexed': blocks0,
            'embed_dim': embed_dim, 'img_size': img_size, 'patch_size': patch,
            'patch_grid': [gh, gw], 'n_train_images': len(used),
            'seed': args.seed, 'split': 'train',
            'phase': 'val (full 256x256, no crop)',
            'model': net['dino_model_name'], 'weights': net['dino_weights'],
            'statistic': 'mean over training images AND patch positions of the '
                         'spatial patch tokens',
            'created': datetime.datetime.now().astimezone().isoformat(
                timespec='seconds'),
            'script': 'dino_analysis/phase1/analyze_dino_spatial_consistency.py',
            'example_files': [os.path.basename(p) for p in used[:5]]}}
        torch.save(means, args.means_cache)
        print(f'  saved -> {os.path.relpath(args.means_cache, _REPO)}')
    else:
        means = torch.load(args.means_cache, map_location='cpu', weights_only=False)
        print(f'  loaded {os.path.relpath(args.means_cache, _REPO)}')
    mm = means['meta']
    if mm['blocks_0indexed'] != blocks0 or mm['embed_dim'] != embed_dim:
        raise SystemExit('cached spatial means were built for a different block '
                         'set / model — rerun with --compute-layer-means')
    if mm.get('split') != 'train':
        raise SystemExit(f'centering means come from split {mm.get("split")!r}, '
                         f'not train — refusing to proceed (validation leakage)')
    print(f'  statistic      : {mm["statistic"]}')
    print(f'  source         : {mm["n_train_images"]} images from the TRAIN split '
          f'(seed {mm["seed"]}) — no validation data enters the mean')
    print(f'  centering mode : {args.centering_mode}')

    # ---------------- validation iteration ----------------
    ds = base.build_dataset(cfg, args.split)
    total = len(ds)
    n_run = min(args.max_samples, total) if smoke else total
    print(f'\n  --- {args.split} split ---')
    print(f'  dataroot lq    : {cfg["datasets"]["val"]["dataroot_lq"].replace("/val_", f"/{args.split}_")}')
    print(f'  total samples considered : {n_run} of {total}')

    per_sample_path = os.path.join(
        out_dir, 'dino_spatial_similarity_val_per_sample.csv')
    cols = ['sample_id', 'feature_type', 'layer_fraction', 'block_index',
            'block_index_0based']
    cols += [f'similarity_{PAIR_KEY[p]}' for p in PAIRS]
    cols += ['control_partner_id'] + [f'control_{PAIR_KEY[p]}' for p in PAIRS]

    skipped, valid_ids, rows_written = [], [], 0
    prev = None                      # (sample_id, raw feats) for the control
    with open(per_sample_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for i in range(n_run):
            sid, paths, batch, reason = verify_triplet(ds, i)
            if reason is not None:
                skipped.append({'sample_id': sid, 'index': i, 'reason': reason})
                print(f'  [skip] {sid}: {reason}')
                continue

            _, raw = sample_features(ext, batch, blocks0, img_size, args.device)
            n_tok = raw['1e5'][0].shape[0]
            if n_tok != gh * gw:
                raise SystemExit(f'{sid}: N_tokens {n_tok} != {gh}*{gw} — patch '
                                 f'grid must not be guessed')
            variants = {'raw': raw,
                        'centered': center(raw, means, args.centering_mode)}
            prev_variants = None
            if prev is not None and not args.no_control:
                prev_variants = {'raw': prev[1],
                                 'centered': center(prev[1], means,
                                                    args.centering_mode)}

            for ft in FEATURE_TYPES:
                for li, (frac, b1, b0) in enumerate(zip(fracs, blocks1, blocks0)):
                    row = {'sample_id': sid, 'feature_type': ft,
                           'layer_fraction': frac, 'block_index': b1,
                           'block_index_0based': b0,
                           'control_partner_id': prev[0] if prev_variants else ''}
                    for a, c in PAIRS:
                        row[f'similarity_{PAIR_KEY[(a, c)]}'] = (
                            f'{base.patchwise_cosine(variants[ft][a][li], variants[ft][c][li]):.6f}')
                        row[f'control_{PAIR_KEY[(a, c)]}'] = (
                            f'{base.patchwise_cosine(variants[ft][a][li], prev_variants[ft][c][li]):.6f}'
                            if prev_variants else '')
                    w.writerow(row)
                    rows_written += 1

            valid_ids.append(sid)
            prev = (sid, raw)
            if len(valid_ids) % 25 == 0:
                print(f'  {len(valid_ids)} valid triplets processed '
                      f'({i + 1}/{n_run} scanned)')

    print(f'\n  total considered        : {n_run}')
    print(f'  valid triplets analysed : {len(valid_ids)}')
    print(f'  skipped                 : {len(skipped)}')
    if skipped:
        for s in skipped:
            print(f'    {s["sample_id"]}: {s["reason"]}')
    print(f'  wrote {os.path.relpath(per_sample_path, _REPO)} '
          f'({rows_written} rows)')
    if not valid_ids:
        raise SystemExit('no valid triplets — nothing to summarise')

    # ---------------- aggregation (from the saved CSV) ----------------
    per_rows, summary = summarize(per_sample_path)
    summary_path = os.path.join(
        out_dir, 'dino_spatial_similarity_val_summary.csv')
    scols = ['feature_type', 'layer_fraction', 'block_index', 'pair',
             'scene_match'] + STATS
    with open(summary_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=scols)
        w.writeheader()
        for r in summary:
            w.writerow({k: (f'{r[k]:.6f}' if isinstance(r[k], float) else r[k])
                        for k in scols})
    print(f'  wrote {os.path.relpath(summary_path, _REPO)}')

    for ft in FEATURE_TYPES:
        print_table(summary, ft, blocks1, fracs)

    # ---------------- by-layer plots ----------------
    print()
    for ft in FEATURE_TYPES:
        plot_by_layer(
            os.path.join(out_dir, f'dino_spatial_similarity_by_layer_{ft}.png'),
            summary, ft, blocks1, fracs, len(valid_ids), args.dpi,
            not args.no_control)

    # ---------------- explicit answers ----------------
    def best_block(ft, pair):
        """-> (block_index, mean) of the highest-scoring block."""
        value, block = max((lookup(summary, ft, b, PAIR_KEY[pair]), b)
                           for b in blocks1)
        return block, value

    rank_raw = sorted(blocks1, key=lambda b: -lookup(summary, 'raw', b, '1e5_vs_1e7'))
    rank_cen = sorted(blocks1, key=lambda b: -lookup(summary, 'centered', b, '1e5_vs_1e7'))
    final = blocks1[-1]
    inter_better = all(lookup(summary, 'centered', b, '1e5_vs_1e7')
                       > lookup(summary, 'centered', final, '1e5_vs_1e7')
                       for b in blocks1[:-1])
    print('\n' + '=' * 78)
    print('PHASE 1 ANSWERS')
    print('=' * 78)
    for n, (ft, pair) in enumerate(
            [('raw', ('1e5', '1e7')), ('centered', ('1e5', '1e7')),
             ('centered', ('1e5', 'render')), ('centered', ('1e7', 'render'))], 1):
        b, v = best_block(ft, pair)
        print(f'  {n}. highest mean {ft:8s} {PAIR_LABEL[pair]:14s}: Block {b} ({v:+.4f})')
    print(f'  5. centering changes the layer ranking: '
          f'{"YES" if rank_raw != rank_cen else "NO"}  '
          f'(raw {rank_raw} vs centered {rank_cen})')
    print(f'  6. all intermediate blocks beat the final block {final} '
          f'(centered 1e5<->1e7): {"YES" if inter_better else "NO"}')
    if 9 in blocks1:
        print(f'  7. Block 9 rank on centered 1e5<->1e7: '
              f'{rank_cen.index(9) + 1} of {len(blocks1)} '
              f'({lookup(summary, "centered", 9, "1e5_vs_1e7"):+.4f})')

    # ---------------- best / worst ----------------
    bw_ft = args.best_worst_feature
    bw_block = args.best_worst_layer or best_block(bw_ft, ('1e5', '1e7'))[0]
    print(f'\n  best/worst selection: {bw_ft} features, Block {bw_block}, '
          f'1e5<->1e7 (chosen '
          + ('by --best-worst-layer' if args.best_worst_layer else
             'as the highest dataset-level mean') + ')')
    cand = sorted(
        [(float(r['similarity_1e5_vs_1e7']), r['sample_id']) for r in per_rows
         if r['feature_type'] == bw_ft and int(r['block_index']) == bw_block],
        reverse=True)
    k = min(args.n_best_worst, len(cand))
    best_list, worst_list = cand[:k], list(reversed(cand[-k:]))
    print(f'    best  : ' + ', '.join(f'{s} ({v:+.4f})' for v, s in best_list))
    print(f'    worst : ' + ', '.join(f'{s} ({v:+.4f})' for v, s in worst_list))

    id_to_index = {os.path.splitext(os.path.basename(p['gt_path']))[0]: n
                   for n, p in enumerate(ds.paths)}
    for kind, lst, d in (('best', best_list, best_dir), ('worst', worst_list, worst_dir)):
        for rank, (val, sid) in enumerate(lst, 1):
            _, _, batch, reason = verify_triplet(ds, id_to_index[sid])
            if reason:
                print(f'  [skip figure] {sid}: {reason}')
                continue
            inputs, raw = sample_features(ext, batch, blocks0, img_size, args.device)
            feats = (raw if bw_ft == 'raw'
                     else center(raw, means, args.centering_mode))
            human = ('RAW spatial DINO features' if bw_ft == 'raw' else
                     f'CENTERED spatial DINO features ({args.centering_mode} '
                     f'training mean, n={mm["n_train_images"]})')
            title = (f'Spatial DINO PCA-1 — Very Noisy Radar (1e5) vs Clean Radar '
                     f'(1e7) vs Render\n{human}   |   {kind.upper()} rank {rank:02d}'
                     f'   |   sample {sid}   |   1e5↔1e7 @ B{bw_block} = {val:+.4f}\n'
                     f'{net["dino_model_name"]}, {gh}x{gw} patch grid   |   one '
                     f'joint PCA per layer over all three inputs, shared display '
                     f'range per layer')
            path = os.path.join(
                d, f'dino_spatial_pca_{kind}_rank{rank:02d}_sample_{sid}_{bw_ft}.png')
            pca_figure_for_sample(path, title, inputs, feats, blocks1, fracs,
                                  gh, gw, ext, args, args.seed)

    # ---------------- metadata ----------------
    meta = {
        'phase': 1,
        'execution_date': started.strftime('%Y-%m-%d'),
        'execution_local_time': started.strftime('%H:%M:%S'),
        'timezone': started.tzname(),
        'utc_offset': started.strftime('%z'),
        'finished_local_time': datetime.datetime.now().astimezone().strftime('%H:%M:%S'),
        'command': ' '.join([os.path.relpath(sys.argv[0], _REPO)] + sys.argv[1:]),
        'smoke_test': smoke,
        'output_dir': os.path.relpath(out_dir, _REPO),
        'dino_model': net['dino_model_name'],
        'dino_hub_source': net['dino_hub_source'],
        'dino_hub_dir': net['dino_hub_dir'],
        'dino_checkpoint': net['dino_weights'],
        'config_read': os.path.relpath(args.opt, _REPO),
        'split': args.split,
        'split_dataroots': {
            k: cfg['datasets']['val'][k].replace('/val_', f'/{args.split}_')
            for k in ('dataroot_lq', 'dataroot_gt', 'dataroot_render')},
        'total_validation_samples_considered': n_run,
        'total_in_split': total,
        'valid_triplets_analyzed': len(valid_ids),
        'skipped_count': len(skipped),
        'skipped_sample_ids': [s['sample_id'] for s in skipped],
        'skip_reasons': skipped,
        'transformer_block_count': n_blocks,
        'selected_blocks_1indexed': blocks1,
        'selected_blocks_0indexed': blocks0,
        'layer_fractions': fracs,
        'patch_size': patch,
        'num_register_tokens': n_reg,
        'special_token_handling': 'CLS + register tokens removed by DINOv2 '
                                  'get_intermediate_layers(return_class_token='
                                  'False); spatial patch tokens only, no pooling',
        'dino_input_resolution': [img_size, img_size],
        'patch_grid': [gh, gw],
        'n_patch_tokens': gh * gw,
        'embedding_dim': embed_dim,
        'feature_types': FEATURE_TYPES,
        'centering': {
            'mode': args.centering_mode,
            'statistic': mm['statistic'],
            'source_split': mm.get('split'),
            'n_training_images': mm['n_train_images'],
            'cache_file': os.path.relpath(args.means_cache, _REPO),
            'validation_leakage': False,
            'existing_pooled_means_examined': pooled_info,
            'existing_pooled_means_compatible': False,
        },
        'primary_metric': 'mean corresponding-patch cosine similarity over the '
                          f'full {embed_dim}-d patch features (never PCA values)',
        'different_scene_control': {
            'enabled': not args.no_control,
            'definition': 'each sample is also compared against the PREVIOUS '
                          'valid sample (a different object); the first valid '
                          'sample has no partner and is blank',
        },
        'uncertainty_method': 'error bars are +/- 1 standard error of the mean '
                              '(std with ddof=1 / sqrt(n)) over valid triplets',
        'pca_usage': 'visualisation only; one joint PCA per layer fitted on the '
                     'concatenated 1e5+1e7+render patch tokens, same basis and '
                     'same shared display range applied to all three rows',
        'pca_display_percentiles': list(args.percentile),
        'best_worst': {
            'feature_type': bw_ft, 'block_index': bw_block,
            'pair': '1e5_vs_1e7',
            'selection': 'programmatic, from the saved per-sample CSV',
            'best': [{'rank': n, 'sample_id': s, 'similarity': v}
                     for n, (v, s) in enumerate(best_list, 1)],
            'worst': [{'rank': n, 'sample_id': s, 'similarity': v}
                      for n, (v, s) in enumerate(worst_list, 1)],
        },
        'answers': {
            k: {'block_index': bb[0], 'mean': bb[1]} for k, bb in (
                ('best_block_raw_1e5_vs_1e7', best_block('raw', ('1e5', '1e7'))),
                ('best_block_centered_1e5_vs_1e7', best_block('centered', ('1e5', '1e7'))),
                ('best_block_centered_1e5_vs_render', best_block('centered', ('1e5', 'render'))),
                ('best_block_centered_1e7_vs_render', best_block('centered', ('1e7', 'render'))),
            )} | {
            'raw_ranking_1e5_vs_1e7': rank_raw,
            'centered_ranking_1e5_vs_1e7': rank_cen,
            'centering_changes_ranking': rank_raw != rank_cen,
            'all_intermediate_beat_final_centered': inter_better,
        },
        'random_seed': args.seed,
        'device': args.device,
        'torch_version': torch.__version__,
    }
    meta_path = os.path.join(out_dir, 'phase1_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'\n  wrote {os.path.relpath(meta_path, _REPO)}')

    # ---------------- devlog ----------------
    if smoke:
        print('  smoke test — devlog NOT touched')
    elif args.no_devlog:
        print('  --no-devlog — devlog NOT touched')
    else:
        ctx = {
            'split': args.split, 'total': n_run, 'valid': len(valid_ids),
            'skipped': len(skipped), 'blocks0': blocks0,
            'model': net['dino_model_name'], 'checkpoint': net['dino_weights'],
            'img_size': img_size, 'patch': patch, 'gh': gh, 'gw': gw,
            'ntok': gh * gw, 'embed_dim': embed_dim,
            'centering_mode': args.centering_mode,
            'n_mean_images': mm['n_train_images'],
            'means_cache': os.path.relpath(args.means_cache, _REPO),
            'seed': args.seed, 'device': args.device,
            'bw_block': bw_block, 'bw_feature': bw_ft,
            'best_str': ', '.join(f'{s} ({v:+.4f})' for v, s in best_list),
            'worst_str': ', '.join(f'{s} ({v:+.4f})' for v, s in worst_list),
            'skip_str': (', '.join(f'{s["sample_id"]} ({s["reason"]})'
                                   for s in skipped) if skipped else 'none'),
        }
        append_devlog(started.strftime('%Y-%m-%d %H:%M %Z'), summary,
                      blocks1, fracs, ctx)

    print('\nPHASE 1 DONE')


if __name__ == '__main__':
    main()
