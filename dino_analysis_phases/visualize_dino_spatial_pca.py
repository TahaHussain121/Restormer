"""Spatial DINO patch-token PCA: Very Noisy Radar (1e5) vs Clean Radar (1e7) vs Render.

QUESTION
    Does a frozen DINOv2 ViT-B/14 preserve a similar SPATIAL/structural
    representation when the same scene is observed as (1) an extremely noisy
    1e5-ray radar heatmap, (2) a clean 1e7-ray radar heatmap, and (3) the
    corresponding black-background render -- and how does that change from
    shallow to deep transformer blocks?

WHAT THIS IS NOT
    This is an ANALYSIS/VISUALISATION experiment. It does not touch the training
    pipeline: it imports the project's dataset class, the project's DINO
    extractor and the project's `dino_preprocess`, and writes only under
    dino_analysis/.

KEY DESIGN POINTS (each one is a sanity check the spec asked for)
  S1. Pairing comes from the project's own dataset class
      (`Dataset_PairedImage_uint16_Render`), which pairs LQ/GT by
      `paired_paths_from_folder` and derives the render path from the GT
      basename. The three basenames are asserted equal -- no filename guessing.
  S2. The DINO model is the project's: dinov2_vitb14 loaded OFFLINE from the
      cached hub dir + .pth named in the arm config, via `DINOv2Extractor`
      (strict=True load, so a silently-random ViT cannot slip through).
  S3. SPATIAL tokens only. `get_intermediate_layers(..., return_class_token=
      False)` returns `out[:, 1 + num_register_tokens:]`, i.e. CLS and register
      tokens are already stripped by DINOv2 itself; we assert
      num_register_tokens == 0 for this checkpoint and assert
      N_tokens == (img_size/patch)^2. NOTHING is pooled anywhere in this script.
  S4. Blocks at 1/4, 2/4, 3/4, 4/4 depth. For the 12-block ViT-B/14 that is
      1-indexed {3,6,9,12} -> 0-indexed {2,5,8,11}. NOTE this deliberately
      differs from the training recipe's {1,4,8,12}; the spec asked for quarter
      depths. Both index conventions are printed.
  S5. JOINT PCA, fitted independently PER LAYER on the concatenation of the
      three inputs' patch tokens, then the SAME transform applied to all three
      rows. One PCA per layer, never one PCA per image.
  S6. SHARED visualisation range per layer: one robust percentile range computed
      over the three rows' PCA values for that layer, applied to all three.
  S7. CENTERING. The repo's existing `dino_feat_mean_*.pt` are POOLED 3072-d
      vectors (mean over patch tokens, 4 layers concatenated) and are therefore
      NOT usable to centre 768-d spatial patch tokens. The script says so
      explicitly and refuses to use them. Instead it computes per-layer spatial
      means over TRAINING images (mean over images AND patch positions),
      cached to dino_analysis/dino_spatial_layer_means.pt.

CENTERING MODE
    --centering-mode per-domain (default): a separate mean per (domain, layer).
        Matches the project's established convention in
        Deraining_Holo/render_radar_similarity.py, which centred render features
        by the render mean and radar features by the radar mean. Removes the
        domain offset, so what is left is object/spatial structure -- which is
        exactly what the scientific question asks about.
    --centering-mode global: one mean per layer, pooled over all three domains.
        This is the literal reading of "one mean_L per layer". It leaves the
        radar-vs-render domain offset inside the residual.
    Both means are computed and cached; the CSV reports similarities for raw,
    per-domain-centered and global-centered. The `_centered` FIGURES use the
    mode selected by --centering-mode (recorded in the metadata JSON).

USAGE
    PY=/home/woody/iwnt/iwnt174h/thesis_dino/code/venv/bin/python
    export TORCH_HOME=/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub

    # first run (no cached spatial means yet):
    $PY dino_analysis/visualize_dino_spatial_pca.py \
        --sample-id 0196 --device cpu \
        --compute-centered --compute-layer-means --num-mean-samples 150

    # later runs reuse the cache:
    $PY dino_analysis/visualize_dino_spatial_pca.py \
        --sample-id 0196 --device cpu --compute-centered
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sklearn.decomposition import PCA

from basicsr.data.radar_render_triplet_dataset import (
    Dataset_PairedImage_uint16_Render)
from basicsr.models.archs.dinov2_feature_extractor import (
    DINOv2Extractor, dino_preprocess, dino_denormalize)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'outputs')
DEFAULT_OPT = os.path.join(
    _REPO, 'Deraining_Holo', 'Options', 'DINO_analysis_data.yml')
DEFAULT_MEANS = os.path.join(HERE, 'dino_spatial_layer_means.pt')
# The E1 pooled means. E1 itself was removed, but these two vectors are kept as
# reference inputs: report_pooled_mean_incompatibility() reads them to record WHY
# spatial centering cannot reuse them. Moved out of the deleted exp3_dino_film/.
POOLED_MEANS = [
    os.path.join(_REPO, 'Deraining_Holo', 'experiment_results',
                 'dino_pooled_means_reference',
                 f'dino_feat_mean_{a}.pt') for a in ('lqDINO', 'renderDINO')]

# Row identity. Keys are used everywhere (features, means, CSV); labels are the
# terminology the spec fixed and must not drift into LR/HR.
DOMAINS = ['1e5', '1e7', 'render']
DOMAIN_LABEL = {
    '1e5': 'Very Noisy Radar\n(1e5 rays)',
    '1e7': 'Clean Radar\n(1e7 rays)',
    'render': 'Render',
}
# which key of the project's dataset batch each row reads
DOMAIN_BATCH_KEY = {'1e5': 'lq', '1e7': 'gt', 'render': 'dino'}


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def build_dataset(cfg, split):
    """The project's own paired dataset, in val phase (no crop, full 256x256).

    Val phase is used even for the train split when computing centering means:
    the analysed sample is a full image, so the mean must be over full images
    too, not over random crops.
    """
    opt = copy.deepcopy(cfg['datasets']['val'])
    for k in ('dataroot_gt', 'dataroot_lq', 'dataroot_render'):
        opt[k] = opt[k].replace('/val_', f'/{split}_')
    opt['phase'] = 'val'
    opt['scale'] = 1
    opt.setdefault('filename_tmpl', '{}')
    return Dataset_PairedImage_uint16_Render(opt)


def pick_sample(ds, sample_id):
    """Resolve --sample-id to a dataset index, and verify the 3-way pairing."""
    stems = [os.path.splitext(os.path.basename(p['gt_path']))[0] for p in ds.paths]
    if sample_id is None:
        idx = 0
    else:
        want = os.path.splitext(str(sample_id))[0]
        if want not in stems:
            raise SystemExit(
                f'sample id {sample_id!r} not in this split '
                f'({len(stems)} images, e.g. {stems[:5]})')
        idx = stems.index(want)

    batch = ds[idx]
    paths = {
        '1e5': batch['lq_path'],
        '1e7': batch['gt_path'],
        'render': ds._render_path(batch['gt_path']),
    }
    # S1: same scene, verified programmatically, not by eyeballing filenames.
    bases = {k: os.path.basename(v) for k, v in paths.items()}
    if len(set(bases.values())) != 1:
        raise SystemExit(f'PAIRING BROKEN -- basenames differ: {bases}')
    for k, v in paths.items():
        if not os.path.isfile(v):
            raise SystemExit(f'missing {k} file: {v}')
    return idx, stems[idx], paths, batch


def batch_to_dino_inputs(batch, img_size, mean, std):
    """Project preprocessing -> the exact tensors DINO is fed, per row.

    1e5 / 1e7: uint16 PNG /65535 -> [1,1,H,W] in [0,1] -> grayscale replicated to
      3 channels, resized to img_size, ImageNet-normalised. This is the lqDINO
      path (basicsr.../dinov2_feature_extractor.dino_preprocess), unchanged. No
      colormap is applied before DINO.
    render: 8-bit PNG -> [0,1] 3-channel -> same resize + ImageNet norm. This is
      the renderDINO path, unchanged.
    """
    out = {}
    for d in DOMAINS:
        img = batch[DOMAIN_BATCH_KEY[d]].unsqueeze(0)      # [1,C,H,W] in [0,1]
        out[d] = dino_preprocess(img, img_size, mean, std)  # [1,3,S,S]
    return out


# ----------------------------------------------------------------------------
# DINO
# ----------------------------------------------------------------------------
def build_extractor(cfg, blocks0, device):
    net = cfg['network_g']
    ext = DINOv2Extractor(
        layers=tuple(blocks0), img_size=net['dino_img_size'],
        model_name=net['dino_model_name'], hub_source=net['dino_hub_source'],
        hub_dir=net['dino_hub_dir'], weights=net['dino_weights'],
        feat_mean=None)                       # uncentered: we centre ourselves
    return ext.to(device).eval()


@torch.no_grad()
def spatial_tokens(ext, x, blocks0):
    """[B,3,S,S] preprocessed -> list over layers of [B, N_patches, C].

    S3: DINOv2 strips CLS + registers internally when return_class_token=False.
    """
    feats = ext.dino.get_intermediate_layers(
        x, n=tuple(blocks0), reshape=False, return_class_token=False, norm=True)
    return [f.float().cpu() for f in feats]


# ----------------------------------------------------------------------------
# centering statistics
# ----------------------------------------------------------------------------
def report_pooled_mean_incompatibility(embed_dim):
    """S7. The repo already has DINO means -- state exactly what they are."""
    found = [p for p in POOLED_MEANS if os.path.isfile(p)]
    info = []
    for p in found:
        obj = torch.load(p, map_location='cpu', weights_only=False)
        meta = obj.get('meta', {}) if isinstance(obj, dict) else {}
        mu = obj['mean'] if isinstance(obj, dict) else obj
        info.append({'path': p, 'width': int(mu.numel()),
                     'arm': meta.get('arm'), 'layers': meta.get('dino_layers'),
                     'n_crops': meta.get('n_crops')})
        print(f'  found existing mean: {os.path.relpath(p, _REPO)}  '
              f'width {mu.numel()}  arm {meta.get("arm")}  '
              f'layers {meta.get("dino_layers")}  n={meta.get("n_crops")} crops')
    if found:
        print('Existing pooled DINO mean is not directly compatible with '
              'spatial patch-token centering.')
        print(f'  reason: those vectors are {info[0]["width"]}-d = 4 layers x '
              f'{embed_dim} MEAN-POOLED over patch tokens (one vector per image);'
              f' spatial centering needs a {embed_dim}-d mean PER LAYER over '
              f'images AND patch positions. They are NOT used here.')
    else:
        print('  no existing pooled DINO mean found on disk.')
    return info


@torch.no_grad()
def compute_layer_means(ext, cfg, blocks0, n_samples, seed, device, img_size):
    """Per-(domain, layer) and global-per-layer spatial patch-token means.

    mean = average over TRAINING images and over all patch positions -> [C].
    """
    ds = build_dataset(cfg, 'train')
    rng = random.Random(seed)
    idxs = sorted(rng.sample(range(len(ds)), min(n_samples, len(ds))))
    print(f'  training pool: {len(ds)} images (train split); '
          f'using {len(idxs)} of them (seed {seed})')

    sums = {d: [torch.zeros(ext.embed_dim, dtype=torch.float64)
                for _ in blocks0] for d in DOMAINS}
    counts = {d: 0 for d in DOMAINS}
    for n, i in enumerate(idxs, 1):
        batch = ds[i]
        xs = batch_to_dino_inputs(batch, img_size, ext.mean.cpu(), ext.std.cpu())
        for d in DOMAINS:
            feats = spatial_tokens(ext, xs[d].to(device), blocks0)
            for li, f in enumerate(feats):
                sums[d][li] += f[0].double().sum(0)      # sum over patches
            counts[d] += feats[0].shape[1]
        if n % 25 == 0 or n == len(idxs):
            print(f'    {n}/{len(idxs)} training images done')

    per_domain = {d: torch.stack([s / counts[d] for s in sums[d]]).float()
                  for d in DOMAINS}                        # [L, C] per domain
    glob = torch.stack([
        sum(sums[d][li] for d in DOMAINS) / sum(counts.values())
        for li in range(len(blocks0))]).float()            # [L, C]
    return per_domain, glob, [ds.paths[i]['gt_path'] for i in idxs]


# ----------------------------------------------------------------------------
# PCA
# ----------------------------------------------------------------------------
def joint_pca(feats_by_domain, n_components, seed):
    """S5. ONE PCA per layer, fitted on 1e5 + 1e7 + render patch tokens jointly,
    then applied unchanged to each of the three."""
    joint = np.concatenate([feats_by_domain[d] for d in DOMAINS], axis=0)
    pca = PCA(n_components=n_components, svd_solver='full', random_state=seed)
    pca.fit(joint)
    proj = {d: pca.transform(feats_by_domain[d]) for d in DOMAINS}
    return pca, proj


def shared_range(proj, comp, lo, hi):
    """S6. One robust percentile range per layer, shared by all three rows."""
    vals = np.concatenate([proj[d][:, comp] for d in DOMAINS])
    return float(np.percentile(vals, lo)), float(np.percentile(vals, hi))


def to_map(vec, gh, gw):
    return vec.reshape(gh, gw)


def upsample(m, size):
    t = torch.from_numpy(np.ascontiguousarray(m)).float()[None, None]
    return torch.nn.functional.interpolate(
        t, size=(size, size), mode='bilinear', align_corners=False)[0, 0].numpy()


# ----------------------------------------------------------------------------
# similarity
# ----------------------------------------------------------------------------
def patchwise_cosine(a, b):
    """Mean cosine similarity between CORRESPONDING patches (same (i,j))."""
    a = torch.from_numpy(a).float()
    b = torch.from_numpy(b).float()
    return float(torch.nn.functional.cosine_similarity(a, b, dim=1).mean())


PAIRS = [('1e5', '1e7'), ('1e5', 'render'), ('1e7', 'render')]


@torch.no_grad()
def control_features(ext, ds, blocks0, idx_self, n_control, seed, device, img_size):
    """DIFFERENT-SCENE control: the same measurement against OTHER objects.

    Without this, a corresponding-patch cosine of e.g. +0.52 is uninterpretable:
    DINO patch tokens of any two images of this dataset share a large common
    component, so a high number does not by itself mean the scene structure was
    preserved. Only the gap (same scene - different scene) does.
    """
    rng = random.Random(seed + 1)
    pool = [i for i in range(len(ds)) if i != idx_self]
    idxs = rng.sample(pool, min(n_control, len(pool)))
    out = []
    for i in idxs:
        b = ds[i]
        xs = batch_to_dino_inputs(b, img_size, ext.mean.cpu(), ext.std.cpu())
        out.append({d: [f[0].numpy() for f in
                        spatial_tokens(ext, xs[d].to(device), blocks0)]
                    for d in DOMAINS})
    return out, [os.path.splitext(os.path.basename(
        ds.paths[i]['gt_path']))[0] for i in idxs]


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------
def input_panel(x, mean, std):
    """Denormalised DINO input, exactly as the model saw it (so preprocessing
    mistakes are visible in the figure)."""
    img = dino_denormalize(x, mean, std)[0].clamp(0, 1).numpy()
    return np.transpose(img, (1, 2, 0))


def make_figure(path, title, inputs, maps, ranges, col_titles, cmap, disp,
                mean, std, dpi, rgb=False):
    nrow, ncol = len(DOMAINS), 1 + len(col_titles)
    # RGB panels carry no colorbar row -- do not reserve the strip for them.
    cb_h, cb_pad = (0.0, 0.012) if rgb else (0.055, 0.078)
    fig = plt.figure(figsize=(2.05 * ncol + 1.15,
                              2.05 * nrow + (1.25 if not rgb else 1.0)))
    gs = GridSpec(nrow + 1, ncol, figure=fig,
                  height_ratios=[1] * nrow + [max(cb_h, 1e-6)],
                  left=0.115, right=0.995, top=0.875, bottom=cb_pad,
                  wspace=0.045, hspace=0.045)

    for r, d in enumerate(DOMAINS):
        ax = fig.add_subplot(gs[r, 0])
        ax.imshow(input_panel(inputs[d], mean, std), interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_ylabel(DOMAIN_LABEL[d], fontsize=9, rotation=0,
                      ha='right', va='center', labelpad=10)
        if r == 0:
            ax.set_title('Input\n(as fed to DINO)', fontsize=9, pad=6)

        for c, ct in enumerate(col_titles, start=1):
            ax = fig.add_subplot(gs[r, c])
            m = maps[c - 1][d]
            if rgb:
                ax.imshow(m, interpolation='bilinear')
            else:
                vmin, vmax = ranges[c - 1]
                ax.imshow(upsample(m, disp), cmap=cmap, vmin=vmin, vmax=vmax,
                          interpolation='bilinear')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(ct, fontsize=9, pad=6)

    if not rgb:
        for c in range(len(col_titles)):
            host = fig.add_subplot(gs[nrow, c + 1])
            host.axis('off')
            # inset so the end labels of neighbouring colorbars cannot collide
            cax = host.inset_axes([0.16, 0.0, 0.68, 1.0])
            vmin, vmax = ranges[c]
            cb = fig.colorbar(
                plt.cm.ScalarMappable(
                    norm=plt.Normalize(vmin, vmax), cmap=cmap),
                cax=cax, orientation='horizontal')
            cb.set_ticks([])
            cb.outline.set_linewidth(0.4)
            host.text(0.14, 0.5, f'{vmin:.0f}', transform=host.transAxes,
                      fontsize=6.5, ha='right', va='center')
            host.text(0.86, 0.5, f'{vmax:.0f}', transform=host.transAxes,
                      fontsize=6.5, ha='left', va='center')
        fig.text(0.5, 0.010, 'PCA-1 (shared range per layer across all three rows)',
                 fontsize=7.5, ha='center')

    fig.suptitle(title, fontsize=9.5, y=0.988, linespacing=1.35)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f'  wrote {os.path.relpath(path, _REPO)}')


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--opt', default=DEFAULT_OPT,
                    help='arm config the DINO model/paths are read from')
    ap.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    ap.add_argument('--sample-id', default='0196',
                    help='image stem, e.g. 0196 (must exist in --split)')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--compute-centered', action='store_true',
                    help='also produce the centered figures/similarities')
    ap.add_argument('--compute-layer-means', action='store_true',
                    help='(re)compute the spatial per-layer training means')
    ap.add_argument('--num-mean-samples', type=int, default=150)
    ap.add_argument('--means-cache', default=DEFAULT_MEANS)
    ap.add_argument('--centering-mode', default='per-domain',
                    choices=['per-domain', 'global'])
    ap.add_argument('--percentile', type=float, nargs=2, default=(1.0, 99.0),
                    metavar=('LO', 'HI'),
                    help='robust shared display range percentiles')
    ap.add_argument('--display-size', type=int, default=224)
    ap.add_argument('--cmap', default='viridis')
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-control', type=int, default=12,
                    help='different-scene control scenes for the similarity '
                         'table (0 disables); makes the cosines interpretable')
    ap.add_argument('--no-rgb', action='store_true',
                    help='skip the optional PCA-RGB figures')
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(OUT_DIR, exist_ok=True)
    started = datetime.datetime.now().astimezone()

    with open(args.opt) as f:
        cfg = yaml.safe_load(f)
    net = cfg['network_g']

    # ---------------- sample ----------------
    print('=' * 78)
    print('SAMPLE (pairing via the project dataset class '
          'Dataset_PairedImage_uint16_Render)')
    print('=' * 78)
    ds = build_dataset(cfg, args.split)
    idx, sample_id, paths, batch = pick_sample(ds, args.sample_id)
    print(f'  split          : {args.split} ({len(ds)} images), index {idx}')
    print(f'  sample ID      : {sample_id}')
    print(f'  1e5  (verynoisy/Very Noisy Radar) : {paths["1e5"]}')
    print(f'  1e7  (clean/Clean Radar)          : {paths["1e7"]}')
    print(f'  render                            : {paths["render"]}')
    print('  pairing verified: all three basenames identical '
          f'({os.path.basename(paths["1e5"])})')

    # ---------------- model ----------------
    print()
    print('=' * 78)
    print('DINO MODEL (as used by this project, offline)')
    print('=' * 78)
    # built with a placeholder layer set; the real block set needs the depth,
    # which is only known once the checkpoint is loaded. ext.layers is set below.
    ext = build_extractor(cfg, (0,), args.device)
    n_blocks = len(ext.dino.blocks)
    patch = int(ext.dino.patch_size)
    img_size = int(net['dino_img_size'])
    embed_dim = int(ext.dino.embed_dim)
    n_reg = int(getattr(ext.dino, 'num_register_tokens', 0))
    gh = gw = img_size // patch

    # S4. quarter depths, 1-indexed -> 0-indexed
    blocks1 = [int(round(n_blocks * k / 4)) for k in (1, 2, 3, 4)]
    blocks0 = [b - 1 for b in blocks1]
    fracs = ['1/4', '2/4', '3/4', '4/4']

    print(f'  model name     : {net["dino_model_name"]}')
    print(f'  hub source     : {net["dino_hub_source"]}')
    print(f'  repo dir       : {net["dino_hub_dir"]}')
    print(f'  checkpoint     : {net["dino_weights"]}')
    print(f'  strict load    : missing={list(ext.load_result.missing_keys)} '
          f'unexpected={list(ext.load_result.unexpected_keys)}')
    print(f'  transformer blocks : {n_blocks}')
    print(f'  embedding dim      : {embed_dim}')
    print(f'  patch size         : {patch}')
    print(f'  register tokens    : {n_reg}')
    print(f'  DINO input res     : {img_size} x {img_size}')
    print(f'  patch grid         : {gh} x {gw} = {gh * gw} patch tokens')
    print(f'  selected blocks    : 1-indexed {blocks1}  ->  0-indexed {blocks0}'
          f'   (depths {", ".join(fracs)})')
    print(f'  NOTE: the TRAINING recipe uses layers 1-indexed [1,4,8,12]; this '
          f'analysis uses quarter depths {blocks1} as specified for this study.')
    if n_reg != 0:
        raise SystemExit(f'unexpected register tokens ({n_reg}) -- token '
                         f'layout must be re-checked before trusting the maps')
    ext.layers = tuple(blocks0)

    # ---------------- features ----------------
    print()
    print('=' * 78)
    print('SPATIAL PATCH TOKENS')
    print('=' * 78)
    inputs = batch_to_dino_inputs(batch, img_size, ext.mean.cpu(), ext.std.cpu())
    for d in DOMAINS:
        print(f'  {d:6s} DINO input tensor {tuple(inputs[d].shape)} '
              f'range [{inputs[d].min():+.3f}, {inputs[d].max():+.3f}]')

    raw = {}          # raw[domain] = list over layers of [N, C]
    for d in DOMAINS:
        fs = spatial_tokens(ext, inputs[d].to(args.device), blocks0)
        raw[d] = [f[0].numpy() for f in fs]
    n_tok = raw['1e5'][0].shape[0]
    # S3 assertion
    if n_tok != gh * gw:
        raise SystemExit(f'N_tokens {n_tok} != H_patches*W_patches {gh * gw} -- '
                         f'token structure must be investigated, not guessed')
    print(f'  token tensor per layer : [1, {n_tok}, {raw["1e5"][0].shape[1]}]  '
          f'(CLS + {n_reg} register tokens already removed by DINOv2)')
    print(f'  check N_tokens == H_patches * W_patches : '
          f'{n_tok} == {gh} * {gw}  OK')
    print('  no pooling of any kind is applied to these features.')

    # ---------------- centering statistics ----------------
    print()
    print('=' * 78)
    print('CENTERING STATISTICS')
    print('=' * 78)
    pooled_info = report_pooled_mean_incompatibility(embed_dim)

    means = None
    means_meta = {}
    if args.compute_centered:
        if args.compute_layer_means or not os.path.isfile(args.means_cache):
            if not args.compute_layer_means:
                print(f'  no cache at {args.means_cache} -- computing it now')
            print(f'  computing spatial per-layer means over training images '
                  f'(n={args.num_mean_samples}) ...')
            per_dom, glob, used = compute_layer_means(
                ext, cfg, blocks0, args.num_mean_samples, args.seed,
                args.device, img_size)
            means = {'per_domain': per_dom, 'global': glob,
                     'meta': {
                         'blocks_1indexed': blocks1, 'blocks_0indexed': blocks0,
                         'embed_dim': embed_dim, 'img_size': img_size,
                         'patch_size': patch, 'patch_grid': [gh, gw],
                         'n_train_images': len(used), 'seed': args.seed,
                         'split': 'train', 'phase': 'val (full 256x256, no crop)',
                         'model': net['dino_model_name'],
                         'weights': net['dino_weights'],
                         'statistic': 'mean over training images AND patch '
                                      'positions of the spatial patch tokens',
                         'created': datetime.datetime.now().astimezone().isoformat(
                             timespec='seconds'),
                         'script': 'dino_analysis/visualize_dino_spatial_pca.py',
                         'example_files': [os.path.basename(p) for p in used[:5]],
                     }}
            torch.save(means, args.means_cache)
            print(f'  saved -> {os.path.relpath(args.means_cache, _REPO)}')
        else:
            means = torch.load(args.means_cache, map_location='cpu',
                               weights_only=False)
            print(f'  loaded cached spatial means from '
                  f'{os.path.relpath(args.means_cache, _REPO)}')
        means_meta = means['meta']
        if (means_meta['blocks_0indexed'] != blocks0
                or means_meta['embed_dim'] != embed_dim):
            raise SystemExit('cached spatial means were built for a different '
                             'block set / model -- rerun with '
                             '--compute-layer-means')
        print(f'  statistic      : {means_meta["statistic"]}')
        print(f'  source         : {means_meta["n_train_images"]} TRAIN-split '
              f'images, seed {means_meta["seed"]}')
        for li, b in enumerate(blocks1):
            nrm = {d: float(means['per_domain'][d][li].norm()) for d in DOMAINS}
            print(f'    block {b:2d}: ||mean|| per-domain '
                  + '  '.join(f'{d}={nrm[d]:.2f}' for d in DOMAINS)
                  + f'   global={float(means["global"][li].norm()):.2f}')
        print(f'  centering mode for the FIGURES : {args.centering_mode}')

    # ---------------- feature variants ----------------
    ctrl_raw, ctrl_ids = [], []
    if args.n_control > 0:
        print()
        print('=' * 78)
        print('DIFFERENT-SCENE CONTROL')
        print('=' * 78)
        print(f'  extracting {args.n_control} other scenes from the '
              f'{args.split} split (seed {args.seed + 1}) ...')
        ctrl_raw, ctrl_ids = control_features(
            ext, ds, blocks0, idx, args.n_control, args.seed, args.device, img_size)
        print(f'  control scenes: {", ".join(ctrl_ids)}')

    variants = {'raw': raw}
    if means is not None:
        variants['centered_per_domain'] = {
            d: [raw[d][li] - means['per_domain'][d][li].numpy()
                for li in range(len(blocks0))] for d in DOMAINS}
        variants['centered_global'] = {
            d: [raw[d][li] - means['global'][li].numpy()
                for li in range(len(blocks0))] for d in DOMAINS}
    fig_centered_key = ('centered_per_domain' if args.centering_mode == 'per-domain'
                        else 'centered_global')

    # ---------------- PCA + similarity ----------------
    col_titles = [f'{f}-depth DINO\n(Block {b})' for f, b in zip(fracs, blocks1)]
    lo, hi = args.percentile
    csv_rows = []
    sim_tables = {}
    pca_meta = {}

    for vkey, feats in variants.items():
        pc1_maps, pc1_ranges, rgb_maps, evr = [], [], [], []
        rows = []
        for li in range(len(blocks0)):
            per_domain = {d: feats[d][li] for d in DOMAINS}
            pca, proj = joint_pca(per_domain, 3, args.seed)
            evr.append([float(v) for v in pca.explained_variance_ratio_])

            vmin, vmax = shared_range(proj, 0, lo, hi)
            pc1_ranges.append((vmin, vmax))
            pc1_maps.append({d: to_map(proj[d][:, 0], gh, gw) for d in DOMAINS})

            # RGB: PC1/2/3, each normalised over the JOINT values of that layer
            rng3 = [shared_range(proj, c, lo, hi) for c in range(3)]
            rgbm = {}
            for d in DOMAINS:
                chans = [np.clip((proj[d][:, c] - rng3[c][0])
                                 / max(rng3[c][1] - rng3[c][0], 1e-12), 0, 1)
                         for c in range(3)]
                rgbm[d] = np.stack([to_map(c, gh, gw) for c in chans], axis=-1)
            rgb_maps.append(rgbm)

            s = {f'similarity_{a}_vs_{b}': patchwise_cosine(feats[a][li],
                                                            feats[b][li])
                 for a, b in PAIRS}
            # different-scene control, centered the same way as `feats`
            for a, b in PAIRS:
                vals = []
                for c in ctrl_raw:
                    cb = c[b][li]
                    if vkey == 'centered_per_domain':
                        cb = cb - means['per_domain'][b][li].numpy()
                    elif vkey == 'centered_global':
                        cb = cb - means['global'][li].numpy()
                    vals.append(patchwise_cosine(feats[a][li], cb))
                s[f'control_{a}_vs_{b}'] = float(np.mean(vals)) if vals else float('nan')
                s[f'gap_{a}_vs_{b}'] = (s[f'similarity_{a}_vs_{b}']
                                        - s[f'control_{a}_vs_{b}'])
            rows.append(s)
            csv_rows.append({
                'feature_type': vkey, 'layer_fraction': fracs[li],
                'block_index': blocks1[li], 'block_index_0based': blocks0[li],
                **{k: f'{v:.4f}' for k, v in s.items()},
                'n_control_scenes': len(ctrl_raw),
                'pca1_vmin': f'{vmin:.4f}', 'pca1_vmax': f'{vmax:.4f}',
                'pc1_explained_var_ratio': f'{evr[-1][0]:.4f}',
            })
        sim_tables[vkey] = rows
        pca_meta[vkey] = {'explained_variance_ratio_pc123': evr,
                          'pc1_shared_range': pc1_ranges}

        if vkey == 'raw':
            tag, human = 'raw', 'RAW spatial DINO features'
        elif vkey == fig_centered_key:
            tag, human = 'centered', (
                f'CENTERED spatial DINO features ({args.centering_mode} '
                f'training mean, n={means_meta.get("n_train_images")})')
        else:
            continue   # the non-selected centering mode: CSV only, no figure

        sub = (f'{human}   |   sample {sample_id} ({args.split} split)\n'
               f'{net["dino_model_name"]}, {gh}x{gw} patch grid   |   one joint '
               f'PCA per layer over all three inputs, shared display range per layer')
        make_figure(
            os.path.join(OUT_DIR, f'dino_spatial_pca_1e5_1e7_render_{tag}.png'),
            'Spatial DINO PCA-1 — Very Noisy Radar (1e5) vs Clean Radar (1e7) '
            f'vs Render\n{sub}',
            inputs, pc1_maps, pc1_ranges, col_titles, args.cmap,
            args.display_size, ext.mean.cpu(), ext.std.cpu(), args.dpi)

        if not args.no_rgb:
            make_figure(
                os.path.join(OUT_DIR,
                             f'dino_spatial_pca_rgb_1e5_1e7_render_{tag}.png'),
                'Spatial DINO PCA-RGB (PC1,PC2,PC3 → R,G,B) — Very Noisy Radar '
                f'(1e5) vs Clean Radar (1e7) vs Render\n{sub}',
                inputs, rgb_maps, None, col_titles, args.cmap,
                args.display_size, ext.mean.cpu(), ext.std.cpu(), args.dpi,
                rgb=True)

    # ---------------- tables ----------------
    print()
    print('=' * 78)
    print('MEAN CORRESPONDING-PATCH COSINE SIMILARITY')
    print('=' * 78)
    for vkey, rows in sim_tables.items():
        print(f'\n  [{vkey}]   SAME scene (and, in brackets, the '
              f'{len(ctrl_raw)}-scene DIFFERENT-scene control)')
        print('  Layer | Block |     1e5 vs 1e7 |  1e5 vs Render |  1e7 vs Render')
        print('  ' + '-' * 68)
        for f, b, s in zip(fracs, blocks1, rows):
            cells = ''.join(
                f' | {s[f"similarity_{a}_vs_{c}"]:+.4f} [{s[f"control_{a}_vs_{c}"]:+.3f}]'
                for a, c in PAIRS)
            print(f'  {f:5s} | {b:5d}{cells}')
        print('  gap (same scene - different scene):')
        for f, b, s in zip(fracs, blocks1, rows):
            cells = ''.join(f' | {s[f"gap_{a}_vs_{c}"]:+14.4f}' for a, c in PAIRS)
            print(f'  {f:5s} | {b:5d}{cells}')

    csv_path = os.path.join(OUT_DIR, 'dino_spatial_similarity_1e5_1e7_render.csv')
    cols = ['feature_type', 'layer_fraction', 'block_index', 'block_index_0based',
            'similarity_1e5_vs_1e7', 'similarity_1e5_vs_render',
            'similarity_1e7_vs_render',
            'control_1e5_vs_1e7', 'control_1e5_vs_render', 'control_1e7_vs_render',
            'gap_1e5_vs_1e7', 'gap_1e5_vs_render', 'gap_1e7_vs_render',
            'n_control_scenes', 'pca1_vmin', 'pca1_vmax',
            'pc1_explained_var_ratio']
    with open(csv_path, 'w') as f:
        f.write(','.join(cols) + '\n')
        for r in csv_rows:
            f.write(','.join(str(r[c]) for c in cols) + '\n')
    print(f'\n  wrote {os.path.relpath(csv_path, _REPO)}')

    # ---------------- metadata ----------------
    meta = {
        'execution_date': started.strftime('%Y-%m-%d'),
        'execution_local_time': started.strftime('%H:%M:%S'),
        'timezone': started.tzname(),
        'utc_offset': started.strftime('%z'),
        'command': ' '.join([os.path.relpath(sys.argv[0], _REPO)] + sys.argv[1:]),
        'sample_id': sample_id,
        'split': args.split,
        'dataset_index': idx,
        'pairing_source': 'basicsr.data.radar_render_triplet_dataset.'
                          'Dataset_PairedImage_uint16_Render (val phase, no crop)',
        'path_1e5_very_noisy_radar': paths['1e5'],
        'path_1e7_clean_radar': paths['1e7'],
        'path_render': paths['render'],
        'config_read': os.path.relpath(args.opt, _REPO),
        'dino_model': net['dino_model_name'],
        'dino_hub_source': net['dino_hub_source'],
        'dino_hub_dir': net['dino_hub_dir'],
        'dino_checkpoint': net['dino_weights'],
        'dino_load_strict': True,
        'transformer_block_count': n_blocks,
        'selected_blocks_1indexed': blocks1,
        'selected_blocks_0indexed': blocks0,
        'layer_fractions': fracs,
        'training_recipe_blocks_1indexed': [1, 4, 8, 12],
        'embedding_dim': embed_dim,
        'patch_size': patch,
        'num_register_tokens': n_reg,
        'dino_input_resolution': [img_size, img_size],
        'patch_grid': [gh, gw],
        'n_patch_tokens': int(n_tok),
        'preprocessing': {
            'radar_1e5_1e7': 'uint16 PNG / 65535 -> [0,1] 1ch -> replicate to 3ch '
                             '-> bilinear resize to 224 -> ImageNet normalise '
                             '(basicsr dinov2_feature_extractor.dino_preprocess)',
            'render': '8-bit PNG -> [0,1] 3ch -> bilinear resize to 224 -> '
                      'ImageNet normalise (same function)',
            'colormap_before_dino': False,
        },
        'pca': {
            'library': 'sklearn.decomposition.PCA (svd_solver=full)',
            'n_components': 3,
            'fit': 'ONE PCA per layer, fitted on the concatenation of the 1e5, '
                   '1e7 and render patch tokens (3 x %d rows), then applied '
                   'unchanged to each of the three' % n_tok,
            'per_layer_independent': True,
            'explained_variance_ratio_pc123': {
                k: v['explained_variance_ratio_pc123'] for k, v in pca_meta.items()},
        },
        'visualization_normalization': {
            'method': f'robust shared percentile range, p{lo}-p{hi} of the JOINT '
                      f'PCA values of that layer over all three inputs; the same '
                      f'(vmin, vmax) is applied to all three rows',
            'percentiles': [lo, hi],
            'pc1_shared_range_per_layer': {
                k: v['pc1_shared_range'] for k, v in pca_meta.items()},
            'colormap': args.cmap,
            'upsampling': f'bilinear {gh}x{gw} -> {args.display_size}x'
                          f'{args.display_size} for display only',
            'rgb': 'PC1/PC2/PC3 -> R/G/B, each channel normalised over that '
                   'layer\'s joint values with the same robust percentiles',
        },
        'centering': {
            'applied': bool(means is not None),
            'mode_used_for_figures': args.centering_mode if means else None,
            'method': 'centered_patch = patch_feature - mean_L, applied BEFORE PCA',
            'statistics_source': (
                f'{means_meta.get("n_train_images")} images from the TRAIN split, '
                f'full 256x256 (no crop), seed {means_meta.get("seed")}; mean over '
                f'images AND patch positions, one [C] vector per selected block'
                if means else None),
            'n_training_images': means_meta.get('n_train_images'),
            'cache_file': (os.path.relpath(args.means_cache, _REPO)
                           if means else None),
            'existing_pooled_means_examined': pooled_info,
            'existing_pooled_means_compatible': False,
            'existing_pooled_means_note':
                'Existing pooled DINO mean is not directly compatible with '
                'spatial patch-token centering: those are 3072-d vectors '
                '(4 layers x 768, mean-pooled over patch tokens); spatial '
                'centering needs a 768-d mean per layer over images and patch '
                'positions. They were NOT used.',
        },
        'similarity_metric': 'mean cosine similarity between corresponding '
                             'patches (same (i,j)), averaged over all patches',
        'different_scene_control': {
            'n_scenes': len(ctrl_raw),
            'scene_ids': ctrl_ids,
            'definition': 'same measurement, but the second image is taken from '
                          'a DIFFERENT object of the same split; averaged over '
                          'the control scenes. gap = same-scene - different-scene.',
            'why': 'DINO patch tokens on this dataset share a large common '
                   'component, so an absolute cosine is not interpretable on its '
                   'own; only the same-vs-different gap is.',
        },
        'random_seed': args.seed,
        'device': args.device,
        'torch_version': torch.__version__,
        'outputs': sorted(os.listdir(OUT_DIR)),
    }
    meta_path = os.path.join(OUT_DIR, 'dino_spatial_analysis_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'  wrote {os.path.relpath(meta_path, _REPO)}')
    print('\nDONE')


if __name__ == '__main__':
    main()
