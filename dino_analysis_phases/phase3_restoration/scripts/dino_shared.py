"""Shared DINOv2 spatial-feature extraction for Phase 3 (E0-Fixed / E1-N-Fixed).

ONE code path. Every Phase-3 consumer -- the layer re-check, the centering-mean
computation, the E1 architecture, the evaluation scripts -- must call the
functions in this module and nothing else. If the extraction ever needs to
change, it changes here, once.

SPEC (Work Order Reference A), all of it enforced or asserted below:

  model         DINOv2 `dinov2_vitb14` (ViT-B/14, embed dim 768, patch 14,
                12 blocks, 0 register tokens), loaded OFFLINE from the cached
                hub dir + .pth with strict=True, via the project's existing
                `DINOv2Extractor` (basicsr/models/archs/dinov2_feature_extractor.py).
  extraction    `dino.get_intermediate_layers(x, n=blocks0, reshape=False,
                return_class_token=False, norm=True)` -- the SAME call Phase 1
                and Phase 2 use. Patch tokens only; DINOv2 strips CLS and
                register tokens itself. No CLS, no pooling, no PCA, no
                multi-layer concatenation.
  preprocessing input float32 [0,1], [B,1|3,H,W]:
                  1. grayscale -> 3 channels by repeat
                  2. bilinear resize, align_corners=False, to 224 (train /
                     matched-128) or 448 (full-256 eval)
                  3. ImageNet normalisation, mean (0.485,0.456,0.406),
                     std (0.229,0.224,0.225)
                This is `dino_preprocess` from the project extractor, unchanged
                -- byte-for-byte the Phase-1/2 transform.
  grid          224/14 = 16 and 448/14 = 32, so radar-px-per-token is ~8 in both
                regimes and the DINO grid equals the Restormer latent grid
                (H/8 x W/8) directly. NEVER resize a 256 image to 224 and
                upsample the 16x16 token map -- `assert_no_interpolation_needed`
                exists to make that a hard error.
  freezing      requires_grad=False on every DINO parameter; `.eval()` is
                re-asserted by `DINOv2Extractor.train()` even when a parent
                module goes to train mode; extraction runs under torch.no_grad.
  centering     `center_tokens` subtracts ONE position-independent [768] vector.
                No per-position mean, no L2 normalisation, no extra LayerNorm.

BLOCK NUMBERING. `get_intermediate_layers` collects x AFTER blk(x), so
0-indexed 5 == the output of the 6th block == "B6". `b1_to_b0(6) == 5`.
It also returns outputs in ASCENDING block order regardless of the order the
indices are passed in (the Phase-2 bug, commit 4240521) -- `extract_blocks`
therefore keys its result by block index and never by position.
"""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from basicsr.models.archs.dinov2_feature_extractor import (  # noqa: E402
    DINOv2Extractor, dino_preprocess)

# --- fixed facts about this checkpoint (asserted at build time) -------------
EMBED_DIM = 768
PATCH_SIZE = 14
N_BLOCKS = 12

# the two scale-consistent regimes; nothing else is allowed
DINO_SIZE_TRAIN128 = 224     # 128 radar crop -> 224 -> 16x16 tokens
DINO_SIZE_EVAL256 = 448      # 256 radar image -> 448 -> 32x32 tokens

DEFAULT_HUB_DIR = os.path.join(
    _REPO, 'torch_hub', 'hub', 'facebookresearch_dinov2_main')
DEFAULT_WEIGHTS = os.path.join(
    _REPO, 'torch_hub', 'hub', 'checkpoints', 'dinov2_vitb14_pretrain.pth')
# the repo-relative cache does not exist; the real one lives on /home/woody and
# is named in Deraining_Holo/Options/DINO_analysis_data.yml. Resolved there.
_ANALYSIS_YML = os.path.join(
    _REPO, 'Deraining_Holo', 'Options', 'DINO_analysis_data.yml')


def dino_paths_from_yaml(path=None):
    """Read dino_hub_dir / dino_weights / dino_model_name / dino_hub_source
    from DINO_analysis_data.yml so Phase 3 uses the SAME checkpoint Phase 1/2
    used. Fails loudly rather than falling back to a download."""
    import yaml
    path = path or _ANALYSIS_YML
    with open(path) as f:
        cfg = yaml.safe_load(f)
    net = cfg['network_g']
    return {
        'model_name': net['dino_model_name'],
        'hub_source': net['dino_hub_source'],
        'hub_dir': net['dino_hub_dir'],
        'weights': net['dino_weights'],
    }


def b1_to_b0(block_1indexed):
    """'B6' -> 5. The only place this conversion is allowed to happen."""
    b0 = int(block_1indexed) - 1
    if not 0 <= b0 < N_BLOCKS:
        raise ValueError(f'block {block_1indexed} outside 1..{N_BLOCKS}')
    return b0


def build_dino(device='cpu', model_name=None, hub_source=None, hub_dir=None,
               weights=None, verbose=True):
    """Frozen, eval-mode DINOv2 ViT-B/14. Strict weight load, offline.

    Returns the project's DINOv2Extractor. Its `layers` argument is irrelevant
    here (it only drives the pooled forward(), which Phase 3 never calls); all
    Phase-3 extraction goes through `extract_blocks` below.
    """
    p = dino_paths_from_yaml()
    ext = DINOv2Extractor(
        layers=(0,),                       # unused; extract_blocks selects
        img_size=DINO_SIZE_TRAIN128,       # unused; sizes passed explicitly
        model_name=model_name or p['model_name'],
        hub_source=hub_source or p['hub_source'],
        hub_dir=hub_dir or p['hub_dir'],
        weights=weights or p['weights'],
        feat_mean=None)                    # centering is done by center_tokens

    # hard checks -- a wrong checkpoint must never slip through silently
    missing = list(getattr(ext.load_result, 'missing_keys', []) or [])
    unexpected = list(getattr(ext.load_result, 'unexpected_keys', []) or [])
    if missing or unexpected:
        raise RuntimeError(f'DINO state_dict mismatch: missing={missing} '
                           f'unexpected={unexpected}')
    nreg = int(getattr(ext.dino, 'num_register_tokens', 0))
    if nreg != 0:
        raise RuntimeError(f'expected 0 register tokens, got {nreg}')
    if int(ext.dino.embed_dim) != EMBED_DIM:
        raise RuntimeError(f'embed_dim {ext.dino.embed_dim} != {EMBED_DIM}')
    if len(ext.dino.blocks) != N_BLOCKS:
        raise RuntimeError(f'{len(ext.dino.blocks)} blocks != {N_BLOCKS}')
    if int(ext.dino.patch_embed.patch_size[0]) != PATCH_SIZE:
        raise RuntimeError('patch size != 14')

    for prm in ext.dino.parameters():
        if prm.requires_grad:
            raise RuntimeError('DINO parameter still requires grad')
    ext = ext.to(device).eval()
    if verbose:
        print(f'  DINO: {p["model_name"]} from {p["hub_dir"]}\n'
              f'        weights {p["weights"]}\n'
              f'        strict load OK (missing=[], unexpected=[]), '
              f'{N_BLOCKS} blocks, dim {EMBED_DIM}, patch {PATCH_SIZE}, '
              f'registers {nreg}, frozen+eval on {device}')
    return ext


def preprocess(img01, dino_size):
    """[B,1|3,H,W] float [0,1] -> [B,3,dino_size,dino_size] ImageNet-normalised.

    Delegates to the project's `dino_preprocess`, so this is exactly the
    Phase-1/2 transform: repeat to 3ch, bilinear/align_corners=False resize,
    (x-mean)/std.
    """
    if dino_size % PATCH_SIZE != 0:
        raise ValueError(f'dino_size {dino_size} not a multiple of {PATCH_SIZE}')
    dev = img01.device
    mean = torch.tensor((0.485, 0.456, 0.406), device=dev,
                        dtype=img01.dtype).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=dev,
                       dtype=img01.dtype).view(1, 3, 1, 1)
    return dino_preprocess(img01, dino_size, mean, std)


@torch.no_grad()
def extract_blocks(ext, img01, blocks0, dino_size):
    """radar [B,1|3,H,W] in [0,1] -> {block0: [B,N,768]} patch tokens.

    `blocks0` are 0-indexed block indices. The result is a DICT keyed by block
    index precisely because get_intermediate_layers always returns ascending
    order regardless of the argument order.
    """
    blocks0 = tuple(int(b) for b in blocks0)
    x = preprocess(img01, dino_size)
    feats = ext.dino.get_intermediate_layers(
        x, n=blocks0, reshape=False, return_class_token=False, norm=True)
    order = sorted(set(blocks0))                      # what DINOv2 actually returns
    if len(feats) != len(order):
        raise RuntimeError(f'{len(feats)} outputs for blocks {order}')
    grid = dino_size // PATCH_SIZE
    out = {}
    for b0, f in zip(order, feats):
        if f.shape[1] != grid * grid or f.shape[2] != EMBED_DIM:
            raise RuntimeError(f'block {b0}: got {tuple(f.shape)}, expected '
                               f'[B,{grid * grid},{EMBED_DIM}]')
        out[b0] = f
    return out


@torch.no_grad()
def extract_block(ext, img01, block0, dino_size):
    """Single-block convenience wrapper -> [B,N,768]."""
    return extract_blocks(ext, img01, (block0,), dino_size)[int(block0)]


def tokens_to_grid(tokens):
    """[B,N,768] -> [B,768,g,g], row-major (verified bit-identical to DINOv2's
    own reshape=True output, REPO_INVESTIGATION_REPORT section F)."""
    b, n, c = tokens.shape
    g = int(round(n ** 0.5))
    if g * g != n:
        raise ValueError(f'{n} tokens is not a square grid')
    return tokens.reshape(b, g, g, c).permute(0, 3, 1, 2).contiguous()


def center_tokens(tokens, mu):
    """D_centered[p] = D_raw[p] - mu, one position-independent [768] vector.

    No L2 normalisation before or after. No per-position mean. No extra
    LayerNorm (DINO's own norm=True is already applied).
    """
    mu = mu.reshape(-1)
    if mu.numel() != tokens.shape[-1]:
        raise ValueError(f'mean width {mu.numel()} != token width '
                         f'{tokens.shape[-1]}')
    return tokens - mu.to(device=tokens.device, dtype=tokens.dtype)


def assert_no_interpolation_needed(dino_grid_hw, latent_hw):
    """Hard gate: the DINO token grid must already equal the Restormer latent
    grid. Phase 3 never interpolates the feature grid -- if this fires, the
    image size / dino_size pairing is wrong."""
    if tuple(dino_grid_hw) != tuple(latent_hw):
        raise RuntimeError(
            f'DINO grid {tuple(dino_grid_hw)} != latent grid {tuple(latent_hw)}; '
            f'Phase 3 forbids feature-grid interpolation. Use radar 128 with '
            f'dino_size {DINO_SIZE_TRAIN128} or radar 256 with '
            f'{DINO_SIZE_EVAL256}.')
    return True


def dino_size_for(radar_size):
    """The ONLY sanctioned radar-size -> dino-size mapping: 8 radar px/token."""
    if radar_size % 8 != 0:
        raise ValueError(f'radar size {radar_size} not divisible by 8')
    tokens = radar_size // 8
    return tokens * PATCH_SIZE            # 128 -> 224, 256 -> 448
