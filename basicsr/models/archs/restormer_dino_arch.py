## Frozen DINOv2 ViT-B/14 feature extractor.
##
## HISTORY: this file also held the E1 FiLM-guidance experiment (RestormerDINO,
## FiLMHead, _film, _StubExtractor) -- a Restormer subclass that modulated the
## bottleneck and decoder stages with pooled DINO features. E1 failed three
## times (FiLM runaway; see experiment_results/exp3_dino_film/E1_DINO_report.md)
## and that code was removed. The full E1 tree is preserved on the `dino_prior`
## branch. What remains here is the extraction path only, which is a dependency
## of dino_analysis/ (visualize_dino_spatial_pca.py and, through it, phase1 and
## phase2). The module path and public names are deliberately UNCHANGED so those
## analysis scripts keep importing without edits.
##
## -------------------------------------------------------------------------
## SHAPE ASSUMPTIONS (every one is flagged; change here if any is wrong)
## -------------------------------------------------------------------------
## A2. DINOv2 ViT-B/14 has embed dim 768 and 12 transformer blocks.
## A3. "Layers {1,4,8,12}" are 1-indexed. DINOv2's get_intermediate_layers
##     takes 0-indexed block indices, so we use {0,3,7,11}. <-- CHECK THIS maps
##     to what your source paper means by "layer 1..12".
## A4. DINO input is grayscale [B,1,H,W] in [0,1]; we tile to 3 channels, resize
##     to 224x224 (bilinear), then ImageNet-normalise. 224/14 = 16 -> a 16x16
##     patch grid. Because we MEAN-POOL patch tokens to one vector per layer,
##     the exact grid size does not reach the caller, so variable input sizes
##     are fine. (dino_analysis/ bypasses the pooling and reads the patch grid
##     directly off self.dino; it only reuses dino_preprocess and the loader.)
## A5. Per selected layer we mean-pool the patch tokens (CLS dropped) -> [B,768];
##     concat the 4 layers -> [B,3072].
## A6. CENTERING. The pooled vector is ~95% a shared constant offset (measured:
##     ||mean||~106 vs ||residual||~22), and the measured object signal lives
##     ENTIRELY in the residual (raw features are object-blind, d~0.03 n.s.;
##     centered d=+0.25/+8.6sigma). A FIXED mean vector, if supplied, is
##     subtracted at the end of forward. Fixed mean, NOT BatchNorm. The mean is
##     a registered buffer. NOTE: the E1 script that produced these vectors
##     (compute_dino_feat_mean.py) was removed with the rest of E1; feat_mean is
##     now an optional argument and defaults to no centering. dino_analysis/
##     computes its own spatial per-layer means separately.
## -------------------------------------------------------------------------

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# ImageNet statistics DINOv2 was trained with (A4).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def dino_preprocess(img, img_size, mean, std):
    """Exact tensor transform applied before DINO. Shared by forward() and the
    debug script so what you inspect is what training uses.

    img: [B,1orC,H,W] in [0,1]. Returns [B,3,img_size,img_size], ImageNet-normed.
    """
    if img.shape[1] == 1:
        img = img.repeat(1, 3, 1, 1)
    img = F.interpolate(img, size=(img_size, img_size),
                        mode='bilinear', align_corners=False)
    return (img - mean) / std


def dino_denormalize(img, mean, std):
    """Invert ImageNet normalisation for display."""
    return img * std + mean


def load_dino_feat_mean(path, feat_dim):
    """Load a precomputed pooled-DINO mean vector -> [1, feat_dim] float32.

    Accepts either a bare tensor or a dict of the form
    {'mean': [D], 'meta': {...}}. Fails loudly on a missing file or a width
    mismatch -- a wrong-layer-set mean must never be silently broadcast.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f'dino_feat_mean must point at a precomputed .pt; got {path!r}')
    obj = torch.load(path, map_location='cpu')
    mu = obj['mean'] if isinstance(obj, dict) else obj
    mu = torch.as_tensor(mu, dtype=torch.float32).reshape(-1)
    if mu.numel() != feat_dim:
        raise ValueError(
            f'dino_feat_mean {path!r} has width {mu.numel()}, expected {feat_dim} '
            f'-- wrong layer set or wrong model?')
    return mu.view(1, feat_dim)


class DINOv2Extractor(nn.Module):
    """Frozen DINOv2 ViT-B/14 multi-layer pooled feature extractor (A2-A5)."""

    def __init__(self, layers=(0, 3, 7, 11), img_size=224,
                 github_repo='facebookresearch/dinov2', model_name='dinov2_vitb14',
                 hub_source='github', hub_dir=None, weights=None, feat_mean=None):
        super().__init__()
        self.layers = tuple(layers)
        self.img_size = img_size
        self.embed_dim = 768                       # A2
        self.feat_dim = self.embed_dim * len(self.layers)  # A5

        if hub_source == 'local':
            # OFFLINE: hub_dir must be the cached repo DIRECTORY (contains
            # hubconf.py), and weights the cached .pth. torch.hub.load treats the
            # first arg as a local path when source='local' -- passing a
            # github-style string here is the bug this replaces. Fail loudly.
            if hub_dir is None or not os.path.isdir(hub_dir):
                raise FileNotFoundError(
                    f'dino_hub_dir must be the cached DINOv2 repo directory '
                    f'(with hubconf.py); got {hub_dir!r}')
            if not os.path.isfile(os.path.join(hub_dir, 'hubconf.py')):
                raise FileNotFoundError(
                    f'{hub_dir!r} has no hubconf.py -- not a torch.hub repo dir')
            if weights is None or not os.path.isfile(weights):
                raise FileNotFoundError(
                    f'dino_weights must point at the cached .pth; got {weights!r}')
            self.dino = torch.hub.load(hub_dir, model_name, source='local',
                                       pretrained=False)
            sd = torch.load(weights, map_location='cpu')
            # strict=True: raises if any key mismatches -> a silently random ViT
            # can never slip through. Store the (empty-on-success) key lists.
            self.load_result = self.dino.load_state_dict(sd, strict=True)
        else:
            # ONLINE fallback (login node): download/cache from GitHub.
            if hub_dir is not None:
                torch.hub.set_dir(hub_dir)
            self.dino = torch.hub.load(github_repo, model_name, source='github',
                                       pretrained=(weights is None))
            if weights is not None:
                sd = torch.load(weights, map_location='cpu')
                self.load_result = self.dino.load_state_dict(sd, strict=True)
            else:
                self.load_result = None   # weights came from pretrained=True

        self.register_buffer('mean', torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

        # A6. Fixed per-arm centering vector. Zeros == no centering (the buffer
        # always exists so the state_dict keys are identical either way).
        self.centered = feat_mean is not None
        mu = (load_dino_feat_mean(feat_mean, self.feat_dim) if self.centered
              else torch.zeros(1, self.feat_dim))
        self.register_buffer('feat_mean', mu)

        for p in self.dino.parameters():
            p.requires_grad_(False)
        self.dino.eval()

    def train(self, mode=True):
        # Keep DINO frozen/eval even when the parent is put in train mode.
        super().train(mode)
        self.dino.eval()
        return self

    @torch.no_grad()
    def forward(self, img):
        # img: [B,1,H,W] or [B,3,H,W] in [0,1]  (A4)
        img = dino_preprocess(img, self.img_size, self.mean, self.std)
        feats = self.dino.get_intermediate_layers(
            img, n=self.layers, reshape=False, return_class_token=False, norm=True)
        # each feats[i]: [B, N_patches, 768] -> mean-pool -> [B,768]  (A5)
        pooled = torch.cat([f.mean(dim=1) for f in feats], dim=1)   # [B, 768*L]
        return pooled - self.feat_mean.to(pooled.dtype)             # A6 centering

