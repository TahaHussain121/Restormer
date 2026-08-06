## Restormer + frozen DINOv2 FiLM guidance.
##
## Subclass of the baseline Restormer (restormer_arch.py) that adds semantic
## guidance from a FROZEN DINOv2 ViT-B/14 as feature-wise linear modulation
## (FiLM) at the bottleneck and the three decoder stages. Nothing else changes:
## the backbone submodules are built by the parent __init__ in the identical
## order, so for a given seed the backbone weights are bit-identical to the
## baseline. At initialisation the FiLM projection is zero, so gamma=0, beta=0
## and (1+gamma)*F + beta == F -> the whole network is numerically identical to
## the baseline until FiLM learns non-zero modulation.
##
## -------------------------------------------------------------------------
## SHAPE ASSUMPTIONS (every one is flagged; change here if any is wrong)
## -------------------------------------------------------------------------
## A1. Backbone is Restormer with dim=48 => FiLM target channel counts are
##       bottleneck (latent)   : dim*8 = 384
##       decoder_level3        : dim*4 = 192
##       decoder_level2        : dim*2 = 96
##       decoder_level1        : dim*2 = 96      (parent uses dim*2 here, not dim)
##     Sum of gamma channels = 768; MLP emits 2*768 = 1536 (gamma||beta).
##     These are read from the actual submodule dims at build time (asserted),
##     so a different `dim` is handled automatically.
## A2. DINOv2 ViT-B/14 has embed dim 768 and 12 transformer blocks.
## A3. "Layers {1,4,8,12}" are 1-indexed. DINOv2's get_intermediate_layers
##     takes 0-indexed block indices, so we use {0,3,7,11}. <-- CHECK THIS maps
##     to what your source paper means by "layer 1..12".
## A4. DINO input: the model is fed the SAME tensor Restormer sees (the LQ crop
##     during training). It is grayscale [B,1,H,W] in [0,1]; we tile to 3
##     channels, resize to 224x224 (bilinear), then ImageNet-normalise.
##     224/14 = 16 -> a 16x16 patch grid. Because we MEAN-POOL patch tokens to
##     one vector per layer, the exact grid size does not reach the FiLM head,
##     so variable Restormer input sizes are fine.
## A5. Per selected layer we mean-pool the patch tokens (CLS dropped) -> [B,768];
##     concat the 4 layers -> [B,3072] as the FiLM head input.
## A6. CENTERING. The pooled vector is ~95% a shared constant offset (measured:
##     ||mean||~106 vs ||residual||~22), and the measured object signal lives
##     ENTIRELY in the residual (raw features are object-blind, d~0.03 n.s.;
##     centered d=+0.25/+8.6sigma). So a FIXED per-arm mean, precomputed over
##     training crops by Deraining_Holo/compute_dino_feat_mean.py, is subtracted
##     at the end of the extractor forward. Fixed mean, NOT BatchNorm: the
##     progressive schedule drops the batch to 2 at 256px and a two-sample mean
##     is noise, not a mean. The mean is a registered buffer, so it travels with
##     the checkpoint and a chained resume cannot silently use a different one.
##     Subtracting a constant cannot break the zero-init identity (gamma=beta=0
##     regardless of the feature), but the sanity check re-verifies it anyway.
## -------------------------------------------------------------------------

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.models.archs.restormer_arch import Restormer

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

    Accepts either a bare tensor or the dict written by
    Deraining_Holo/compute_dino_feat_mean.py ({'mean': [D], 'meta': {...}}).
    Fails loudly on a missing file or a width mismatch -- a wrong-arm or
    wrong-layer-set mean must never be silently broadcast into training.
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


class _StubExtractor(nn.Module):
    """Deterministic stand-in for DINO used ONLY by the wiring sanity check.

    Returns a fixed random feature of the correct width so the identity test
    can run offline. NEVER use for training -- carries no semantic information.
    Honours feat_mean so the sanity check exercises the same centering path.
    """

    def __init__(self, feat_dim, feat_mean=None):
        super().__init__()
        self.feat_dim = feat_dim
        mu = (load_dino_feat_mean(feat_mean, feat_dim) if feat_mean is not None
              else torch.zeros(1, feat_dim))
        self.register_buffer('feat_mean', mu)

    def forward(self, x):
        g = torch.Generator(device='cpu').manual_seed(0)
        f = torch.randn(x.shape[0], self.feat_dim, generator=g)
        return f.to(x.device, x.dtype) - self.feat_mean.to(x.device, x.dtype)


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


class FiLMHead(nn.Module):
    """MLP: pooled DINO vector -> per-channel (gamma, beta) for each stage.

    Final projection is zero-initialised, so gamma=beta=0 at init (identity).
    """

    def __init__(self, in_dim, channels, hidden=512):
        super().__init__()
        self.channels = list(channels)          # e.g. [384,192,96,96]
        self.total = sum(self.channels)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * self.total),  # gamma || beta
        )
        # zero-init the LAST linear -> identity at init (ControlNet-style).
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feat):
        out = self.mlp(feat)                    # [B, 2*total]
        gamma_all, beta_all = out[:, :self.total], out[:, self.total:]
        gammas = torch.split(gamma_all, self.channels, dim=1)
        betas = torch.split(beta_all, self.channels, dim=1)
        return gammas, betas


def _film(x, gamma, beta):
    # (1 + gamma) * x + beta ; gamma,beta: [B,C] broadcast over H,W.
    B, C = gamma.shape
    return x * (1 + gamma.view(B, C, 1, 1)) + beta.view(B, C, 1, 1)


class RestormerDINO(Restormer):
    """Restormer with frozen-DINOv2 FiLM at bottleneck + decoder stages."""

    def __init__(self,
                 # --- DINO / FiLM options (everything else forwarded to Restormer) ---
                 dino_layers=(0, 3, 7, 11),
                 dino_img_size=224,
                 dino_model_name='dinov2_vitb14',
                 dino_hub_source='github',
                 dino_hub_dir=None,
                 dino_weights=None,
                 dino_feat_mean=None,      # .pt with the per-arm pooled mean (A6)
                 dino_stub=False,          # True -> offline stub, wiring test only
                 film_hidden=512,
                 **restormer_kwargs):
        super().__init__(**restormer_kwargs)   # builds backbone first (identical init)

        # Read the real target channel counts from the built submodules (A1).
        ch_bottleneck = self.latent[0].attn.qkv.in_channels
        ch_dec3 = self.decoder_level3[0].attn.qkv.in_channels
        ch_dec2 = self.decoder_level2[0].attn.qkv.in_channels
        ch_dec1 = self.decoder_level1[0].attn.qkv.in_channels
        self.film_channels = [ch_bottleneck, ch_dec3, ch_dec2, ch_dec1]

        if dino_stub:
            self.dino = _StubExtractor(768 * len(dino_layers),
                                       feat_mean=dino_feat_mean)
        else:
            self.dino = DINOv2Extractor(
                layers=dino_layers, img_size=dino_img_size,
                model_name=dino_model_name, hub_source=dino_hub_source,
                hub_dir=dino_hub_dir, weights=dino_weights,
                feat_mean=dino_feat_mean)

        self.film = FiLMHead(self.dino.feat_dim, self.film_channels, hidden=film_hidden)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self.dino, 'train'):
            self.dino.train(False)   # stay frozen/eval
        return self

    def forward(self, inp_img, dino_img=None, apply_film=True):
        # --- semantic guidance ---
        # dino_img is the image DINO looks at. If None, DINO sees the same LQ
        # input as Restormer (Variant B). If provided (e.g. the black-bg render),
        # DINO sees that instead (Variant A). Restormer always processes inp_img.
        if apply_film:
            feat = self.dino(inp_img if dino_img is None else dino_img)
            gammas, betas = self.film(feat)

        # --- baseline Restormer forward, with FiLM at 4 points ---
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)
        if apply_film:
            latent = _film(latent, gammas[0], betas[0])            # bottleneck

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        if apply_film:
            out_dec_level3 = _film(out_dec_level3, gammas[1], betas[1])

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        if apply_film:
            out_dec_level2 = _film(out_dec_level2, gammas[2], betas[2])

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        if apply_film:
            out_dec_level1 = _film(out_dec_level1, gammas[3], betas[3])

        out_dec_level1 = self.refinement(out_dec_level1)

        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1
