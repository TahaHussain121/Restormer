"""Restormer + frozen spatial DINOv2 B6 guidance at the latent (Phase-3, E1-N-Fixed).

WHAT THIS IS
    The stock `Restormer` with ONE addition: patch tokens from a frozen DINOv2
    ViT-B/14 block 6, extracted from the SAME noisy 1e5 tensor the network is
    restoring, centered by a fixed training-set mean, and added into
    `inp_enc_level4` through a ZERO-INITIALIZED 1x1 convolution, immediately
    before the 8 latent Transformer blocks:

        inp_enc_level4 = self.down3_4(out_enc_level3)
        guided         = inp_enc_level4 + self.P(D_centered)
        latent         = self.latent(guided)

    Call it a ZERO-INITIALIZED RESIDUAL PROJECTION. It is not FiLM, not adaLN.
    There is no learnable alpha, no gate, no multiplicative path, no
    cross-attention, no concatenation, no decoder injection. E1 failed three
    times on FiLM runaway (CONTEXT.md); none of that returns here.

WHY IT STARTS AS AN EXACT NO-OP
    P.weight and P.bias are zero, so P(D) == 0 at initialization and E1's step-0
    output is identical to E0's. Because dL/dW = dL/d(guided) (x) D, P still gets
    a non-zero gradient on the very first backward, so the branch can grow if it
    is useful and stays at zero if it is not. Both facts are asserted by the
    smoke test.

THE SINGLE-STREAM GUARANTEE
    `ImageCleanModel.optimize_parameters` calls `net_g(self.lq)`. This module
    derives the DINO input from that same `inp_img` tensor object inside
    `forward`. There is no second dataset stream, so the failure mode named in
    REPO_INVESTIGATION_REPORT section N.1 -- `train.py` progressively sub-cropping
    only `lq`/`gt` and leaving a third tensor misaligned -- cannot occur by
    construction. `_dino_capture` exists so the smoke test can prove it with
    `torch.equal` rather than trusting this paragraph.

SCALE CONSISTENCY (never interpolate the feature grid)
    mode 'train128':  radar 128 -> DINO 224 -> 256 tokens -> 16x16 == latent 16x16
    mode 'eval256' :  radar 256 -> DINO 448 -> 1024 tokens -> 32x32 == latent 32x32
    224/14 = 16 and 448/14 = 32, so radar-px-per-token stays ~8 in both regimes.
    The mode is an EXPLICIT flag (`set_dino_mode`), never inferred from tensor
    size; `dino_shared.assert_no_interpolation_needed` is the hard gate that
    fires if the mode and the input size ever disagree.

CHECKPOINTS
    The frozen DINOv2 weights are NOT written into the checkpoint (86M extra
    params x 150 checkpoints would be ~60 GB of duplicated frozen weights). They
    are loaded from the offline cache at construction. `state_dict` strips them
    and `load_state_dict` re-injects the live ones before delegating, so
    `strict=True` still catches any real key mismatch in the trained part. The
    two centering means ARE kept in the checkpoint -- they are part of the
    experiment definition.
"""

import os
import sys
from collections import OrderedDict

import torch
import torch.nn as nn

from basicsr.models.archs.restormer_arch import Restormer

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
_P3_SCRIPTS = os.path.join(_REPO, 'dino_analysis_phases', 'phase3_restoration',
                           'scripts')
if _P3_SCRIPTS not in sys.path:
    sys.path.insert(0, _P3_SCRIPTS)

import dino_shared                                   # noqa: E402

DINO_PREFIX = 'dino_ext.'


class RestormerDinoSpatial(Restormer):
    """Restormer with a zero-initialized residual DINOv2-B6 latent projection.

    Every Restormer argument keeps its stock meaning and default. The dino_*
    arguments are the only additions and are all surfaced in the YAML.
    """

    MODES = ('train128', 'eval256')

    def __init__(self,
                 inp_channels=3, out_channels=3, dim=48,
                 num_blocks=(4, 6, 6, 8), num_refinement_blocks=4,
                 heads=(1, 2, 4, 8), ffn_expansion_factor=2.66, bias=False,
                 LayerNorm_type='WithBias', dual_pixel_task=False,
                 # ---- DINO guidance -------------------------------------
                 dino_enabled=True,
                 dino_model='dinov2_vitb14',
                 dino_block=6,                    # 1-indexed; B6 -> index 5
                 dino_frozen=True,
                 dino_source='same_lq',
                 dino_injection='latent',
                 dino_projection='conv1x1',
                 dino_init='zero',
                 dino_embed_dim=768,
                 dino_mean_train128=None,
                 dino_mean_eval256=None,
                 dino_default_mode='train128'):

        # ---------------------------------------------------------------
        # 1. the ENTIRE Restormer trunk is constructed first, by the stock
        #    __init__, so its RNG consumption is byte-for-byte what E0 does.
        # ---------------------------------------------------------------
        super().__init__(
            inp_channels=inp_channels, out_channels=out_channels, dim=dim,
            num_blocks=list(num_blocks),
            num_refinement_blocks=num_refinement_blocks, heads=list(heads),
            ffn_expansion_factor=ffn_expansion_factor, bias=bias,
            LayerNorm_type=LayerNorm_type, dual_pixel_task=dual_pixel_task)

        # ---- contract checks: refuse anything this experiment forbids ----
        if not dino_enabled:
            raise ValueError('RestormerDinoSpatial requires dino_enabled: true; '
                             'the no-DINO control is E0 with type: Restormer')
        # 'same_lq'  -- E1-addition: DINO sees the same 1e5 tensor Restormer gets
        # 'render'   -- E1-render:   DINO sees the aligned render (subclass
        #               RestormerDinoSpatialRender; this class never selects it)
        if dino_source not in ('same_lq', 'render'):
            raise ValueError(f'dino_source must be same_lq or render, '
                             f'got {dino_source!r}')
        if dino_injection != 'latent':
            raise ValueError(f'dino_injection must be latent, got {dino_injection!r}')
        if dino_projection != 'conv1x1':
            raise ValueError(f'dino_projection must be conv1x1, got {dino_projection!r}')
        if dino_init != 'zero':
            raise ValueError(f'dino_init must be zero, got {dino_init!r}')
        if not dino_frozen:
            raise ValueError('dino_frozen must be true')
        if dino_default_mode not in self.MODES:
            raise ValueError(f'dino_default_mode must be one of {self.MODES}')

        self.dino_enabled = True
        self.dino_model_name = dino_model
        self.dino_block_1indexed = int(dino_block)
        self.dino_block_0indexed = dino_shared.b1_to_b0(dino_block)
        self.dino_embed_dim = int(dino_embed_dim)
        self.dino_source = dino_source
        self.dino_injection = dino_injection
        self.dino_mode = dino_default_mode
        self.latent_channels = int(dim * 2 ** 3)          # 384

        # ---------------------------------------------------------------
        # 2. RNG FENCE. Building the ViT and the projection both consume RNG.
        #    Snapshot the generator state, build them, restore it -- so every
        #    RNG draw AFTER model construction (data order, crop positions,
        #    augmentation flags) is identical to E0's for the same seed. The
        #    trunk above is already unaffected because it was built first.
        # ---------------------------------------------------------------
        cpu_rng = torch.get_rng_state()
        cuda_rng = (torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None)
        try:
            self.dino_ext = dino_shared.build_dino(
                device='cpu', model_name=dino_model, verbose=False)
            for p in self.dino_ext.parameters():
                p.requires_grad_(False)
            self.dino_ext.eval()

            # the zero-initialized residual projection, built LAST
            self.P = nn.Conv2d(self.dino_embed_dim, self.latent_channels,
                               kernel_size=1)
            nn.init.zeros_(self.P.weight)
            nn.init.zeros_(self.P.bias)
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

        # ---------------------------------------------------------------
        # 3. centering means: one [768] vector per evaluation regime.
        #    Both are loaded eagerly and both live in the checkpoint.
        # ---------------------------------------------------------------
        self.dino_mean_paths = {'train128': dino_mean_train128,
                                'eval256': dino_mean_eval256}
        self.register_buffer('mu_train128',
                             self._load_mean(dino_mean_train128, 'train128'))
        self.register_buffer('mu_eval256',
                             self._load_mean(dino_mean_eval256, 'eval256'))

        # diagnostics filled in by forward(); read by the model wrapper's
        # stability logging and by the smoke tests. Never used by the math.
        self.last_dino_stats = {}
        self.last_dino_mode = None
        self.last_mean_key = None
        self._capture_dino_io = False
        self._dino_capture = {}

    # ------------------------------------------------------------------
    # means
    # ------------------------------------------------------------------
    def _load_mean(self, path, key):
        if path is None:
            raise ValueError(f'dino_mean_{key} is required')
        if not os.path.isfile(path):
            raise FileNotFoundError(f'dino_mean_{key} not found: {path}')
        obj = torch.load(path, map_location='cpu', weights_only=False)
        mu = obj['mean'] if isinstance(obj, dict) else obj
        mu = torch.as_tensor(mu, dtype=torch.float32).reshape(-1)
        if mu.numel() != self.dino_embed_dim:
            raise ValueError(f'{path}: width {mu.numel()} != {self.dino_embed_dim}')
        if not torch.isfinite(mu).all():
            raise ValueError(f'{path}: non-finite entries in the centering mean')
        meta = obj.get('meta', {}) if isinstance(obj, dict) else {}
        # a mean built for the wrong block or the wrong scale must never be
        # accepted silently -- it would bias every token by a constant.
        if meta:
            if int(meta.get('block_1indexed', self.dino_block_1indexed)) != \
                    self.dino_block_1indexed:
                raise ValueError(f'{path}: block {meta.get("block_1indexed")} != '
                                 f'B{self.dino_block_1indexed}')
            want = (dino_shared.DINO_SIZE_TRAIN128 if key == 'train128'
                    else dino_shared.DINO_SIZE_EVAL256)
            if int(meta.get('dino_input_size', want)) != want:
                raise ValueError(f'{path}: dino_input_size '
                                 f'{meta.get("dino_input_size")} != {want}')
            if str(meta.get('source_split', 'train')) != 'train':
                raise ValueError(f'{path}: source_split is '
                                 f'{meta.get("source_split")!r}, must be train')
        return mu

    def set_dino_mode(self, mode):
        """EXPLICIT regime switch. Never inferred from tensor shape."""
        if mode not in self.MODES:
            raise ValueError(f'dino mode must be one of {self.MODES}, got {mode!r}')
        self.dino_mode = mode
        return self

    def _mean_and_size(self, mode):
        if mode == 'train128':
            return self.mu_train128, dino_shared.DINO_SIZE_TRAIN128
        return self.mu_eval256, dino_shared.DINO_SIZE_EVAL256

    # ------------------------------------------------------------------
    # the prior
    # ------------------------------------------------------------------
    def dino_prior(self, inp_img, mode, latent_hw):
        """inp_img [B,1,H,W] in [0,1] -> centered B6 grid [B,768,H/8,W/8]."""
        mu, dino_size = self._mean_and_size(mode)
        if self._capture_dino_io:
            # the EXACT tensor object handed to DINO extraction, for the
            # crop-stream identity proof in the smoke test
            self._dino_capture['source'] = inp_img
        with torch.no_grad():
            tokens = dino_shared.extract_block(
                self.dino_ext, inp_img, self.dino_block_0indexed, dino_size)
            tokens = dino_shared.center_tokens(tokens, mu)
            grid = dino_shared.tokens_to_grid(tokens)
        if self._capture_dino_io:
            self._dino_capture['preprocessed'] = dino_shared.preprocess(
                inp_img, dino_size)
            self._dino_capture['tokens'] = tokens
            self._dino_capture['grid'] = grid
        # HARD GATE: the DINO grid must already equal the latent grid.
        dino_shared.assert_no_interpolation_needed(grid.shape[-2:], latent_hw)
        return grid

    # ------------------------------------------------------------------
    # forward -- stock Restormer.forward with exactly one added line
    # ------------------------------------------------------------------
    def forward(self, inp_img, dino_mode=None):
        mode = dino_mode or self.dino_mode
        if mode not in self.MODES:
            raise ValueError(f'unknown dino mode {mode!r}')

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)

        # ---------------- DINO guidance, and nothing else ----------------
        d_centered = self.dino_prior(inp_img, mode, inp_enc_level4.shape[-2:])
        projected = self.P(d_centered.to(inp_enc_level4.dtype))
        guided = inp_enc_level4 + projected
        self._record_stats(inp_enc_level4, projected, mode)
        # -----------------------------------------------------------------

        latent = self.latent(guided)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1

    @torch.no_grad()
    def _record_stats(self, latent_in, projected, mode):
        eps = 1e-8
        ln = float(latent_in.detach().norm())
        pn = float(projected.detach().norm())
        self.last_dino_stats = {
            'latent_norm': ln,
            'projected_norm': pn,
            'injection_ratio': pn / (ln + eps),
        }
        self.last_dino_mode = mode
        self.last_mean_key = ('mu_train128' if mode == 'train128'
                              else 'mu_eval256')

    # ------------------------------------------------------------------
    # keep the frozen ViT out of every checkpoint
    # ------------------------------------------------------------------
    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        return OrderedDict((k, v) for k, v in sd.items()
                           if not k.startswith(DINO_PREFIX))

    def load_state_dict(self, state_dict, strict=True):
        """DINO weights come from the offline cache, never from a checkpoint.

        They are re-injected here so `strict=True` remains meaningful for
        everything that IS trained -- a genuine key mismatch still raises.
        """
        merged = OrderedDict(state_dict)
        for k, v in super().state_dict().items():
            if k.startswith(DINO_PREFIX):
                merged.setdefault(k, v)
        return super().load_state_dict(merged, strict=strict)

    def train(self, mode=True):
        """DINO stays in eval() no matter what the parent does."""
        super().train(mode)
        self.dino_ext.eval()
        self.dino_ext.dino.eval()
        return self
