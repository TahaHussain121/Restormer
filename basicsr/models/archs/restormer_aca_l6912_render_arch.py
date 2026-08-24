"""aca-L6912: ACA fusion over the DINO layer set {6,9,12}.

Part of the ACA layer ladder. Fusion is IDENTICAL across the whole ladder --
gated channel cross-attention, the `DinoAca` block imported from `dino_aca.py`,
the same object dinolight-render runs. **Only the layer set changes:**

    aca-L6            {6}          no AFFM (a 1-layer softmax is a no-op)
    aca-L36           {3,6}        AFFM then ACA
    aca-L6912         {6,9,12}     AFFM then ACA
    dinolight-render  {3,6,9,12}   AFFM then ACA

Across that ladder the ONLY parameter difference is the AFFM scoring convs,
**769 per layer**, i.e. well under 1% of the arm. **A difference across this
ladder is therefore NOT attributable to capacity.** (Against *addition-render*
the story is different and must not be conflated: that comparison is ~4.6x the
parameters and IS capacity confounded.)

THE PIPELINE, replacing exactly one tensor at `inp_enc_level4`:

    for l in {6,9,12}:  D_l = centered DINOv2 patch grid  [B,768,g,g]
    s_l   = Conv2d(768,1,1)(GELU(D_l))
    W     = softmax([s_6, s_9, s_12], dim=layers)   -> [B,3,g,g], sums to 1 per position
    D_fus = sum_l W_l * D_l                  -> still 768 channels
    D_proj = P(D_fus)                        -> [B,384,g,g]
    guided = project_out(F_sa + alpha*F_ca) + F

`DinoAffm` is IMPORTED from the affm-render arch and `DinoAca` from
`dino_aca.py`, so the two blocks are the same code objects every other arm in
the family runs. **The layer set is hard-coded in this file, not a config
flag** -- each arm is its own module so a change to one can never silently
alter another.

MEANS ARE SHARED, NOT RECOMPUTED. This arm loads the same per-layer mean files
dinolight-render and affm-render load. Identical centering is what keeps the
ladder comparable on the feature side; the B6 entry is asserted byte-identical
to the mean addition-render trains with.

MONITORING follows dinolight-render exactly. `projected_norm` logs
`||alpha * F_ca||` measured BEFORE the zero-init output conv, which makes this
arm's `injection_ratio` roughly an ORDER OF MAGNITUDE smaller than the addition
arms' ~1.03 and **NOT directly comparable to them**. The true injected delta is
logged separately as `aca_injected_norm`.

INITIALISATION: AFFM scoring convs zero -> the layer softmax starts exactly
uniform at 1/3 = 0.3333. `project_out` zero -> step-0 output
bit-identical to E0. `alpha_logit = -2.0` -> alpha ~ 0.119. Temperatures 1.0
per head (Restormer's multiplicative MDTA convention). Two zero-inits sit in
series (`project_out`, then `P`), so there is a THREE-step gradient staircase:
step 1 only `project_out`; step 2 the feature path and `P`; step 3 everything
including AFFM and alpha.

RNG ORDER: trunk built first by the parent, ViT and `P` inside the parent's
fence, AFFM and ACA inside a fence added here -- so every trunk weight stays
byte-identical to E0's for the same seed.
"""

import os
import sys

import torch

from basicsr.models.archs.restormer_dino_render_arch import (
    RestormerDinoSpatialRender)
from basicsr.models.archs.restormer_dino_affm_render_arch import DinoAffm
from basicsr.models.archs.dino_aca import DinoAca

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
_P3SCRIPTS = os.path.join(_REPO, 'dino_analysis_phases', 'phase3_restoration',
                          'scripts')
if _P3SCRIPTS not in sys.path:
    sys.path.insert(0, _P3SCRIPTS)

import dino_shared                                            # noqa: E402

LAYERS_1INDEXED = [6, 9, 12]          # HARD-CODED. Not a config flag.
REFERENCE_BLOCK = 6                 # must be in the set; ties the arm to B6


class RestormerAcaL6912Render(RestormerDinoSpatialRender):

    def __init__(self, *args, **kwargs):
        fusion = kwargs.pop('dino_fusion', 'aca')
        if fusion != 'aca':
            raise ValueError(f'RestormerAcaL6912Render requires dino_fusion: aca, got {fusion!r}')
        if 'dino_layers' in kwargs:
            raise ValueError(
                'aca-L6912 hard-codes its layer set [6, 9, 12]; dino_layers is not a '
                'config key for this arm. Use the arm whose name matches the '
                'set you want.')
        means_train = dict(kwargs.pop('dino_means_train128', {}) or {})
        means_eval = dict(kwargs.pop('dino_means_eval256', {}) or {})
        self.aca_heads = int(kwargs.pop('dino_aca_heads', 6))
        self.aca_alpha_init = float(kwargs.pop('dino_aca_alpha_init', -2.0))
        self.aca_stats_freq = int(kwargs.pop('dino_aca_stats_freq', 5000))
        self.aca_trunk_bias = bool(kwargs.get('bias', False))
        self.aca_trunk_ln = str(kwargs.get('LayerNorm_type', 'WithBias'))

        kwargs.setdefault('dino_block', REFERENCE_BLOCK)
        if int(kwargs['dino_block']) != REFERENCE_BLOCK:
            raise ValueError(f'dino_block is fixed at {REFERENCE_BLOCK} for '
                             f'this arm; the set is {LAYERS_1INDEXED}')
        for key, src in (('dino_mean_train128', means_train),
                         ('dino_mean_eval256', means_eval)):
            k = _key(src, REFERENCE_BLOCK)
            if k is None:
                raise ValueError(f'{key.replace("dino_mean_", "dino_means_")} '
                                 f'must contain block {REFERENCE_BLOCK}')
            kwargs.setdefault(key, src[k])

        super().__init__(*args, **kwargs)
        self.dino_fusion = fusion
        self.dino_layers_1indexed = list(LAYERS_1INDEXED)
        self.dino_layers_0indexed = [dino_shared.b1_to_b0(b)
                                     for b in LAYERS_1INDEXED]

        # ---- RNG FENCE around AFFM + ACA construction ----------------------
        cpu_rng = torch.get_rng_state()
        cuda_rng = (torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None)
        try:
            self.affm = DinoAffm(len(LAYERS_1INDEXED), self.dino_embed_dim)
            self.aca = DinoAca(dim=self.latent_channels, heads=self.aca_heads,
                               bias=self.aca_trunk_bias,
                               alpha_init=self.aca_alpha_init,
                               LayerNorm_type=self.aca_trunk_ln)
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

        # ---- one SHARED mean per (layer, regime) ---------------------------
        self.dino_mean_paths_per_layer = {'train128': {}, 'eval256': {}}
        for key, src in (('train128', means_train), ('eval256', means_eval)):
            for b1 in LAYERS_1INDEXED:
                k = _key(src, b1)
                if k is None:
                    raise ValueError(f'dino_means_{key} has no entry for B{b1}')
                self.dino_mean_paths_per_layer[key][b1] = src[k]
                self.register_buffer(f'mu_b{b1}_{key}',
                                     self._load_layer_mean(src[k], key, b1))
        for key in ('train128', 'eval256'):
            mine = getattr(self, f'mu_b{REFERENCE_BLOCK}_{key}')
            theirs = getattr(self, f'mu_{key}')
            if not torch.equal(mine, theirs):
                raise ValueError(
                    f'B{REFERENCE_BLOCK} {key} mean differs from the reference '
                    f'mean (max abs {float((mine - theirs).abs().max()):.3e})')

        self._aca_forward_count = 0
        self.last_attn_stats = {}
        self.last_affm_weights = None

    # ------------------------------------------------------------------
    def _load_layer_mean(self, path, key, block_1indexed):
        """The parent's guard, parameterised by block. Never weakened: every
        check the parent makes is made here, plus a domain check."""
        if path is None or not os.path.isfile(path):
            raise FileNotFoundError(
                f'dino_means_{key}[{block_1indexed}] not found: {path}')
        obj = torch.load(path, map_location='cpu', weights_only=False)
        mu = obj['mean'] if isinstance(obj, dict) else obj
        mu = torch.as_tensor(mu, dtype=torch.float32).reshape(-1)
        if mu.numel() != self.dino_embed_dim:
            raise ValueError(f'{path}: width {mu.numel()} != {self.dino_embed_dim}')
        if not torch.isfinite(mu).all():
            raise ValueError(f'{path}: non-finite entries')
        meta = obj.get('meta', {}) if isinstance(obj, dict) else {}
        if meta:
            got = int(meta.get('block_1indexed', block_1indexed))
            if got != int(block_1indexed):
                raise ValueError(f'{path}: metadata says block {got}, '
                                 f'configured as B{block_1indexed}')
            want = (dino_shared.DINO_SIZE_TRAIN128 if key == 'train128'
                    else dino_shared.DINO_SIZE_EVAL256)
            if int(meta.get('dino_input_size', want)) != want:
                raise ValueError(f'{path}: dino_input_size '
                                 f'{meta.get("dino_input_size")} != {want}')
            if str(meta.get('source_split', 'train')) != 'train':
                raise ValueError(f'{path}: source_split is '
                                 f'{meta.get("source_split")!r}, must be train')
            if str(meta.get('domain', 'render')) != 'render':
                raise ValueError(f'{path}: domain is {meta.get("domain")!r}, '
                                 f'must be render')
        return mu

    # ------------------------------------------------------------------
    def dino_prior(self, inp_img, mode, latent_hw):
        """render -> AFFM-fused centered grid [B,768,g,g] over {6,9,12}."""
        _, dino_size = self._mean_and_size(mode)
        mus = {b1: getattr(self, f'mu_b{b1}_{mode}')
               for b1 in self.dino_layers_1indexed}
        if self._capture_dino_io:
            self._dino_capture['source'] = inp_img
        with torch.no_grad():
            tok = dino_shared.extract_blocks(
                self.dino_ext, inp_img, self.dino_layers_0indexed, dino_size)
            grids = [dino_shared.tokens_to_grid(
                dino_shared.center_tokens(tok[b0], mus[b1]))
                for b1, b0 in zip(self.dino_layers_1indexed,
                                  self.dino_layers_0indexed)]
        grids = [g.to(self.affm.score[0].weight.dtype) for g in grids]
        fused, w = self.affm(grids)
        self._last_affm_w = w
        if self._capture_dino_io:
            self._dino_capture['layer_grids'] = grids
            self._dino_capture['affm_weights'] = w
            self._dino_capture['grid'] = fused
            self._dino_capture['preprocessed'] = dino_shared.preprocess(
                inp_img, dino_size)
        dino_shared.assert_no_interpolation_needed(fused.shape[-2:], latent_hw)
        return fused

    # ------------------------------------------------------------------
    def forward(self, inp_img, dino_mode=None):
        mode = dino_mode or self.dino_mode
        if mode not in self.MODES:
            raise ValueError(f'unknown dino mode {mode!r}')
        if inp_img.shape[1] != self.inp_stack_channels:
            raise ValueError(
                f'expected a stacked [B,2,H,W] input (radar, render), got '
                f'{list(inp_img.shape)}.')

        radar = inp_img[:, 0:1]
        render = inp_img[:, 1:2]
        if self._capture_dino_io:
            self._dino_capture['radar'] = radar
            self._dino_capture['render'] = render
            self._dino_capture['stacked'] = inp_img

        inp_enc_level1 = self.patch_embed(radar)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        inp_enc_level4 = self.down3_4(out_enc_level3)

        d_fused = self.dino_prior(render, mode, inp_enc_level4.shape[-2:])
        d_proj = self.P(d_fused.to(inp_enc_level4.dtype))
        self._aca_forward_count += 1
        want = (self._capture_dino_io
                or (self._aca_forward_count - 1) % self.aca_stats_freq == 0)
        guided, ca_term, aca_stats = self.aca(inp_enc_level4, d_proj,
                                              collect_stats=want)
        self._record_stats(inp_enc_level4, ca_term, mode)
        if want:
            with torch.no_grad():
                w = self._last_affm_w
                err = float((w.sum(dim=1) - 1.0).abs().max())
                if err > 1e-4:
                    raise RuntimeError(
                        f'AFFM weights do not sum to 1 (max dev {err:.3e})')
                means = w.mean(dim=(0, 2, 3))
                stats = {f'affm_w_b{b1}': float(means[i])
                         for i, b1 in enumerate(self.dino_layers_1indexed)}
                stats['affm_w_sum'] = float(means.sum())
                stats.update(aca_stats)
                self.last_attn_stats = stats
                self.last_affm_weights = w.detach()
        if self._capture_dino_io:
            self._dino_capture['d_proj'] = d_proj
            self._dino_capture['ca_term'] = ca_term
            self._dino_capture['guided'] = guided

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
            out_dec_level1 = self.output(out_dec_level1) + radar
        return out_dec_level1


def _key(d, b1):
    for k in d:
        if str(k) == str(b1):
            return k
    return None
