"""dinolight-render: DINOLight's method adapted to radar, end to end.

This is NOT a one-factor ablation and must never be reported as one. It changes
TWO things at once versus addition-render — the LAYER COUNT (four depths instead
of one) and the FUSION OPERATOR (gated channel cross-attention instead of a
residual add) — and it is by a wide margin the largest arm in the ladder. It
answers "does the published method transfer to our data", not "which factor
matters". The single-factor arms are `affm-render` (layer count alone, addition
unchanged) and the planned `aca-L6` (operator alone, B6 unchanged).

    addition-render   D = centered B6                        guided = F + P(D)
    affm-render       D = sum_l W_l * centered D_l            guided = F + P(D)
    dinolight-render  D = sum_l W_l * centered D_l            guided = ACA(F, P(D))

THE THREE STAGES

  1. AFFM over {3,6,9,12}. `DinoAffm` is IMPORTED from the affm-render arch, not
     reimplemented, so stage 1 is byte-identical to the arm already training and
     the two remain comparable on the feature side. It also means this arm
     inherits that module's GELU. (The work order's pseudocode said SiLU; the
     instruction "identical to the arm already built" was taken as the stronger
     one, since comparability between the two arms is worth more than matching
     DINOLight's activation. Recorded in the devlog.)
  2. `P = Conv2d(768, 384, 1)` — the SAME projection addition-render uses, same
     shape, same bias, but here it feeds the attention instead of being added.
  3. `DinoAca` from `dino_aca.py` — see that module for why the attention is
     channel-transposed and not spatial. Short version: our failed spatial
     cross-attention arm scored +2.00 dB at crop128 and -7.56 dB at full256 from
     identical weights because a 256-token softmax does not survive becoming a
     1024-token softmax. A channel softmax is the same shape at both scales.

WHAT IS UNTOUCHED. Exactly one tensor changes: the one handed to the latent at
`inp_enc_level4`. Encoder, decoder, the eight latent transformer blocks,
refinement, skips, the global residual against the radar and the loss are
identical to E0 and to every other arm.

MONITORING, AND ONE HONEST WRINKLE

`latent_norm` and `injection_ratio` keep their tags and their gate. For
`projected_norm` this arm logs **||alpha * F_ca||**, the actual DINO
contribution, as the work order specifies -- it is the quantity that is
comparable *in spirit* to the other arms' ||P(D)||.

  It starts at EXACTLY ZERO, like every other arm's ||P(D)||, because `P` is
  also zero-initialised: the prior entering the attention is 0, so F_ca is 0 and
  alpha*F_ca is 0. `injection_ratio` therefore rises from 0 and the gate reads
  the same shape of quantity it reads everywhere else. The true injected delta
  into F is logged separately as `aca_injected_norm`; at init both are 0.

Additional observations every `dino_aca_stats_freq` forwards, published through
the model wrapper's existing generic hook and arriving with a `dino/` prefix:
the four AFFM weights and their sum, `aca_alpha`, `aca_ca_to_sa_ratio`,
`aca_injected_norm`, the per-head temperatures of both attentions, and the
per-head channel-attention entropy of both. None of them is a gate rule.

INITIALISATION
  * AFFM scoring convs zero -> the layer softmax starts exactly uniform at 0.25.
  * `project_out` zero -> `guided == F` -> step-0 output bit-identical to E0.
  * `alpha_logit = -2.0` -> alpha ~ 0.119. The gate opens quietly and, crucially,
    can close again.
  * temperatures 1.0 per head, Restormer's convention (multiplicative).
  * A THREE-STEP GRADIENT STAIRCASE, measured, not assumed. Two zero-inits sit
    in series -- `project_out` at the ACA output and `P` at the prior's entrance:
      step 1  only `project_out` has gradient (everything else reaches the loss
              only through it, and it is 0)
      step 2  `project_out` has moved, so the FEATURE path (Q/K/V from F) and
              `P` become live. The DINO CROSS path is still dead: `P` had zero
              gradient at step 1, so AdamW left it at exactly 0, so `d_proj` is
              still 0 and `to_k_cross`/`to_v_cross`/AFFM/alpha have dL/dW = 0.
      step 3  `P` has moved, `d_proj` is non-zero, and EVERYTHING is live.
    Measured max|dL/dW|: alpha 0 -> 0 -> 1.80e-05, AFFM 0 -> 0 -> 5.67e-06.
    This is arithmetic, not a defect, and 3 of 300,000 steps is not a cost worth
    redesigning for. If it ever mattered, dropping `P`'s zero-init would shorten
    it to two steps without affecting step-0 equality (which `project_out`
    alone already guarantees) -- deliberately NOT done here, because `P` is
    inherited from the parent under the config's `dino_init: zero` and keeping
    it identical to the other arms is worth more than one step.

RNG ORDER. The parent builds the whole Restormer trunk first; the ViT and `P`
are built inside the parent's own RNG save/restore fence; the AFFM and the ACA
are built inside a fence added here. Every trunk weight is byte-identical to
E0's for the same seed, and every later draw (data order, crop offsets,
augmentation flags) is unshifted.
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


class RestormerDinoLightRender(RestormerDinoSpatialRender):

    def __init__(self, *args, **kwargs):
        fusion = kwargs.pop('dino_fusion', 'aca')
        if fusion != 'aca':
            raise ValueError(
                f'RestormerDinoLightRender requires dino_fusion: aca, got '
                f'{fusion!r} (addition is RestormerDinoSpatialRender, '
                f'AFFM+addition is RestormerDinoAffmRender)')
        layers1 = list(kwargs.pop('dino_layers', (3, 6, 9, 12)))
        means_train = dict(kwargs.pop('dino_means_train128', {}) or {})
        means_eval = dict(kwargs.pop('dino_means_eval256', {}) or {})
        self.aca_heads = int(kwargs.pop('dino_aca_heads', 6))
        self.aca_alpha_init = float(kwargs.pop('dino_aca_alpha_init', -2.0))
        self.aca_stats_freq = int(kwargs.pop('dino_aca_stats_freq', 5000))
        # READ, never pop: the parent still needs them. Threading the trunk's
        # own `bias` and `LayerNorm_type` into the ACA is what keeps this module
        # from drifting from Restormer's convention -- the config sets them once.
        self.dino_trunk_bias = bool(kwargs.get('bias', False))
        self.dino_trunk_ln = str(kwargs.get('LayerNorm_type', 'WithBias'))

        if len(layers1) < 2 or sorted(layers1) != list(layers1) \
                or len(set(layers1)) != len(layers1):
            raise ValueError(f'dino_layers must be ascending and unique, got '
                             f'{layers1}')
        ref_block = 6
        if ref_block not in layers1:
            raise ValueError(f'dino_layers must contain B{ref_block}; got {layers1}')
        kwargs.setdefault('dino_block', ref_block)
        if int(kwargs['dino_block']) != ref_block:
            raise ValueError('dino_block is fixed at 6 for this arm; the layer '
                             'set is given by dino_layers')
        for key, src in (('dino_mean_train128', means_train),
                         ('dino_mean_eval256', means_eval)):
            k = _key(src, ref_block)
            if k is None:
                raise ValueError(f'{key.replace("dino_mean_", "dino_means_")} '
                                 f'must contain an entry for block {ref_block}')
            kwargs.setdefault(key, src[k])

        super().__init__(*args, **kwargs)
        self.dino_fusion = fusion
        self.dino_layers_1indexed = [int(b) for b in layers1]
        self.dino_layers_0indexed = [dino_shared.b1_to_b0(b) for b in layers1]

        # ---- RNG FENCE around AFFM + ACA construction ----------------------
        cpu_rng = torch.get_rng_state()
        cuda_rng = (torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None)
        try:
            self.affm = DinoAffm(len(self.dino_layers_1indexed),
                                 self.dino_embed_dim)
            self.aca = DinoAca(dim=self.latent_channels, heads=self.aca_heads,
                               bias=self.dino_trunk_bias,
                               alpha_init=self.aca_alpha_init,
                               LayerNorm_type=self.dino_trunk_ln)
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

        # ---- one centering mean per (layer, regime) ------------------------
        self.dino_mean_paths_per_layer = {'train128': {}, 'eval256': {}}
        for key, src in (('train128', means_train), ('eval256', means_eval)):
            for b1 in self.dino_layers_1indexed:
                k = _key(src, b1)
                if k is None:
                    raise ValueError(f'dino_means_{key} has no entry for B{b1}')
                self.dino_mean_paths_per_layer[key][b1] = src[k]
                self.register_buffer(f'mu_b{b1}_{key}',
                                     self._load_layer_mean(src[k], key, b1))
        for key in ('train128', 'eval256'):
            mine = getattr(self, f'mu_b{ref_block}_{key}')
            theirs = getattr(self, f'mu_{key}')
            if not torch.equal(mine, theirs):
                raise ValueError(
                    f'B{ref_block} {key} mean differs from the reference mean '
                    f'(max abs {float((mine - theirs).abs().max()):.3e})')

        self._aca_forward_count = 0
        self.last_attn_stats = {}
        self.last_affm_weights = None

    # ------------------------------------------------------------------
    # means: the parent's guard, parameterised by block (never weakened)
    # ------------------------------------------------------------------
    def _load_layer_mean(self, path, key, block_1indexed):
        if path is None or not os.path.isfile(path):
            raise FileNotFoundError(
                f'dino_means_{key}[{block_1indexed}] not found: {path}')
        obj = torch.load(path, map_location='cpu', weights_only=False)
        mu = obj['mean'] if isinstance(obj, dict) else obj
        mu = torch.as_tensor(mu, dtype=torch.float32).reshape(-1)
        if mu.numel() != self.dino_embed_dim:
            raise ValueError(f'{path}: width {mu.numel()} != {self.dino_embed_dim}')
        if not torch.isfinite(mu).all():
            raise ValueError(f'{path}: non-finite entries in the centering mean')
        meta = obj.get('meta', {}) if isinstance(obj, dict) else {}
        if meta:
            got = int(meta.get('block_1indexed', block_1indexed))
            if got != int(block_1indexed):
                raise ValueError(f'{path}: metadata says block {got}, configured '
                                 f'as the B{block_1indexed} mean')
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
                                 f'must be render for this arm')
        return mu

    # ------------------------------------------------------------------
    def dino_prior(self, inp_img, mode, latent_hw):
        """render -> AFFM-fused centered grid [B,768,g,g]. Same as affm-render."""
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
                f'{list(inp_img.shape)}. dinolight-render needs '
                f'Dataset_PairedImage_uint16_RenderStacked.')

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

        # ---- the ONLY difference: AFFM -> P -> ACA, in place of F + P(D) ----
        d_fused = self.dino_prior(render, mode, inp_enc_level4.shape[-2:])
        d_proj = self.P(d_fused.to(inp_enc_level4.dtype))
        self._aca_forward_count += 1
        want = (self._capture_dino_io
                or (self._aca_forward_count - 1) % self.aca_stats_freq == 0)
        guided, ca_term, aca_stats = self.aca(inp_enc_level4, d_proj,
                                              collect_stats=want)
        # projected_norm := ||alpha * F_ca||, the DINO contribution (see the
        # module docstring for why this starts NON-zero while the injected
        # delta is exactly zero).
        self._record_stats(inp_enc_level4, ca_term, mode)
        if want:
            with torch.no_grad():
                w = self._last_affm_w
                err = float((w.sum(dim=1) - 1.0).abs().max())
                if err > 1e-4:
                    raise RuntimeError(
                        f'AFFM weights do not sum to 1 (max deviation {err:.3e})')
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
        # ---------------------------------------------------------------------

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
