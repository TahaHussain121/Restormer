"""affm-render: the LAYER-COUNT ablation of addition-render.

ONE thing differs from `RestormerDinoSpatialRender`: how many DINO layers the
prior is built from.

    addition-render   D = centered B6 grid                       one layer
    affm-render       D = sum_l W_l * centered D_l,  l in {3,6,9,12}

The injection operator is UNCHANGED -- the same zero-initialized 1x1 projection
768 -> 384 and the same residual addition at `inp_enc_level4`. No attention, no
second injection point, no change to the gate. One factor at a time.

WHY {3, 6, 9, 12} AND NOT THE PAPERS' SETS
    DINOLight uses {1, 6, 12}; Perceive-IR and DSGIR use {1, 4, 8, 12}. We use
    the quarter depths {3, 6, 9, 12} for two reasons that are specific to this
    project and are not a claim about those papers:
      * the set CONTAINS B6, so this arm differs from addition-render by layer
        COUNT alone -- B6 is not swapped out, it is joined;
      * all four have a MEASURED fixed-128 correspondence on our radar data
        (WO1 Task 1.3, n=339: B3 0.6091, B6 0.6694, B9 0.5482, B12 0.2337).
        B1, B4 and B8 have never been measured here, so a set containing them
        would mix a layer-count change with an unmeasured-layer change.

THE FUSION -- ADAPTIVE FEATURE FUSION MODULE (AFFM)
    Adopted from DINOLight (arXiv 2603.12579) and CITED, not claimed. It is
    used because its output stays 768 channels:

        s_l   = Conv2d(768, 1, 1)( GELU(D_l) )          -> [B, 1, g, g]
        W     = softmax( [s_3, s_6, s_9, s_12], dim=1 ) -> [B, 4, g, g]
        D_fus = sum_l W[:, l:l+1] * D_l                 -> [B, 768, g, g]

    The softmax runs ACROSS LAYERS at each spatial position, so the four
    weights sum to 1 at every one of the 256 positions. Because the result is a
    weighted SUM and not a concatenation, `P` stays Conv2d(768, 384, 1) and the
    arm lands within ~1% of addition-render's parameter count. A naive
    four-layer concatenation would have needed Conv2d(3072, 384, 1) =
    1,180,032 parameters and would have reintroduced exactly the capacity
    confound this design exists to avoid.

    ACTIVATION. DINOLight's module is described with SiLU; this repo
    standardises on GELU (`restormer_arch.py:91`, `F.gelu`), so GELU is used
    here. The activation is not load-bearing -- it only shapes the scoring
    conv's input -- and the deviation is recorded in the devlog.

CENTERING -- FOUR MEANS, NOT ONE
    Every layer is centered with ITS OWN position-independent [768] train-only
    mean, BEFORE the scoring conv sees it, because the four blocks sit in
    genuinely different places in feature space (norms 59.2 / 56.9 / 55.4 /
    75.7 at train128). Centering all four against B6's mean would bias three of
    them by a constant and would make the AFFM scores read that bias rather
    than the content. The B6 mean is byte-identical to the file addition-render
    already trains with -- asserted in `__init__`, not assumed.

PARAMETERS (+298,372 over E0)
    P              768*384 + 384 = 295,296     (unchanged from addition-render)
    AFFM score x4  4 * (768 + 1) =   3,076
    Against addition-render's 295,296 that is +3,076, about 1%. This arm is
    very nearly parameter-matched to its own reference, which is the point: a
    decline across the layer ladder cannot be explained away by capacity.

INITIALIZATION
    The scoring convs are ZERO-initialized (weight and bias), so every s_l is 0
    and the softmax starts EXACTLY uniform at 0.25 per layer -- deterministic
    and seed-independent, so the arm's starting prior is the unweighted mean of
    the four centered grids rather than an arbitrary draw. `P` stays
    zero-initialized, so P(D_fused) == 0 and the step-0 output is E0's exactly.

    THERE IS A ONE-STEP GRADIENT STAIRCASE, and it is unavoidable given the
    zero-init of `P`. D_fused is non-zero from step 0 (a zero score still gives
    a weight of 0.25), so `P` takes gradient on the first backward. But
    D_fused reaches the loss ONLY through `P`, and d(P(D))/dD == P.weight == 0,
    so the AFFM scoring convs receive EXACTLY zero gradient on that first
    backward. From step 2 `P` is non-zero and they learn normally. Measured:
    step 1  |dL/dP| 1.6e-02, |dL/dAFFM| 0 for all four layers
    step 2  |dL/dP| 2.1e-02, |dL/dAFFM| 6.8e-05 .. 1.8e-03
    This is the same structure the attention arms have, one level shallower,
    and it is asserted in the smoke test rather than assumed away.

RNG ORDER
    The parent builds the whole Restormer trunk first, then fences the ViT and
    `P`. This class adds its own fence around the AFFM construction, so every
    trunk weight and every later draw (data order, crop offsets, augmentation
    flags) stays byte-identical to E0's for the same seed.

MONITORING
    `latent_norm`, `projected_norm` and `injection_ratio` are inherited and go
    to the same tags, the same dino_stability.csv and the same gate, unchanged.
    Additionally, every `dino_affm_stats_freq` forwards, the position-averaged
    softmax weights are published through the model wrapper's existing generic
    observation hook (`last_attn_stats`), arriving as the TensorBoard tags
    `dino/affm_w_b3`, `dino/affm_w_b6`, `dino/affm_w_b9`, `dino/affm_w_b12`.

    THE TAG PREFIX IS `dino/`, NOT `affm/`, ON PURPOSE: the model wrapper
    prefixes every observation with `dino/`, and giving this arm an `affm/`
    prefix would require editing a shared file that five finished arms also
    run. The names are unambiguous as they stand. Like the attention arms'
    statistics, the value is a step function -- it is MEASURED every
    `dino_affm_stats_freq` forwards and REPRINTED at every `print_freq`, so a
    log reader must de-duplicate before plotting.

    These are OBSERVATIONS. No gate rule is defined on them, deliberately.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F_nn

from basicsr.models.archs.restormer_dino_render_arch import (
    RestormerDinoSpatialRender)

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
_P3SCRIPTS = os.path.join(_REPO, 'dino_analysis_phases', 'phase3_restoration',
                          'scripts')
if _P3SCRIPTS not in sys.path:
    sys.path.insert(0, _P3SCRIPTS)

import dino_shared                                            # noqa: E402


class DinoAffm(nn.Module):
    """DINOLight's Adaptive Feature Fusion Module (arXiv 2603.12579).

    A per-position softmax over LAYERS. Input: a list of L centered grids, each
    [B, C, g, g]. Output: one [B, C, g, g] weighted sum plus the [B, L, g, g]
    weight map that produced it.
    """

    def __init__(self, n_layers, embed_dim=768):
        super().__init__()
        self.n_layers = int(n_layers)
        self.embed_dim = int(embed_dim)
        self.score = nn.ModuleList([
            nn.Conv2d(self.embed_dim, 1, kernel_size=1, bias=True)
            for _ in range(self.n_layers)])
        for conv in self.score:
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)      # -> every score 0 -> softmax uniform

    def forward(self, grids):
        if len(grids) != self.n_layers:
            raise ValueError(f'expected {self.n_layers} grids, got {len(grids)}')
        shapes = {tuple(g.shape) for g in grids}
        if len(shapes) != 1:
            raise ValueError(f'AFFM needs identical grid shapes, got {shapes}')
        scores = torch.cat([conv(F_nn.gelu(g))
                            for conv, g in zip(self.score, grids)], dim=1)
        w = scores.softmax(dim=1)                         # across LAYERS
        fused = sum(w[:, i:i + 1] * grids[i] for i in range(self.n_layers))
        return fused, w


class RestormerDinoAffmRender(RestormerDinoSpatialRender):

    def __init__(self, *args, **kwargs):
        # surfaced in the YAML so the fusion is visible in the config, not only
        # in the class name.
        fusion = kwargs.pop('dino_fusion', 'affm')
        if fusion != 'affm':
            raise ValueError(
                f'RestormerDinoAffmRender requires dino_fusion: affm, got '
                f'{fusion!r} (single-layer addition is '
                f'RestormerDinoSpatialRender)')
        layers1 = list(kwargs.pop('dino_layers', (3, 6, 9, 12)))
        means_train = dict(kwargs.pop('dino_means_train128', {}) or {})
        means_eval = dict(kwargs.pop('dino_means_eval256', {}) or {})
        self.affm_stats_freq = int(kwargs.pop('dino_affm_stats_freq', 5000))

        if len(layers1) < 2:
            raise ValueError(f'dino_layers needs >= 2 layers, got {layers1}')
        if len(set(layers1)) != len(layers1):
            raise ValueError(f'dino_layers has duplicates: {layers1}')
        if sorted(layers1) != list(layers1):
            raise ValueError(f'dino_layers must be ascending, got {layers1}')

        # the parent still owns ONE block index and ONE mean pair. Both are set
        # to B6 so that (a) its validation runs on a layer that really is in
        # our set and (b) `mu_train128` / `mu_eval256` stay exactly the buffers
        # addition-render carries -- which is what the equality assert below
        # compares this arm's own B6 mean against.
        ref_block = 6
        if ref_block not in layers1:
            raise ValueError(f'dino_layers must contain the reference block '
                             f'B{ref_block}; got {layers1}')
        kwargs.setdefault('dino_block', ref_block)
        if int(kwargs['dino_block']) != ref_block:
            raise ValueError('dino_block is fixed at 6 for this arm; the layer '
                             'set is given by dino_layers')
        for key, src in (('dino_mean_train128', means_train),
                         ('dino_mean_eval256', means_eval)):
            if str(ref_block) not in {str(k) for k in src}:
                raise ValueError(f'{key.replace("dino_mean_", "dino_means_")} '
                                 f'must contain an entry for block {ref_block}')
            kwargs.setdefault(key, src[_key(src, ref_block)])

        super().__init__(*args, **kwargs)
        self.dino_fusion = fusion
        self.dino_layers_1indexed = [int(b) for b in layers1]
        self.dino_layers_0indexed = [dino_shared.b1_to_b0(b) for b in layers1]

        # ---- RNG FENCE around the AFFM construction ------------------------
        # nn.Conv2d draws from the generator even when the draw is overwritten
        # by a zero-init, so without this every later draw would shift away
        # from E0's for the same seed.
        cpu_rng = torch.get_rng_state()
        cuda_rng = (torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None)
        try:
            self.affm = DinoAffm(len(self.dino_layers_1indexed),
                                 self.dino_embed_dim)
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
                path = src[k]
                self.dino_mean_paths_per_layer[key][b1] = path
                self.register_buffer(f'mu_b{b1}_{key}',
                                     self._load_layer_mean(path, key, b1))

        # ---- the correctness check the work order asks for, in code --------
        # this arm's B6 mean must BE addition-render's B6 mean, in both
        # regimes. A silently different mean would change every token by a
        # constant and make the layer comparison meaningless.
        for key in ('train128', 'eval256'):
            mine = getattr(self, f'mu_b{ref_block}_{key}')
            theirs = getattr(self, f'mu_{key}')
            if not torch.equal(mine, theirs):
                raise ValueError(
                    f'B{ref_block} {key} mean differs from the reference mean '
                    f'(max abs {float((mine - theirs).abs().max()):.3e}); '
                    f'{self.dino_mean_paths_per_layer[key][ref_block]} vs '
                    f'{self.dino_mean_paths[key]}')

        self._affm_forward_count = 0
        self.last_attn_stats = {}       # the wrapper's generic observation hook
        self.last_affm_weights = None   # [B, L, g, g], for the offline dump

    # ------------------------------------------------------------------
    # means
    # ------------------------------------------------------------------
    def _load_layer_mean(self, path, key, block_1indexed):
        """The parent's `_load_mean`, with the block it validates against made
        explicit instead of fixed at `self.dino_block_1indexed`.

        The parent guard is CORRECT and is deliberately left alone: for a
        single-layer arm a mean whose metadata names a different block is
        always an error. This arm legitimately loads four different blocks, so
        it needs the same checks parameterised by block rather than a weaker
        version of them. Every other check -- width, finiteness, DINO input
        size for the regime, train-only provenance -- is identical, and the B6
        entry additionally has to equal the parent's own buffer (asserted in
        `__init__`), which ties this loader back to the parent's.
        """
        if path is None:
            raise ValueError(f'dino_means_{key}[{block_1indexed}] is required')
        if not os.path.isfile(path):
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
                raise ValueError(f'{path}: metadata says block {got}, but it is '
                                 f'configured as the B{block_1indexed} mean')
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
    def _means_for(self, mode):
        return {b1: getattr(self, f'mu_b{b1}_{mode}')
                for b1 in self.dino_layers_1indexed}

    # ------------------------------------------------------------------
    # the prior -- the ONLY method that differs from addition-render
    # ------------------------------------------------------------------
    def dino_prior(self, inp_img, mode, latent_hw):
        """render [B,1,H,W] in [0,1] -> AFFM-fused centered grid [B,768,g,g]."""
        _, dino_size = self._mean_and_size(mode)
        mus = self._means_for(mode)
        if self._capture_dino_io:
            self._dino_capture['source'] = inp_img

        with torch.no_grad():
            tok = dino_shared.extract_blocks(
                self.dino_ext, inp_img, self.dino_layers_0indexed, dino_size)
            grids = []
            for b1, b0 in zip(self.dino_layers_1indexed,
                              self.dino_layers_0indexed):
                centered = dino_shared.center_tokens(tok[b0], mus[b1])
                grids.append(dino_shared.tokens_to_grid(centered))

        # AFFM is TRAINABLE and must stay outside no_grad; the frozen ViT above
        # is inside it.
        grids = [g.to(self.affm.score[0].weight.dtype) for g in grids]
        fused, w = self.affm(grids)

        self._affm_forward_count += 1
        # (count - 1) % freq == 0, NOT count % freq == 1: the two agree for
        # every freq > 1, but the second form NEVER fires at freq == 1, which
        # silently produced a run with no AFFM observations at all. Found by
        # the CPU integration smoke run, which sets freq 1 deliberately.
        if self._capture_dino_io or \
                (self._affm_forward_count - 1) % self.affm_stats_freq == 0:
            with torch.no_grad():
                sums = w.sum(dim=1)
                err = float((sums - 1.0).abs().max())
                if err > 1e-4:
                    raise RuntimeError(
                        f'AFFM weights do not sum to 1 (max deviation {err:.3e})')
                means = w.mean(dim=(0, 2, 3))
                self.last_attn_stats = {
                    f'affm_w_b{b1}': float(means[i])
                    for i, b1 in enumerate(self.dino_layers_1indexed)}
                self.last_attn_stats['affm_w_sum'] = float(means.sum())
                self.last_affm_weights = w.detach()

        if self._capture_dino_io:
            self._dino_capture['preprocessed'] = dino_shared.preprocess(
                inp_img, dino_size)
            self._dino_capture['tokens'] = tok
            self._dino_capture['layer_grids'] = grids
            self._dino_capture['affm_weights'] = w
            self._dino_capture['grid'] = fused

        # HARD GATE: unchanged. The fused grid must already equal the latent
        # grid; Phase 3 never interpolates the feature grid.
        dino_shared.assert_no_interpolation_needed(fused.shape[-2:], latent_hw)
        return fused


def _key(d, b1):
    """Accept both integer and string keys from YAML ({3: path} or {'3': path})."""
    for k in d:
        if str(k) == str(b1):
            return k
    return None
