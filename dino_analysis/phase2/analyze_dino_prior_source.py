"""PHASE 2 — DINO prior source: Very Noisy Radar (1e5) vs Render.

QUESTION
    At the DINO depth Phase 1 selected, which available source gives a spatial
    representation better aligned with the CLEAN 1e7 target:
        (1) DINO extracted from the Very Noisy Radar (1e5), or
        (2) DINO extracted from the corresponding Render?
    DINO(1e7) is the reference in both cases.

WHY ABSOLUTE SIMILARITY IS NOT ENOUGH
    Phase 1 measured substantial corresponding-patch cosine between DIFFERENT
    scenes (e.g. centered B6 1e5<->1e7 same-scene +0.686 vs wrong-object +0.623).
    A source can therefore look "closer to clean" purely by sharing a generic
    domain offset. Phase 2 reports two quantities side by side:
        absolute      : same-scene cosine to DINO(1e7)
        scene advantage: same-scene cosine - different-scene cosine
    and the paired deltas between the two sources for each. A conclusion is only
    drawn where both agree, or the disagreement is stated explicitly.

WHAT IS REUSED (nothing re-derived)
    Phase 1 module (dino_analysis/phase1/analyze_dino_spatial_consistency.py):
        verify_triplet          -> the exact triplet check Phase 1 used
        sample_features/center  -> extraction + training-mean centering
        summarize helpers/style -> aggregation conventions
        pca_figure_for_sample   -> representative-sample visualisation
    Phase 0 module (dino_analysis/visualize_dino_spatial_pca.py), via Phase 1:
        build_dataset, build_extractor, spatial_tokens, batch_to_dino_inputs,
        patchwise_cosine, joint_pca / shared_range / make_figure
    Phase 1 OUTPUTS:
        the selected block (read from the summary CSV, never hardcoded)
        the validated sample-ID list (read from the per-sample CSV)

SCOPE
    Phase 2 only. Restormer is untouched, nothing is trained, and no claim is
    made about PSNR/SSIM -- that is Phase 3.
"""

import argparse
import csv
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

_PHASE2 = os.path.dirname(os.path.abspath(__file__))
_DINO_ANALYSIS = os.path.dirname(_PHASE2)
_REPO = os.path.dirname(_DINO_ANALYSIS)
_PHASE1 = os.path.join(_DINO_ANALYSIS, 'phase1')
for _p in (_PHASE1, _DINO_ANALYSIS, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import analyze_dino_spatial_consistency as p1   # noqa: E402
import visualize_dino_spatial_pca as base       # noqa: E402

try:
    from scipy.stats import wilcoxon
    _HAVE_SCIPY = True
except ImportError:                              # pragma: no cover
    _HAVE_SCIPY = False

P1_OUT = os.path.join(_PHASE1, 'outputs')
P1_SUMMARY = os.path.join(P1_OUT, 'dino_spatial_similarity_val_summary.csv')
P1_PER_SAMPLE = os.path.join(P1_OUT, 'dino_spatial_similarity_val_per_sample.csv')
P1_META = os.path.join(P1_OUT, 'phase1_metadata.json')
OUT_DEFAULT = os.path.join(_PHASE2, 'outputs')
DEVLOG = os.path.join(_PHASE2, 'PHASE2_DEVLOG.md')

FEATURE_TYPES = ['raw', 'centered']
# distribution metrics summarised per (feature_type, block)
METRICS = [
    'similarity_1e5_to_clean', 'similarity_render_to_clean',
    'delta_render_over_1e5',
    'different_scene_1e5_to_clean', 'different_scene_render_to_clean',
    'scene_advantage_1e5', 'scene_advantage_render', 'delta_scene_advantage',
    'similarity_1e5_to_render',
]
# metrics that are paired differences -> win rates + Wilcoxon are meaningful
PAIRED = {
    'delta_render_over_1e5': ('similarity_render_to_clean',
                              'similarity_1e5_to_clean'),
    'delta_scene_advantage': ('scene_advantage_render', 'scene_advantage_1e5'),
}


# ---------------------------------------------------------------------------
# Phase 1 discovery
# ---------------------------------------------------------------------------
def load_phase1():
    """Read Phase 1 outputs. Stop loudly rather than guess anything."""
    for p in (P1_SUMMARY, P1_PER_SAMPLE, P1_META):
        if not os.path.isfile(p):
            raise SystemExit(
                f'Phase 1 output missing: {p}\n'
                f'Run Phase 1 first:\n'
                f'  python dino_analysis/phase1/analyze_dino_spatial_consistency.py '
                f'--device cuda')
    summary = list(csv.DictReader(open(P1_SUMMARY)))
    per_sample = list(csv.DictReader(open(P1_PER_SAMPLE)))
    meta = json.load(open(P1_META))
    if meta.get('smoke_test'):
        raise SystemExit('Phase 1 metadata says smoke_test=true — Phase 2 needs '
                         'a full Phase 1 run, not a smoke run')
    return summary, per_sample, meta


def select_block(summary):
    """Primary block = highest mean CENTERED same-scene 1e5<->1e7 in Phase 1."""
    cand = [(float(r['mean']), int(r['block_index']), r['layer_fraction'])
            for r in summary
            if r['feature_type'] == 'centered' and r['pair'] == '1e5_vs_1e7'
            and r['scene_match'] == 'same']
    if not cand:
        raise SystemExit('Phase 1 summary has no centered/1e5_vs_1e7/same rows — '
                         'cannot select a block; re-run Phase 1')
    cand.sort(reverse=True)
    if len(cand) > 1 and abs(cand[0][0] - cand[1][0]) < 1e-9:
        raise SystemExit(f'Phase 1 block selection is ambiguous: blocks '
                         f'{cand[0][1]} and {cand[1][1]} tie at {cand[0][0]:.6f}')
    primary = {'block': cand[0][1], 'fraction': cand[0][2], 'mean': cand[0][0]}
    secondary = ({'block': cand[1][1], 'fraction': cand[1][2], 'mean': cand[1][0]}
                 if len(cand) > 1 else None)
    return primary, secondary, cand


def phase1_sample_ids(per_sample):
    """The exact validated sample set Phase 1 analysed, in Phase 1 order."""
    seen, ids = set(), []
    for r in per_sample:
        if r['sample_id'] not in seen:
            seen.add(r['sample_id'])
            ids.append(r['sample_id'])
    return ids


def phase1_control(summary, block, ft, pair):
    """Phase 1's different-scene control mean, reused for the complementarity note."""
    for r in summary:
        if (r['feature_type'] == ft and int(r['block_index']) == block
                and r['pair'] == pair and r['scene_match'] == 'different'):
            return float(r['mean'])
    return float('nan')


# ---------------------------------------------------------------------------
# deterministic different-scene control
# ---------------------------------------------------------------------------
def derangement(n, seed):
    """Seeded permutation with NO fixed point: sample i is never its own control.

    Deterministic given (n, seed), so the pairing is reproducible, and the SAME
    mapping is used for both candidate sources so the comparison is fair.
    """
    if n < 2:
        raise SystemExit('need at least 2 samples to build a different-scene control')
    rng = random.Random(seed)
    idx = list(range(n))
    for _ in range(1000):
        rng.shuffle(idx)
        fixed = [i for i in range(n) if idx[i] == i]
        if not fixed:
            break
        for i in fixed:                       # repair by swapping with a neighbour
            j = (i + 1) % n
            idx[i], idx[j] = idx[j], idx[i]
        if all(idx[i] != i for i in range(n)):
            break
    if any(idx[i] == i for i in range(n)):
        raise SystemExit('failed to build a derangement')
    return idx


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract(ext, batch, domains, blocks1, img_size, device):
    """{domain: {block_1indexed: [N,C]}} — spatial patch tokens, no pooling.

    KEYED BY BLOCK, deliberately. DINOv2's `get_intermediate_layers` walks the
    blocks in order and appends whenever `i in blocks_to_take`, so it returns
    outputs in ASCENDING block order no matter what order `n` is given in.
    Indexing the returned list positionally against a caller-ordered block list
    silently mismatches features and blocks (and, once centering is applied,
    subtracts the wrong block's mean). Returning a dict removes the trap.
    """
    order = sorted(set(int(b) for b in blocks1))       # the order DINOv2 will use
    blocks0 = [b - 1 for b in order]
    inputs = base.batch_to_dino_inputs(batch, img_size, ext.mean.cpu(), ext.std.cpu())
    stack = torch.cat([inputs[d] for d in domains], dim=0).to(device)
    per_block = base.spatial_tokens(ext, stack, blocks0)
    return inputs, {d: {order[k]: per_block[k][di].numpy()
                        for k in range(len(order))}
                    for di, d in enumerate(domains)}


def centered(feat, means, domain, layer_idx, mode):
    mu = (means['per_domain'][domain][layer_idx] if mode == 'per-domain'
          else means['global'][layer_idx])
    return feat - mu.numpy()


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def describe(a):
    a = np.asarray(a, dtype=np.float64)
    return {'n': len(a), 'mean': float(a.mean()),
            'std': float(a.std(ddof=1)) if len(a) > 1 else float('nan'),
            'sem': float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else float('nan'),
            'median': float(np.median(a)),
            'q1': float(np.percentile(a, 25)), 'q3': float(np.percentile(a, 75)),
            'min': float(a.min()), 'max': float(a.max())}


def paired_stats(diff, tol):
    """Win rates, Wilcoxon and Cohen's dz for a paired difference vector."""
    d = np.asarray(diff, dtype=np.float64)
    tied = np.abs(d) <= tol
    out = {
        'pct_positive': 100.0 * float(np.mean((d > 0) & ~tied)),
        'pct_negative': 100.0 * float(np.mean((d < 0) & ~tied)),
        'pct_tied': 100.0 * float(np.mean(tied)),
        'cohens_dz': (float(d.mean() / d.std(ddof=1))
                      if len(d) > 1 and d.std(ddof=1) > 0 else float('nan')),
        'wilcoxon_stat': float('nan'), 'wilcoxon_p': float('nan'),
    }
    if _HAVE_SCIPY and len(d) > 1 and np.any(d != 0):
        try:
            st, p = wilcoxon(d, zero_method='wilcox', alternative='two-sided')
            out['wilcoxon_stat'], out['wilcoxon_p'] = float(st), float(p)
        except ValueError:
            pass
    return out


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
C_1E5, C_REN = '#0072B2', '#D55E00'


def _paired_panel(ax, a, b, labels, ylabel, title):
    """Slopegraph: one thin line per sample + box/mean overlay."""
    x = [0, 1]
    for va, vb in zip(a, b):
        ax.plot(x, [va, vb], color='0.55', lw=0.4, alpha=0.10, zorder=1,
                solid_capstyle='butt')
    bp = ax.boxplot([a, b], positions=x, widths=0.22, showfliers=False,
                    patch_artist=True, zorder=3,
                    medianprops=dict(color='black', lw=1.6))
    for patch, col in zip(bp['boxes'], (C_1E5, C_REN)):
        patch.set_facecolor(col); patch.set_alpha(0.55); patch.set_edgecolor(col)
    for xi, v, col in zip(x, (a, b), (C_1E5, C_REN)):
        ax.plot([xi], [np.mean(v)], marker='D', ms=6, color=col,
                markeredgecolor='black', markeredgewidth=0.6, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(axis='y', alpha=0.25, lw=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def _scatter_panel(ax, a, b, lab_a, lab_b, title):
    ax.scatter(a, b, s=9, alpha=0.35, color='#444444', edgecolors='none', zorder=2)
    lo = min(min(a), min(b)); hi = max(max(a), max(b))
    pad = 0.03 * (hi - lo if hi > lo else 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls='--', lw=1.0,
            color='0.3', zorder=3, label='y = x (equal)')
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(lab_a); ax.set_ylabel(lab_b)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def plot_paired(path, a, b, block, ft, n, ylabel, head, labels, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8))
    _paired_panel(axes[0], a, b, labels, ylabel, 'Paired per-sample values')
    _scatter_panel(axes[1], a, b, labels[0], labels[1],
                   'Above the line ⇒ Render higher')
    win = 100.0 * float(np.mean(np.asarray(b) > np.asarray(a)))
    fig.suptitle(f'{head}\nBlock {block}, {ft} features, n = {n} validation '
                 f'triplets   |   Render higher on {win:.1f}% of samples',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f'  wrote {os.path.relpath(path, _REPO)}')


def plot_delta_hist(path, d, block, ft, n, dpi):
    d = np.asarray(d)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.hist(d, bins=40, color='#7F9FC0', edgecolor='white', lw=0.5)
    ax.axvline(0, color='black', lw=1.4, ls='-', label='0 = sources tied')
    ax.axvline(d.mean(), color=C_REN, lw=1.4, ls='--',
               label=f'mean = {d.mean():+.4f}')
    ax.axvline(np.median(d), color='#009E73', lw=1.4, ls=':',
               label=f'median = {np.median(d):+.4f}')
    ax.set_xlabel('delta_render_over_1e5  =  cos(Render, 1e7) − cos(1e5, 1e7)\n'
                  '← 1e5 closer to clean        Render closer to clean →')
    ax.set_ylabel('validation samples')
    ax.set_title(f'Which source is closer to the clean DINO representation?\n'
                 f'Block {block}, {ft} features, n = {n}   |   '
                 f'{100.0 * float(np.mean(d > 0)):.1f}% of samples > 0',
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis='y', alpha=0.25, lw=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f'  wrote {os.path.relpath(path, _REPO)}')


# ---------------------------------------------------------------------------
# devlog
# ---------------------------------------------------------------------------
def append_devlog(stamp, rows, stats, ctx):
    def g(ft, metric, field='mean'):
        return stats[(ft, ctx['block'], metric)][field]

    def pstat(ft, metric, field):
        return stats[(ft, ctx['block'], metric)].get(field, float('nan'))

    d_abs = g('centered', 'delta_render_over_1e5')
    d_adv = g('centered', 'delta_scene_advantage')
    abs_winner = 'Render' if d_abs > 0 else '1e5'
    adv_winner = 'Render' if d_adv > 0 else '1e5'
    agree = abs_winner == adv_winner
    dz_adv = pstat('centered', 'delta_scene_advantage', 'cohens_dz')
    # An absolute gap far larger than the scene-advantage gap means the lead is
    # mostly generic domain similarity -- it shows up against wrong-scene targets
    # too -- rather than scene-specific content. Say so explicitly.
    ratio = abs(d_abs) / abs(d_adv) if d_adv != 0 else float('inf')
    mostly_generic = ratio >= 5.0
    magnitude_note = (
        f'- **The absolute gap is {ratio:.1f}x the scene-advantage gap.** Almost all of '
        f'{abs_winner}\'s absolute lead also shows up against WRONG-scene targets, so it is '
        f'generic domain similarity, not scene-specific content. The scene-advantage '
        f'column is the one that reflects usable structure (delta {d_adv:+.4f}, dz {dz_adv:+.3f}).'
        if mostly_generic else
        f'- The absolute gap ({d_abs:+.4f}) and the scene-advantage gap ({d_adv:+.4f}) are of '
        f'comparable order (ratio {ratio:.1f}x), so the absolute lead is not purely a '
        f'generic domain offset.')

    entry = f"""
## {stamp} — DINO Prior Source: 1e5 Radar vs Render

### Why
Phase 1 established that Block {ctx['block']} ({ctx['fraction']} depth) gives the strongest
cross-noise spatial consistency between the 1e5 and 1e7 radar representations
(centered mean {ctx['p1_mean']:+.4f}). The open question is which source we could actually
feed a restoration model: the noisy 1e5 radar itself, or the corresponding
render. Phase 2 asks which of the two produces a spatial DINO representation
better aligned with the clean 1e7 target, judged both in absolute terms and
against a different-scene control.

### What we did
- read the primary block from the Phase 1 summary CSV (not hardcoded)
- reused the exact Phase 1 validation triplet set ({ctx['n']} samples)
- compared DINO(1e5) with DINO(1e7), and DINO(Render) with DINO(1e7)
- evaluated raw and centered features
- built a deterministic seeded derangement as the different-scene control, using
  the same mapping for both candidate sources
- measured scene-specific advantage (same-scene − different-scene)
- ran paired Wilcoxon signed-rank tests on both quantities
- visualised representative best/worst/tied samples

### How
- selected block: **B{ctx['block']}** ({ctx['fraction']} depth), chosen as the highest mean
  centered same-scene 1e5↔1e7 in `dino_analysis/phase1/outputs/dino_spatial_similarity_val_summary.csv`
- secondary block (context only): {ctx['secondary_str']}
- model `{ctx['model']}`, checkpoint `{ctx['checkpoint']}`
- {ctx['n']} valid triplets (Phase 1 had {ctx['p1_n']}); {ctx['missing_str']}
- patch grid {ctx['gh']}×{ctx['gw']} = {ctx['ntok']} tokens, {ctx['embed_dim']}-d, patch {ctx['patch']}, input {ctx['img_size']}²
- centering: {ctx['centering_mode']} spatial means over {ctx['n_mean_images']} TRAIN images (unchanged from Phase 1)
- metric: mean corresponding-patch cosine on full {ctx['embed_dim']}-d features (never PCA)
- different-scene control: seeded derangement, seed {ctx['seed']}, no fixed points
- test: paired Wilcoxon signed-rank; magnitude reported as mean/median paired difference and Cohen's dz

### Results

**CENTERED features, Block {ctx['block']} (primary):**

| Quantity | 1e5 → clean | Render → clean |
|---|---|---|
| same-scene mean | {g('centered', 'similarity_1e5_to_clean'):+.4f} | {g('centered', 'similarity_render_to_clean'):+.4f} |
| same-scene median | {g('centered', 'similarity_1e5_to_clean', 'median'):+.4f} | {g('centered', 'similarity_render_to_clean', 'median'):+.4f} |
| different-scene mean | {g('centered', 'different_scene_1e5_to_clean'):+.4f} | {g('centered', 'different_scene_render_to_clean'):+.4f} |
| **scene advantage (mean)** | **{g('centered', 'scene_advantage_1e5'):+.4f}** | **{g('centered', 'scene_advantage_render'):+.4f}** |
| scene advantage (median) | {g('centered', 'scene_advantage_1e5', 'median'):+.4f} | {g('centered', 'scene_advantage_render', 'median'):+.4f} |

- direct delta (Render − 1e5), absolute: mean {g('centered', 'delta_render_over_1e5'):+.4f}, median {g('centered', 'delta_render_over_1e5', 'median'):+.4f}
  → Render wins on {pstat('centered', 'delta_render_over_1e5', 'pct_positive'):.1f}%, 1e5 wins on {pstat('centered', 'delta_render_over_1e5', 'pct_negative'):.1f}%, tied {pstat('centered', 'delta_render_over_1e5', 'pct_tied'):.1f}%
  → Wilcoxon p = {pstat('centered', 'delta_render_over_1e5', 'wilcoxon_p'):.3e}, Cohen's dz = {pstat('centered', 'delta_render_over_1e5', 'cohens_dz'):+.3f}
- delta scene advantage (Render − 1e5): mean {g('centered', 'delta_scene_advantage'):+.4f}, median {g('centered', 'delta_scene_advantage', 'median'):+.4f}
  → Render larger on {pstat('centered', 'delta_scene_advantage', 'pct_positive'):.1f}%, 1e5 larger on {pstat('centered', 'delta_scene_advantage', 'pct_negative'):.1f}%
  → Wilcoxon p = {pstat('centered', 'delta_scene_advantage', 'wilcoxon_p'):.3e}, Cohen's dz = {pstat('centered', 'delta_scene_advantage', 'cohens_dz'):+.3f}

**RAW features, Block {ctx['block']}:**

| Quantity | 1e5 → clean | Render → clean |
|---|---|---|
| same-scene mean | {g('raw', 'similarity_1e5_to_clean'):+.4f} | {g('raw', 'similarity_render_to_clean'):+.4f} |
| different-scene mean | {g('raw', 'different_scene_1e5_to_clean'):+.4f} | {g('raw', 'different_scene_render_to_clean'):+.4f} |
| **scene advantage (mean)** | **{g('raw', 'scene_advantage_1e5'):+.4f}** | **{g('raw', 'scene_advantage_render'):+.4f}** |

- direct delta mean {g('raw', 'delta_render_over_1e5'):+.4f} (Render wins {pstat('raw', 'delta_render_over_1e5', 'pct_positive'):.1f}%)
- delta scene advantage mean {g('raw', 'delta_scene_advantage'):+.4f} (Render larger {pstat('raw', 'delta_scene_advantage', 'pct_positive'):.1f}%)

**Complementarity (same scene, centered, B{ctx['block']}):** mean 1e5↔Render {g('centered', 'similarity_1e5_to_render'):+.4f},
median {g('centered', 'similarity_1e5_to_render', 'median'):+.4f}; Phase 1 different-scene control for the same pair/block
was {ctx['p1_ctrl_1e5_render']:+.4f}.

**Representative samples:** {ctx['rep_str']}

### Interpretation
- On ABSOLUTE alignment to the clean DINO representation, **{abs_winner}** is ahead
  (centered mean delta {d_abs:+.4f}).
- On SCENE-SPECIFIC advantage over the different-scene control, **{adv_winner}** is ahead
  (centered mean delta {d_adv:+.4f}, dz {dz_adv:+.3f}).
- The two criteria {'point the same way' if agree else 'DISAGREE'}{', but that is not the whole story' if agree and mostly_generic else ''}.{'' if agree else ' Absolute similarity is therefore partly generic domain similarity rather than scene content, exactly the failure mode Phase 1 warned about — the scene-advantage column is the one to trust.'}
{magnitude_note}
- 1e5 and Render are not interchangeable representations (same-scene 1e5↔Render
  mean {g('centered', 'similarity_1e5_to_render'):+.4f}), so complementarity is not ruled out; testing fusion is
  out of scope here.
- No claim is made about restoration PSNR/SSIM. This measures alignment to the
  clean DINO spatial representation only; whether that translates into
  restoration quality is Phase 3.

### Outputs
- `dino_analysis/phase2/outputs/dino_prior_source_per_sample.csv`
- `dino_analysis/phase2/outputs/dino_prior_source_summary.csv`
- `dino_analysis/phase2/outputs/dino_prior_source_paired_comparison.png`
- `dino_analysis/phase2/outputs/dino_prior_source_scene_advantage.png`
- `dino_analysis/phase2/outputs/dino_prior_source_delta_histogram.png`
- `dino_analysis/phase2/outputs/representative_samples/`
- `dino_analysis/phase2/outputs/phase2_metadata.json`

### Next Step
Phase 3 — controlled restoration experiments informed by Phase 1 and Phase 2.
Not implemented.

---
"""
    with open(DEVLOG, 'a') as f:
        f.write(entry)
    print(f'  appended entry to {DEVLOG}')


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description='Phase 2: is DINO(1e5) or DINO(Render) better aligned with '
                    'the clean DINO(1e7) target, at the Phase-1-selected block?',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-samples', type=int, default=None,
                    help='smoke test: only the first N Phase-1 samples. Output '
                         'goes to outputs/smoke/ so a full run is never '
                         'overwritten, and the devlog is not touched.')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--opt', default=base.DEFAULT_OPT)
    ap.add_argument('--means-cache', default=base.DEFAULT_MEANS)
    ap.add_argument('--centering-mode', default=None,
                    choices=['per-domain', 'global'],
                    help='default: whatever Phase 1 recorded in its metadata')
    ap.add_argument('--block', type=int, default=None,
                    help='override the Phase-1-selected primary block')
    ap.add_argument('--no-secondary', action='store_true',
                    help='skip the optional second-best block')
    ap.add_argument('--tie-tol', type=float, default=1e-3,
                    help='|delta| <= tol counts as a tie in the win rates')
    ap.add_argument('--percentile', type=float, nargs=2, default=(1.0, 99.0),
                    metavar=('LO', 'HI'))
    ap.add_argument('--display-size', type=int, default=224)
    ap.add_argument('--cmap', default='viridis')
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--skip-consistency-check', action='store_true',
                    help='skip verifying similarity_1e5_to_clean against Phase 1')
    ap.add_argument('--no-devlog', action='store_true')
    args = ap.parse_args()

    smoke = args.max_samples is not None
    out_dir = args.out_dir or (os.path.join(OUT_DEFAULT, 'smoke') if smoke
                               else OUT_DEFAULT)
    rep_dir = os.path.join(out_dir, 'representative_samples')
    for d in (out_dir, rep_dir):
        os.makedirs(d, exist_ok=True)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    started = datetime.datetime.now().astimezone()

    print('=' * 78)
    print('PHASE 2 — DINO prior source: Very Noisy Radar (1e5) vs Render')
    print('=' * 78)
    print(f'  started    : {started.isoformat(timespec="seconds")}')
    print(f'  output dir : {os.path.relpath(out_dir, _REPO)}'
          + ('   (SMOKE TEST — full-run outputs untouched)' if smoke else ''))

    # ---------------- Phase 1 discovery ----------------
    p1_summary, p1_per_sample, p1_meta = load_phase1()
    primary, secondary, ranked = select_block(p1_summary)
    if args.block is not None:
        match = [c for c in ranked if c[1] == args.block]
        if not match:
            raise SystemExit(f'--block {args.block} is not in the Phase 1 results')
        primary = {'block': match[0][1], 'fraction': match[0][2], 'mean': match[0][0]}
    block = primary['block']
    blocks_analyzed = [block]
    if secondary and not args.no_secondary and secondary['block'] != block:
        blocks_analyzed.append(secondary['block'])

    p1_ids = phase1_sample_ids(p1_per_sample)
    print('\n  --- Phase 1 inputs ---')
    print(f'  summary     : {os.path.relpath(P1_SUMMARY, _REPO)}')
    print(f'  per-sample  : {os.path.relpath(P1_PER_SAMPLE, _REPO)}')
    print(f'  metadata    : {os.path.relpath(P1_META, _REPO)}')
    print(f'  Phase 1 run : {p1_meta["execution_date"]} {p1_meta["execution_local_time"]} '
          f'{p1_meta["timezone"]}, {p1_meta["valid_triplets_analyzed"]} valid triplets')
    print('\n  --- block selection (from Phase 1, not hardcoded) ---')
    for m, b, fr in ranked:
        tag = ('  <== PRIMARY' if b == block else
               ('  <== secondary' if secondary and b == secondary['block']
                and b in blocks_analyzed else ''))
        print(f'    Block {b:>2} ({fr}): centered same-scene 1e5<->1e7 mean {m:+.4f}{tag}')
    print(f'  selected block   : {block}  ({primary["fraction"]} depth)')
    print(f'  Phase 1 mean     : {primary["mean"]:+.4f}')
    print(f'  selection reason : highest mean CENTERED same-scene corresponding-'
          f'patch cosine for 1e5<->1e7 across the Phase 1 validation set'
          + ('  [OVERRIDDEN by --block]' if args.block is not None else ''))
    if len(blocks_analyzed) > 1:
        print(f'  secondary block  : {blocks_analyzed[1]} (context only; '
              f'conclusions come from Block {block})')

    # ---------------- model ----------------
    with open(args.opt) as f:
        cfg = yaml.safe_load(f)
    net = cfg['network_g']
    blocks0 = [b - 1 for b in blocks_analyzed]
    ext = base.build_extractor(cfg, tuple(blocks0), args.device)
    n_blocks = len(ext.dino.blocks)
    patch = int(ext.dino.patch_size)
    img_size = int(net['dino_img_size'])
    embed_dim = int(ext.dino.embed_dim)
    n_reg = int(getattr(ext.dino, 'num_register_tokens', 0))
    gh = gw = img_size // patch
    ext.layers = tuple(blocks0)

    print('\n  --- DINO (identical to Phase 1) ---')
    print(f'  model / checkpoint : {net["dino_model_name"]}  |  {net["dino_weights"]}')
    print(f'  strict load        : missing={list(ext.load_result.missing_keys)} '
          f'unexpected={list(ext.load_result.unexpected_keys)}')
    print(f'  blocks (1-idx)     : {blocks_analyzed} -> 0-idx {blocks0}')
    print(f'  patch grid         : {gh} x {gw} = {gh * gw} tokens, {embed_dim}-d, '
          f'patch {patch}, input {img_size}')
    print(f'  special tokens     : CLS + {n_reg} registers removed by DINOv2; '
          f'spatial patch tokens only, no pooling')
    for key, want, got in (('model', p1_meta['dino_model'], net['dino_model_name']),
                           ('checkpoint', p1_meta['dino_checkpoint'], net['dino_weights']),
                           ('patch_grid', p1_meta['patch_grid'], [gh, gw]),
                           ('embedding_dim', p1_meta['embedding_dim'], embed_dim)):
        if want != got:
            raise SystemExit(f'DINO config differs from Phase 1 ({key}: '
                             f'Phase1={want!r} now={got!r}) — refusing to compare')
    print('  verified identical to Phase 1 metadata (model, checkpoint, grid, dim)')

    # ---------------- centering ----------------
    mode = args.centering_mode or p1_meta['centering']['mode']
    if not os.path.isfile(args.means_cache):
        raise SystemExit(f'centering means missing: {args.means_cache} — Phase 2 '
                         f'must reuse the Phase 1 training means, not recompute '
                         f'them from validation data')
    means = torch.load(args.means_cache, map_location='cpu', weights_only=False)
    mm = means['meta']
    if mm.get('split') != 'train':
        raise SystemExit(f'means come from split {mm.get("split")!r}, not train — '
                         f'refusing (validation leakage)')
    if mm['embed_dim'] != embed_dim:
        raise SystemExit('cached means have the wrong width for this model')
    means_blocks1 = mm['blocks_1indexed']
    for b in blocks_analyzed:
        if b not in means_blocks1:
            raise SystemExit(f'no cached spatial mean for block {b} '
                             f'(cache has {means_blocks1})')
    layer_idx = {b: means_blocks1.index(b) for b in blocks_analyzed}
    print('\n  --- centering (reused from Phase 1, training-derived) ---')
    print(f'  cache        : {os.path.relpath(args.means_cache, _REPO)}')
    print(f'  statistic    : {mm["statistic"]}')
    print(f'  source       : {mm["n_train_images"]} TRAIN images (seed {mm["seed"]}) '
          f'— no validation leakage')
    print(f'  mode         : {mode}')

    # ---------------- sample set ----------------
    ds = base.build_dataset(cfg, p1_meta['split'])
    id_to_index = {os.path.splitext(os.path.basename(p['gt_path']))[0]: n
                   for n, p in enumerate(ds.paths)}
    use_ids = p1_ids[:args.max_samples] if smoke else p1_ids
    ids, missing = [], []
    for sid in use_ids:
        if sid not in id_to_index:
            missing.append({'sample_id': sid, 'reason': 'not present in the split'})
            continue
        _, _, batch, reason = p1.verify_triplet(ds, id_to_index[sid])
        if reason is not None:
            missing.append({'sample_id': sid, 'reason': reason})
            continue
        ids.append(sid)
    n = len(ids)
    print('\n  --- sample set (reused from Phase 1) ---')
    print(f'  Phase 1 valid triplets : {p1_meta["valid_triplets_analyzed"]}')
    print(f'  requested here         : {len(use_ids)}'
          + ('  (smoke subset)' if smoke else ''))
    print(f'  Phase 2 valid triplets : {n}')
    print(f'  not reproduced         : {len(missing)}')
    for m_ in missing:
        print(f'    {m_["sample_id"]}: {m_["reason"]}')
    if n < 2:
        raise SystemExit('need at least 2 valid triplets')

    # ---------------- different-scene control ----------------
    perm = derangement(n, args.seed)
    control_id = {ids[i]: ids[perm[i]] for i in range(n)}
    assert all(k != v for k, v in control_id.items())
    print(f'\n  --- different-scene control ---')
    print(f'  construction : seeded derangement (no fixed points), seed {args.seed}, '
          f'n = {n}')
    print(f'  same mapping used for BOTH candidate sources (fair comparison)')
    print(f'  example      : ' + ', '.join(f'{ids[i]}->{control_id[ids[i]]}'
                                           for i in range(min(4, n))))

    # ---------------- pass 1: clean (1e7) targets ----------------
    print(f'\n  --- pass 1/2: extracting DINO(1e7) targets for {n} samples ---')
    clean = {}
    for c, sid in enumerate(ids, 1):
        _, f = extract(ext, ds[id_to_index[sid]], ['1e7'], blocks_analyzed,
                       img_size, args.device)
        if f['1e7'][blocks_analyzed[0]].shape[0] != gh * gw:
            raise SystemExit(f'{sid}: token count != {gh}*{gw}')
        clean[sid] = f['1e7']
        if c % 50 == 0 or c == n:
            print(f'    {c}/{n}')

    # ---------------- pass 2: candidate sources ----------------
    print(f'  --- pass 2/2: extracting DINO(1e5) and DINO(Render), '
          f'computing metrics ---')
    per_path = os.path.join(out_dir, 'dino_prior_source_per_sample.csv')
    cols = (['sample_id', 'feature_type', 'block_index', 'layer_fraction',
             'is_primary_block'] + METRICS + ['different_scene_target_sample_id'])
    frac_of = {b: next(fr for m_, bb, fr in ranked if bb == b)
               for b in blocks_analyzed}

    with open(per_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for c, sid in enumerate(ids, 1):
            _, src = extract(ext, ds[id_to_index[sid]], ['1e5', 'render'],
                             blocks_analyzed, img_size, args.device)
            j = control_id[sid]
            for b in blocks_analyzed:
                for ft in FEATURE_TYPES:
                    if ft == 'raw':
                        a = src['1e5'][b]; r = src['render'][b]
                        t = clean[sid][b]; t_diff = clean[j][b]
                    else:
                        li = layer_idx[b]
                        a = centered(src['1e5'][b], means, '1e5', li, mode)
                        r = centered(src['render'][b], means, 'render', li, mode)
                        t = centered(clean[sid][b], means, '1e7', li, mode)
                        t_diff = centered(clean[j][b], means, '1e7', li, mode)
                    s_1e5 = base.patchwise_cosine(a, t)
                    s_ren = base.patchwise_cosine(r, t)
                    d_1e5 = base.patchwise_cosine(a, t_diff)
                    d_ren = base.patchwise_cosine(r, t_diff)
                    vals = {
                        'similarity_1e5_to_clean': s_1e5,
                        'similarity_render_to_clean': s_ren,
                        'delta_render_over_1e5': s_ren - s_1e5,
                        'different_scene_1e5_to_clean': d_1e5,
                        'different_scene_render_to_clean': d_ren,
                        'scene_advantage_1e5': s_1e5 - d_1e5,
                        'scene_advantage_render': s_ren - d_ren,
                        'delta_scene_advantage': (s_ren - d_ren) - (s_1e5 - d_1e5),
                        'similarity_1e5_to_render': base.patchwise_cosine(a, r),
                    }
                    w.writerow({'sample_id': sid, 'feature_type': ft,
                                'block_index': b, 'layer_fraction': frac_of[b],
                                'is_primary_block': int(b == block),
                                'different_scene_target_sample_id': j,
                                **{k: f'{v:.6f}' for k, v in vals.items()}})
            if c % 50 == 0 or c == n:
                print(f'    {c}/{n}')
    print(f'  wrote {os.path.relpath(per_path, _REPO)}')

    # ---------------- cross-check against Phase 1 ----------------
    # similarity_1e5_to_clean IS Phase 1's similarity_1e5_vs_1e7 -- same features,
    # same centering, same samples. If the two disagree, something is misaligned
    # (this check exists because an earlier version indexed DINOv2's ascending
    # block output positionally against a caller-ordered block list, silently
    # pairing block-3 features with the block-6 mean).
    rows = list(csv.DictReader(open(per_path)))
    if not args.skip_consistency_check:
        if args.centering_mode and args.centering_mode != p1_meta['centering']['mode']:
            print('\n  Phase 1 cross-check SKIPPED: --centering-mode differs from '
                  'Phase 1, so the centered values are not expected to match')
        else:
            p1_lookup = {(r['sample_id'], r['feature_type'], int(r['block_index'])):
                         float(r['similarity_1e5_vs_1e7']) for r in p1_per_sample}
            checked, worst, worst_key = 0, 0.0, None
            for r in rows:
                k = (r['sample_id'], r['feature_type'], int(r['block_index']))
                if k not in p1_lookup:
                    continue
                diff = abs(float(r['similarity_1e5_to_clean']) - p1_lookup[k])
                checked += 1
                if diff > worst:
                    worst, worst_key = diff, k
            if checked == 0:
                raise SystemExit('Phase 1 cross-check found no comparable rows')
            if worst > 1e-4:
                raise SystemExit(
                    f'PHASE 1 CROSS-CHECK FAILED: similarity_1e5_to_clean differs '
                    f'from Phase 1 similarity_1e5_vs_1e7 by {worst:.6f} at '
                    f'{worst_key} (tolerance 1e-4). The two must be the identical '
                    f'quantity -- features and blocks are misaligned somewhere. '
                    f'Results NOT trustworthy; not writing plots or devlog.')
            print(f'\n  Phase 1 cross-check PASSED: {checked} rows match Phase 1 '
                  f'similarity_1e5_vs_1e7 (max abs diff {worst:.2e})')

    # ---------------- summary (from the saved CSV) ----------------
    series, stats = {}, {}
    for r in rows:
        key = (r['feature_type'], int(r['block_index']))
        for m_ in METRICS:
            series.setdefault((*key, m_), []).append(float(r[m_]))
    for (ft, b, m_), vals in series.items():
        st = describe(vals)
        if m_ in PAIRED:
            st.update(paired_stats(vals, args.tie_tol))
        stats[(ft, b, m_)] = st

    scols = ['feature_type', 'block_index', 'layer_fraction', 'is_primary_block',
             'metric', 'n', 'mean', 'std', 'sem', 'median', 'q1', 'q3', 'min', 'max',
             'pct_positive', 'pct_negative', 'pct_tied', 'cohens_dz',
             'wilcoxon_stat', 'wilcoxon_p', 'tie_tolerance']
    sum_path = os.path.join(out_dir, 'dino_prior_source_summary.csv')
    with open(sum_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=scols)
        w.writeheader()
        for b in blocks_analyzed:
            for ft in FEATURE_TYPES:
                for m_ in METRICS:
                    st = stats[(ft, b, m_)]
                    row = {'feature_type': ft, 'block_index': b,
                           'layer_fraction': frac_of[b],
                           'is_primary_block': int(b == block), 'metric': m_,
                           'tie_tolerance': args.tie_tol if m_ in PAIRED else ''}
                    for k in scols:
                        if k in row:
                            continue
                        v = st.get(k, '')
                        row[k] = (f'{v:.6f}' if isinstance(v, float) else v)
                    w.writerow(row)
    print(f'  wrote {os.path.relpath(sum_path, _REPO)}')

    # ---------------- printed summary ----------------
    for ft in FEATURE_TYPES:
        s = lambda m_, f='mean': stats[(ft, block, m_)][f]      # noqa: E731
        print('\n' + '=' * 78)
        print(f'PHASE 2 — {ft.upper()} FEATURES     Selected block: B{block} '
              f'({primary["fraction"]} depth), n = {n}')
        print('=' * 78)
        print('  Absolute clean alignment (same scene):')
        print(f'    1e5    -> clean : mean {s("similarity_1e5_to_clean"):+.4f}   '
              f'median {s("similarity_1e5_to_clean", "median"):+.4f}   '
              f'sd {s("similarity_1e5_to_clean", "std"):.4f}')
        print(f'    Render -> clean : mean {s("similarity_render_to_clean"):+.4f}   '
              f'median {s("similarity_render_to_clean", "median"):+.4f}   '
              f'sd {s("similarity_render_to_clean", "std"):.4f}')
        print(f'    Render wins absolute similarity on : '
              f'{s("delta_render_over_1e5", "pct_positive"):.1f}% of samples')
        print(f'    1e5    wins absolute similarity on : '
              f'{s("delta_render_over_1e5", "pct_negative"):.1f}% of samples')
        print(f'    tied (|delta| <= {args.tie_tol})            : '
              f'{s("delta_render_over_1e5", "pct_tied"):.1f}%')
        print(f'    mean direct delta (Render - 1e5)   : '
              f'{s("delta_render_over_1e5"):+.4f}   '
              f'(median {s("delta_render_over_1e5", "median"):+.4f})')
        print('\n  Scene-specific advantage (same-scene minus different-scene):')
        print(f'    1e5    : same {s("similarity_1e5_to_clean"):+.4f}   '
              f'different {s("different_scene_1e5_to_clean"):+.4f}   '
              f'advantage {s("scene_advantage_1e5"):+.4f} '
              f'(median {s("scene_advantage_1e5", "median"):+.4f})')
        print(f'    Render : same {s("similarity_render_to_clean"):+.4f}   '
              f'different {s("different_scene_render_to_clean"):+.4f}   '
              f'advantage {s("scene_advantage_render"):+.4f} '
              f'(median {s("scene_advantage_render", "median"):+.4f})')
        print(f'    Render has larger scene-specific advantage on : '
              f'{s("delta_scene_advantage", "pct_positive"):.1f}% of samples')
        print(f'    1e5    has larger scene-specific advantage on : '
              f'{s("delta_scene_advantage", "pct_negative"):.1f}% of samples')
        print(f'    mean delta_scene_advantage : {s("delta_scene_advantage"):+.4f}   '
              f'(median {s("delta_scene_advantage", "median"):+.4f})')
        print('\n  Paired tests (Wilcoxon signed-rank, two-sided) and magnitude:')
        for m_, lab in (('delta_render_over_1e5', 'absolute similarity'),
                        ('delta_scene_advantage', 'scene advantage')):
            print(f'    {lab:20s}: W = {s(m_, "wilcoxon_stat"):.1f}   '
                  f'p = {s(m_, "wilcoxon_p"):.3e}   n = {s(m_, "n")}   '
                  f'mean diff {s(m_):+.4f}   median diff {s(m_, "median"):+.4f}   '
                  f'dz = {s(m_, "cohens_dz"):+.3f}')
        print(f'\n  Complementarity: same-scene 1e5<->Render mean '
              f'{s("similarity_1e5_to_render"):+.4f} '
              f'(median {s("similarity_1e5_to_render", "median"):+.4f}); '
              f'Phase 1 different-scene control for this pair/block: '
              f'{phase1_control(p1_summary, block, ft, "1e5_vs_render"):+.4f}')
    print('\n  NOTE: n is large, so small differences can reach p < 0.05. Read the '
          'mean/median paired difference and dz, not the p-value alone.')
    print('  NOTE: this measures alignment to the clean DINO representation only. '
          'It does NOT predict restoration PSNR/SSIM — that is Phase 3.')

    # ---------------- plots (primary block, centered) ----------------
    print()
    prim = [r for r in rows
            if int(r['block_index']) == block and r['feature_type'] == 'centered']
    a_abs = [float(r['similarity_1e5_to_clean']) for r in prim]
    b_abs = [float(r['similarity_render_to_clean']) for r in prim]
    a_adv = [float(r['scene_advantage_1e5']) for r in prim]
    b_adv = [float(r['scene_advantage_render']) for r in prim]
    d_abs = [float(r['delta_render_over_1e5']) for r in prim]

    plot_paired(os.path.join(out_dir, 'dino_prior_source_paired_comparison.png'),
                a_abs, b_abs, block, 'centered', len(prim),
                'Corresponding-patch cosine to DINO(1e7)',
                'Absolute alignment with the clean 1e7 DINO representation',
                ['1e5 → clean', 'Render → clean'], args.dpi)
    plot_paired(os.path.join(out_dir, 'dino_prior_source_scene_advantage.png'),
                a_adv, b_adv, block, 'centered', len(prim),
                'Same-scene cosine − different-scene cosine',
                'Scene-specific advantage over the different-scene control',
                ['1e5 advantage', 'Render advantage'], args.dpi)
    plot_delta_hist(os.path.join(out_dir, 'dino_prior_source_delta_histogram.png'),
                    d_abs, block, 'centered', len(prim), args.dpi)

    # ---------------- representative samples ----------------
    by_delta = sorted(((float(r['delta_render_over_1e5']), r['sample_id'])
                       for r in prim), reverse=True)
    by_adv = sorted(((float(r['delta_scene_advantage']), r['sample_id'])
                     for r in prim), reverse=True)
    near = min(prim, key=lambda r: abs(float(r['delta_render_over_1e5'])))

    def tag_for(val, wins_tag, fallback_tag, wins_why, fallback_why):
        """Name the file after what the number actually says.

        If e.g. every sample has delta > 0, there IS no '1e5 wins' sample, and
        calling one that would be a mislabelled figure.
        """
        return (wins_tag, wins_why) if val else (fallback_tag, fallback_why)

    t_hi, w_hi = tag_for(
        by_delta[0][0] > 0, 'render_wins', 'render_best_case',
        'Render beats 1e5 by the largest margin (absolute)',
        'smallest 1e5 advantage; Render never wins outright in this split')
    t_lo, w_lo = tag_for(
        by_delta[-1][0] < 0, '1e5_wins', '1e5_best_case',
        '1e5 beats Render by the largest margin (absolute)',
        'smallest Render advantage; 1e5 never wins outright in this split')
    ta_hi, wa_hi = tag_for(
        by_adv[0][0] > 0, 'render_scene_advantage', 'render_best_scene_case',
        'largest scene-specific advantage for Render',
        'smallest 1e5 scene-advantage lead; Render never leads in this split')
    ta_lo, wa_lo = tag_for(
        by_adv[-1][0] < 0, '1e5_scene_advantage', '1e5_best_scene_case',
        'largest scene-specific advantage for 1e5',
        'smallest Render scene-advantage lead; 1e5 never leads in this split')
    reps = [
        (t_hi, by_delta[0][1], by_delta[0][0], w_hi),
        (t_lo, by_delta[-1][1], by_delta[-1][0], w_lo),
        ('near_tie', near['sample_id'], float(near['delta_render_over_1e5']),
         'the sample closest to tied on absolute alignment (smallest |delta|)'),
        (ta_hi, by_adv[0][1], by_adv[0][0], wa_hi),
        (ta_lo, by_adv[-1][1], by_adv[-1][0], wa_lo),
    ]
    print('\n  representative samples (centered, B%d):' % block)
    for tag, sid, val, why in reps:
        print(f'    {tag:24s} {sid}  ({val:+.4f})  — {why}')

    # the panels show all Phase-1 blocks so the selected block sits in context
    all_blocks1 = p1_meta['selected_blocks_1indexed']
    all_blocks0 = p1_meta['selected_blocks_0indexed']
    all_fracs = p1_meta['layer_fractions']
    ext.layers = tuple(all_blocks0)
    rep_meta = []
    for tag, sid, val, why in reps:
        _, _, batch, reason = p1.verify_triplet(ds, id_to_index[sid])
        if reason:
            print(f'  [skip figure] {sid}: {reason}')
            continue
        inputs, fd = extract(ext, batch, base.DOMAINS, all_blocks1, img_size,
                             args.device)
        # -> list ordered like all_blocks1, centred with THAT block's own mean
        feats = {d: [centered(fd[d][b], means, d, means_blocks1.index(b), mode)
                     for b in all_blocks1] for d in base.DOMAINS}
        title = (f'Spatial DINO PCA-1 — Very Noisy Radar (1e5) vs Clean Radar '
                 f'(1e7) vs Render\nPhase 2 representative: {tag.replace("_", " ")}'
                 f'   |   sample {sid}   |   B{block} centered, '
                 f'delta = {val:+.4f}\ncentered features ({mode} training mean); '
                 f'one joint PCA per layer, shared display range per layer')
        path = os.path.join(rep_dir, f'phase2_{tag}_sample_{sid}.png')
        p1.pca_figure_for_sample(path, title, inputs, feats, all_blocks1,
                                 all_fracs, gh, gw, ext, args, args.seed)
        rep_meta.append({'tag': tag, 'sample_id': sid, 'value': val,
                         'criterion': why,
                         'file': os.path.relpath(path, _REPO)})
    ext.layers = tuple(blocks0)

    # ---------------- metadata ----------------
    meta = {
        'phase': 2,
        'execution_date': started.strftime('%Y-%m-%d'),
        'execution_local_time': started.strftime('%H:%M:%S'),
        'timezone': started.tzname(),
        'utc_offset': started.strftime('%z'),
        'finished_local_time': datetime.datetime.now().astimezone().strftime('%H:%M:%S'),
        'command': ' '.join([os.path.relpath(sys.argv[0], _REPO)] + sys.argv[1:]),
        'smoke_test': smoke,
        'output_dir': os.path.relpath(out_dir, _REPO),
        'phase1_source_files': {
            'summary': os.path.relpath(P1_SUMMARY, _REPO),
            'per_sample': os.path.relpath(P1_PER_SAMPLE, _REPO),
            'metadata': os.path.relpath(P1_META, _REPO),
            'phase1_executed': f'{p1_meta["execution_date"]} '
                               f'{p1_meta["execution_local_time"]} {p1_meta["timezone"]}',
        },
        'selected_block': block,
        'selected_block_layer_fraction': primary['fraction'],
        'selected_block_phase1_mean_1e5_vs_1e7_centered': primary['mean'],
        'block_selection_reason':
            'highest mean CENTERED same-scene corresponding-patch cosine for '
            '1e5<->1e7 in the Phase 1 validation summary'
            + (' (OVERRIDDEN by --block)' if args.block is not None else ''),
        'phase1_block_ranking_centered_1e5_vs_1e7': [
            {'block_index': b, 'layer_fraction': fr, 'mean': m_}
            for m_, b, fr in ranked],
        'blocks_analyzed': blocks_analyzed,
        'secondary_block': (blocks_analyzed[1] if len(blocks_analyzed) > 1 else None),
        'split': p1_meta['split'],
        'phase1_valid_triplets': p1_meta['valid_triplets_analyzed'],
        'phase2_requested_samples': len(use_ids),
        'phase2_valid_triplets': n,
        'phase1_samples_not_reproduced': missing,
        'dino_model': net['dino_model_name'],
        'dino_checkpoint': net['dino_weights'],
        'dino_hub_source': net['dino_hub_source'],
        'transformer_block_count': n_blocks,
        'patch_size': patch,
        'patch_grid': [gh, gw],
        'n_patch_tokens': gh * gw,
        'embedding_dim': embed_dim,
        'dino_input_resolution': [img_size, img_size],
        'num_register_tokens': n_reg,
        'special_token_handling': 'CLS + registers removed by DINOv2; spatial '
                                  'patch tokens only, no pooling, no CLS-only',
        'feature_types': FEATURE_TYPES,
        'centering': {
            'mode': mode,
            'cache_file': os.path.relpath(args.means_cache, _REPO),
            'statistic': mm['statistic'],
            'source_split': mm.get('split'),
            'n_training_images': mm['n_train_images'],
            'recomputed_in_phase2': False,
            'validation_leakage': False,
        },
        'primary_metric': 'mean corresponding-patch cosine similarity on the full '
                          f'{embed_dim}-d spatial patch features (never PCA values)',
        'different_scene_control': {
            'construction': 'seeded derangement of the Phase 1 sample-ID list '
                            '(random.Random(seed).shuffle with fixed-point repair)',
            'seed': args.seed,
            'deterministic': True,
            'no_fixed_points': True,
            'same_mapping_for_both_sources': True,
            'pairing': control_id,
        },
        'statistical_tests': {
            'test': 'paired Wilcoxon signed-rank (two-sided)',
            'scipy_available': _HAVE_SCIPY,
            'applied_to': list(PAIRED.keys()),
            'effect_size': "Cohen's dz on the paired difference; win rates with "
                           f'tie tolerance {args.tie_tol}',
            'caveat': 'n is large; statistical significance does not imply a '
                      'practically meaningful difference',
        },
        'representative_sample_criteria': [
            {'tag': t, 'criterion': w_} for t, _, _, w_ in reps],
        'representative_samples': rep_meta,
        'limitation': 'Phase 2 measures alignment to the clean DINO spatial '
                      'representation only. It makes no claim about restoration '
                      'PSNR/SSIM, which is Phase 3.',
        'random_seed': args.seed,
        'device': args.device,
        'torch_version': torch.__version__,
    }
    meta_path = os.path.join(out_dir, 'phase2_metadata.json')
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
            'block': block, 'fraction': primary['fraction'],
            'p1_mean': primary['mean'], 'n': n,
            'p1_n': p1_meta['valid_triplets_analyzed'],
            'missing_str': (f'{len(missing)} Phase 1 samples not reproduced'
                            if missing else 'all Phase 1 samples reproduced'),
            'secondary_str': (f'B{blocks_analyzed[1]}' if len(blocks_analyzed) > 1
                              else 'not evaluated'),
            'model': net['dino_model_name'], 'checkpoint': net['dino_weights'],
            'gh': gh, 'gw': gw, 'ntok': gh * gw, 'embed_dim': embed_dim,
            'patch': patch, 'img_size': img_size, 'centering_mode': mode,
            'n_mean_images': mm['n_train_images'], 'seed': args.seed,
            'p1_ctrl_1e5_render': phase1_control(p1_summary, block, 'centered',
                                                 '1e5_vs_render'),
            'rep_str': ', '.join(f'{t} = {s_}' for t, s_, _, _ in reps),
        }
        append_devlog(started.strftime('%Y-%m-%d %H:%M %Z'), rows, stats, ctx)

    print('\nPHASE 2 DONE')


if __name__ == '__main__':
    main()
