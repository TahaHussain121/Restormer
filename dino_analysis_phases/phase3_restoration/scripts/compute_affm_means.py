"""Per-layer production centering means for the AFFM arm, blocks {3,6,9,12}.

THIS SCRIPT DOES NOT REIMPLEMENT ANYTHING. It imports
`compute_production_means` and rebinds only its block constants, so the
sampling, the crop draw, the preprocessing, the accumulation dtype, the token
count and the metadata are the SAME CODE OBJECTS that produced
`render_B6_train128_dino224_mean.pt`. The only difference between this run and
that one is which block index is asked of `dino_shared.extract_block`.

That matters for one reason above all: the crop RNG is keyed
`f'{seed}:{stem}:{regime}'`, with no block in the key, so every layer sees the
IDENTICAL crops the existing B6 mean saw. Recomputing B6 through this path must
therefore reproduce the existing file to floating-point tolerance, and
`--verify-against` makes that a hard check rather than a hope.

Train split only, asserted upstream by the imported code. Never val, never test.
Never the 1e7 target.

Usage:
  # a new layer -> writes means/render_B3_train128_dino224_mean.pt
  compute_affm_means.py --block 3 --regime train128 --domain render

  # the correctness check -> writes nothing into means/, only compares
  compute_affm_means.py --block 6 --regime train128 --domain render \
      --verify-against .../means/render_B6_train128_dino224_mean.pt \
      --outdir <scratch>
"""

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE3 = os.path.dirname(_HERE)
_REPO = os.path.dirname(os.path.dirname(_PHASE3))
for p in (_REPO, os.path.join(_REPO, 'dino_analysis_phases'), _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import compute_production_means as base                      # noqa: E402
import dino_shared                                           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--block', type=int, required=True,
                    help='1-indexed DINO block (3, 6, 9 or 12)')
    ap.add_argument('--verify-against', default=None,
                    help='an existing mean .pt this run must reproduce')
    args, rest = ap.parse_known_args()

    # --- the ONLY mutation: which block the shared code extracts -------------
    base.BLOCK_1IDX = int(args.block)
    base.BLOCK_0IDX = dino_shared.b1_to_b0(args.block)
    print(f'[affm-means] block B{base.BLOCK_1IDX} -> 0-indexed '
          f'{base.BLOCK_0IDX}; every other step is compute_production_means '
          f'unchanged')

    sys.argv = [sys.argv[0]] + rest
    base.main()

    if args.verify_against:
        # base.main() has already written its file; find it the same way it did
        ns = _parse_rest(rest)
        produced = os.path.join(ns['outdir'],
                                base.mean_filename(ns['domain'], ns['regime']))
        a = _load(produced)
        b = _load(args.verify_against)
        if a.shape != b.shape:
            raise SystemExit(f'shape {tuple(a.shape)} != {tuple(b.shape)}')
        d = float((a - b).abs().max())
        rel = d / float(b.abs().max())
        print(f'\n[affm-means] VERIFY B{args.block} {ns["regime"]}')
        print(f'  recomputed : {produced}')
        print(f'  reference  : {args.verify_against}')
        print(f'  shape      : {tuple(a.shape)}   norms '
              f'{float(a.norm()):.6f} vs {float(b.norm()):.6f}')
        print(f'  MAX ABS DIFF = {d:.3e}   (relative to max|mu| = {rel:.3e})')
        print(f'  -> {"PASS" if rel < 1e-5 else "FAIL"}')
        if rel >= 1e-5:
            raise SystemExit('recomputed B6 mean does not match the mean '
                             'addition-render trains with; the extraction or '
                             'preprocessing path differs from training')


def _parse_rest(rest):
    out = {'outdir': os.path.join(_PHASE3, 'means'), 'domain': '1e5',
           'regime': None}
    for i, tok in enumerate(rest):
        if tok in ('--outdir', '--domain', '--regime') and i + 1 < len(rest):
            out[tok[2:]] = rest[i + 1]
    return out


def _load(path):
    obj = torch.load(path, map_location='cpu', weights_only=False)
    mu = obj['mean'] if isinstance(obj, dict) else obj
    return torch.as_tensor(mu, dtype=torch.float32).reshape(-1)


if __name__ == '__main__':
    main()
