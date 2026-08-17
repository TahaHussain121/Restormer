"""WORK ORDER 1, Task 1.2 -- does the shared Phase-3 extractor reproduce the
Phase-1 representation exactly?

OLD path (Phase 1, unchanged code, imported not copied):
    dino_analysis_phases/visualize_dino_spatial_pca.py
      build_dataset(cfg,'val') -> ds[i] -> batch_to_dino_inputs(224)
      -> build_extractor -> spatial_tokens(...)            [1,256,768] per block

NEW path (Phase 3):
    phase3_restoration/scripts/dino_shared.py
      build_dino -> extract_blocks(img01, blocks0, dino_size=224)

Identical preprocessing on both sides: the full 256x256 image, [0,1], resized
to 224. This measures IMPLEMENTATION FIDELITY ONLY. It is not the layer
selection experiment (that is Task 1.3, at the 128 crop scale).

Read-only: loads data and writes one JSON under
phase3_restoration/results/wo1_verification/.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys

import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE3 = os.path.dirname(_HERE)
_REPO = os.path.dirname(os.path.dirname(_PHASE3))
for p in (_REPO, os.path.join(_REPO, 'dino_analysis_phases')):
    if p not in sys.path:
        sys.path.insert(0, p)

import visualize_dino_spatial_pca as base        # noqa: E402  (the Phase-1 lib)
import dino_shared                                # noqa: E402

BLOCKS1 = [3, 6, 9, 12]
BLOCKS0 = [b - 1 for b in BLOCKS1]
DOMAINS = ['1e5', '1e7']                          # render not used by Phase 3


def git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=_REPO).decode().strip()
    except Exception:                              # noqa: BLE001
        return 'unknown'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--opt', default=os.path.join(
        _REPO, 'Deraining_Holo', 'Options', 'DINO_analysis_data.yml'))
    ap.add_argument('--split', default='val')
    ap.add_argument('--sample-ids', nargs='+',
                    default=['0196', '6115', '3017', '4467'])
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--atol', type=float, default=1e-6)
    ap.add_argument('--rtol', type=float, default=1e-5)
    ap.add_argument('--out', default=os.path.join(
        _PHASE3, 'results', 'wo1_verification', 'phase1_equivalence.json'))
    args = ap.parse_args()

    with open(args.opt) as f:
        cfg = yaml.safe_load(f)
    img_size = int(cfg['network_g']['dino_img_size'])
    assert img_size == 224, f'Phase 1 ran at 224, yml says {img_size}'

    ds = base.build_dataset(cfg, args.split)
    stems = [os.path.splitext(os.path.basename(p['gt_path']))[0]
             for p in ds.paths]

    print(f'== OLD path extractor (Phase 1: build_extractor) ==')
    ext_old = base.build_extractor(cfg, BLOCKS0, args.device)
    print(f'   load_result missing={list(ext_old.load_result.missing_keys)} '
          f'unexpected={list(ext_old.load_result.unexpected_keys)}')
    print(f'== NEW path extractor (Phase 3: dino_shared.build_dino) ==')
    ext_new = dino_shared.build_dino(device=args.device)

    results = []
    for sid in args.sample_ids:
        if sid not in stems:
            raise SystemExit(f'sample {sid} not in {args.split} split')
        i = stems.index(sid)
        batch = ds[i]

        # --- OLD --------------------------------------------------------
        inputs = base.batch_to_dino_inputs(
            batch, img_size, ext_old.mean.cpu(), ext_old.std.cpu())
        old_tok = {}
        for d in DOMAINS:
            per_block = base.spatial_tokens(
                ext_old, inputs[d].to(args.device), BLOCKS0)
            for k, b0 in enumerate(sorted(BLOCKS0)):
                old_tok[(d, b0)] = per_block[k].float().cpu()

        # --- NEW --------------------------------------------------------
        new_tok = {}
        new_pre = {}
        for d in DOMAINS:
            img01 = batch[base.DOMAIN_BATCH_KEY[d]].unsqueeze(0)   # [1,C,256,256]
            new_pre[d] = dino_shared.preprocess(img01, img_size)
            got = dino_shared.extract_blocks(
                ext_new, img01.to(args.device), BLOCKS0, img_size)
            for b0, f in got.items():
                new_tok[(d, b0)] = f.float().cpu()

        for d in DOMAINS:
            # preprocessed-input equivalence, before the ViT
            pre_err = float((new_pre[d] - inputs[d]).abs().max())
            for b0 in sorted(BLOCKS0):
                a, b = old_tok[(d, b0)], new_tok[(d, b0)]
                assert a.shape == b.shape, (a.shape, b.shape)
                diff = (a - b).abs()
                rec = {
                    'sample_id': sid, 'domain': d,
                    'block_1indexed': b0 + 1, 'block_0indexed': b0,
                    'shape': list(a.shape),
                    'preproc_max_abs_err': pre_err,
                    'max_abs_err': float(diff.max()),
                    'mean_abs_err': float(diff.mean()),
                    'allclose': bool(torch.allclose(a, b, atol=args.atol,
                                                    rtol=args.rtol)),
                    'bitwise_identical': bool(torch.equal(a, b)),
                    'old_norm': float(a.norm()), 'new_norm': float(b.norm()),
                }
                results.append(rec)
                print(f'  {sid} {d:>6} B{b0 + 1:<2} shape {rec["shape"]} '
                      f'maxabs {rec["max_abs_err"]:.3e} '
                      f'meanabs {rec["mean_abs_err"]:.3e} '
                      f'allclose {rec["allclose"]} '
                      f'bitwise {rec["bitwise_identical"]}')

        # grid reshape check (row-major ordering, vs DINOv2's own reshape=True)
        x = dino_shared.preprocess(
            batch['lq'].unsqueeze(0).to(args.device), img_size)
        ref = ext_new.dino.get_intermediate_layers(
            x, n=(5,), reshape=True, return_class_token=False, norm=True)[0]
        mine = dino_shared.tokens_to_grid(new_tok[('1e5', 5)].to(args.device))
        grid_ok = bool(torch.equal(ref.float().cpu(), mine.float().cpu()))
        print(f'  {sid} tokens_to_grid == DINOv2 reshape=True : {grid_ok} '
              f'{list(mine.shape)}')

    summary = {
        'max_abs_err_overall': max(r['max_abs_err'] for r in results),
        'mean_abs_err_overall': sum(r['mean_abs_err'] for r in results) / len(results),
        'all_allclose': all(r['allclose'] for r in results),
        'all_bitwise_identical': all(r['bitwise_identical'] for r in results),
        'atol': args.atol, 'rtol': args.rtol,
        'n_comparisons': len(results),
        'grid_reshape_matches_dinov2': grid_ok,
    }
    print('\nSUMMARY', json.dumps(summary, indent=2))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({
            'task': 'WO1 Task 1.2 -- Phase-1 extractor equivalence',
            'created': datetime.datetime.now().astimezone().isoformat(),
            'git_commit': git_commit(),
            'hostname': os.uname().nodename,
            'device': args.device,
            'torch': torch.__version__,
            'split': args.split, 'sample_ids': args.sample_ids,
            'blocks_1indexed': BLOCKS1, 'blocks_0indexed': BLOCKS0,
            'dino_input_size': img_size,
            'radar_input': 'full 256x256 image (Phase-1 protocol)',
            'summary': summary, 'per_comparison': results,
        }, f, indent=2)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
