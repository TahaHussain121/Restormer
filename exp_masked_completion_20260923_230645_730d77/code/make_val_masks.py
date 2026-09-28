"""Build the FIXED synthetic-completion validation task, once, before training.

339 validation clean targets are damaged by the fixed recipe with a dedicated
seed (4242). The masks and damaged images are stored inside the experiment root
so checkpoint selection is reproducible and identical at every validation check.
Refuses to regenerate: the task is fixed once.
"""
import os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mc_common as mc                                    # noqa: E402
import corruption                                          # noqa: E402

SEED = 4242
OUT = os.path.join(mc.CACHE, 'val_synth_masks.npz')


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} exists; the validation task is fixed once')
    ids = mc.split_ids('val')
    _, gt_u = mc.load_split_uint16('val', ids)
    rng = np.random.default_rng(SEED)
    dmg, mask, infos = corruption.corrupt_batch(mc.to_unit(gt_u), rng)
    mc.assert_inside_root(OUT)
    mc.ensure_dir(mc.CACHE)
    np.savez_compressed(OUT, ids=np.array(ids), damaged=dmg,
                        mask=mask.astype(np.uint8), seed=np.array([SEED]))
    fr = mask.reshape(len(ids), -1).mean(1)
    n_bg = [sum(r['background_region'] for r in i['regions']) for i in infos]
    rec = {'file': OUT, 'seed': SEED, 'n_images': len(ids),
           'recipe': corruption.recipe_record(),
           'mask_frac_mean': float(fr.mean()), 'mask_frac_min': float(fr.min()),
           'mask_frac_max': float(fr.max()),
           'n_images_with_empty_mask': int((fr == 0).sum()),
           'mean_regions_per_image': float(np.mean([i['n_regions'] for i in infos])),
           'mean_background_regions_per_image': float(np.mean(n_bg)),
           'images_with_at_least_one_background_region': int(sum(b > 0 for b in n_bg)),
           'created': mc.now(), 'git_commit': mc.git_commit()}
    mc.write_json_exclusive(os.path.join(mc.CACHE, 'val_synth_masks_provenance.json'), rec)
    print({k: v for k, v in rec.items() if k != 'recipe'})


if __name__ == '__main__':
    main()
