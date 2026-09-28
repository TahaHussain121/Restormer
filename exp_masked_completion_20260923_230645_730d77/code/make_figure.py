"""The qualitative panel: the EIGHT fixed cases of refiner_e0/qualitative_cases.json
(4 recovery, 2 control, 2 typical), reused without selection or cherry-picking.

Columns: noisy | target | E0 | oracle mask | FILL-RING | FILL-DIFF | completion
| correction (completion - E0). Identical display scales in every row: images
gamma 0.5 on [0,1], correction on +-0.15. Crops are the 96x96 display crops
recorded in the existing case file; the 48x48 window is outlined.
"""
import json, os, sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.patches import Rectangle                      # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mc_common as mc                                        # noqa: E402

COLS = [('noisy', 'noisy', 'img'), ('target', 'target', 'img'), ('e0', 'E0', 'img'),
        ('mask', 'oracle mask', 'mask'), ('fill_ring', 'FILL-RING', 'img'),
        ('fill_diff', 'FILL-DIFF', 'img'), ('comp', 'completion', 'img'),
        ('corr', 'completion - E0', 'diff')]
PD = os.path.join(mc.RESULTS, 'qualitative_cases')


def load(i, nm):
    return mc.load_uint16(os.path.join(PD, f'{i}_{nm}.png')).astype(np.float64) / 65535.


def main():
    with open(mc.CASES_JSON) as f:
        cases = json.load(f)['cases']
    W = 48
    fig, axes = plt.subplots(len(cases), len(COLS),
                             figsize=(2.05 * len(COLS), 2.15 * len(cases)))
    for r, c in enumerate(cases):
        i = c['id']
        y0, x0, S = c['crop_yx_size']
        wy, wx, _ = c['window_yx_size']
        imgs = {nm: load(i, nm) for nm in ('noisy', 'target', 'e0', 'mask', 'comp',
                                           'fill_ring', 'fill_diff')}
        imgs['corr'] = imgs['comp'] - imgs['e0']
        for col, (nm, title, kind) in enumerate(COLS):
            ax = axes[r, col]
            a = imgs[nm][y0:y0 + S, x0:x0 + S]
            if kind == 'img':
                ax.imshow(np.clip(a, 0, 1) ** 0.5, cmap='gray', vmin=0, vmax=1)
            elif kind == 'mask':
                ax.imshow(a > 0.5, cmap='gray', vmin=0, vmax=1)
            else:
                ax.imshow(a, cmap='bwr', vmin=-0.15, vmax=0.15)
            ax.add_patch(Rectangle((wx - x0 - .5, wy - y0 - .5), W, W, fill=False,
                                   ec='lime', lw=0.8))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{c['role']}\n{i}", fontsize=8)
    fig.suptitle('ORACLE-ASSISTED masked completion on actual E0 failures (validation, '
                 'fixed cases). The mask is derived from the ground truth and is not '
                 'available at inference.\nImages gamma 0.5 on [0,1]; correction on '
                 '+-0.15; green box = the fixed 48x48 window.', fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = mc.assert_inside_root(os.path.join(mc.RESULTS, 'figures', 'fixed_cases_oracle.png'))
    mc.ensure_dir(os.path.dirname(out))
    if os.path.exists(out):
        raise SystemExit(f'{out} exists; refusing to overwrite')
    fig.savefig(out, dpi=140)
    print('wrote', out)


if __name__ == '__main__':
    main()
