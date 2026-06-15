import re, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP = '../experiments/Holo_Baseline_Restormer'
logs = sorted(glob.glob(os.path.join(EXP, 'train_*.log')))

iter_re = re.compile(r'epoch:\s*(\d+),\s*iter:\s*([\d,]+)')
val_re  = re.compile(r'psnr:\s*([0-9.]+)\s*#\s*ssim:\s*([0-9.]+)')

rows = {}  # iter -> (epoch, psnr, ssim)
ep, it = 0, 0
for lf in logs:
    with open(lf) as f:
        for line in f:
            m = iter_re.search(line)
            if m:
                ep = int(m.group(1)); it = int(m.group(2).replace(',', '')); continue
            v = val_re.search(line)
            if v:
                rows[it] = (ep, float(v.group(1)), float(v.group(2)))

iters = sorted(rows)
data = [(i, rows[i][0], rows[i][1], rows[i][2]) for i in iters]   # iter, epoch, psnr, ssim
psnrs = [d[2] for d in data]
best_idx = int(np.argmax(psnrs))
print(f'{len(data)} validations | best PSNR {psnrs[best_idx]:.4f} @ iter {data[best_idx][0]}')

# split into N column-blocks for a compact, readable layout
n = len(data)
nblocks = 3
per = -(-n // nblocks)  # ceil
headers = ['epoch', 'iter', 'PSNR', 'SSIM']

fig, axes = plt.subplots(1, nblocks, figsize=(4.0 * nblocks, 0.34 * per + 1.2))
if nblocks == 1:
    axes = [axes]

gi = 0  # global index to track best highlighting
for b, ax in enumerate(axes):
    ax.axis('off')
    chunk = data[b*per:(b+1)*per]
    cells = [[f'{e}', f'{i//1000}k' if i % 1000 == 0 else str(i), f'{p:.4f}', f'{s:.4f}']
             for (i, e, p, s) in chunk]
    if not cells:
        continue
    tbl = ax.table(cellText=cells, colLabels=headers, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 1.25)
    # style header
    for c in range(len(headers)):
        tbl[0, c].set_facecolor('#34495e'); tbl[0, c].get_text().set_color('white')
        tbl[0, c].get_text().set_fontweight('bold')
    # zebra + best-row highlight
    for r, (i, e, p, s) in enumerate(chunk, start=1):
        gidx = b*per + (r-1)
        for c in range(len(headers)):
            if gidx == best_idx:
                tbl[r, c].set_facecolor('#ffe08a')   # best row
            elif r % 2 == 0:
                tbl[r, c].set_facecolor('#f2f4f5')

fig.suptitle('Holo Baseline Restormer — validation PSNR / SSIM per checkpoint '
             f'(best: {psnrs[best_idx]:.4f} dB @ {data[best_idx][0]//1000}k, highlighted)',
             fontweight='bold', y=0.99, fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(EXP, 'val_metrics_table.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print('saved ->', os.path.abspath(out))
