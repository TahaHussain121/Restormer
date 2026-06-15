import re, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP = '../experiments/Holo_Baseline_Restormer'
logs = sorted(glob.glob(os.path.join(EXP, 'train_*.log')))
print(f'parsing {len(logs)} log(s)')

iter_re = re.compile(r'iter:\s*([\d,]+).*?l_pix:\s*([0-9.eE+-]+)')
val_re  = re.compile(r'psnr:\s*([0-9.]+)\s*#\s*ssim:\s*([0-9.]+)')

train_iter, train_loss = [], []
val_iter, val_psnr, val_ssim = [], [], []
cur_iter = 0

for lf in logs:
    with open(lf) as f:
        for line in f:
            m = iter_re.search(line)
            if m:
                cur_iter = int(m.group(1).replace(',', ''))
                train_iter.append(cur_iter)
                train_loss.append(float(m.group(2)))
                continue
            v = val_re.search(line)
            if v:
                val_iter.append(cur_iter)
                val_psnr.append(float(v.group(1)))
                val_ssim.append(float(v.group(2)))

# dedupe val by iter (resume overlaps) keeping last
seen = {}
for it, p, s in zip(val_iter, val_psnr, val_ssim):
    seen[it] = (p, s)
val_iter = sorted(seen)
val_psnr = [seen[i][0] for i in val_iter]
val_ssim = [seen[i][1] for i in val_iter]

best_i = int(np.argmax(val_psnr))
print(f'train points: {len(train_iter)} | val points: {len(val_iter)}')
print(f'best val PSNR {val_psnr[best_i]:.4f} @ iter {val_iter[best_i]}')
print(f'final val PSNR {val_psnr[-1]:.4f} @ iter {val_iter[-1]}')

stages = [(92000,'128px'),(156000,'160px'),(204000,'192px'),(300000,'256px')]

fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

# 1) train loss (raw + smoothed)
ax = axes[0]
ax.plot(train_iter, train_loss, color='lightsteelblue', lw=0.6, label='l_pix (raw)')
if len(train_loss) > 50:
    w = 50
    sm = np.convolve(train_loss, np.ones(w)/w, mode='valid')
    ax.plot(train_iter[w-1:], sm, color='navy', lw=1.5, label=f'l_pix (MA{w})')
ax.set_title('Training loss (L1)', fontweight='bold')
ax.set_xlabel('iteration'); ax.set_ylabel('l_pix'); ax.legend(); ax.grid(alpha=0.3)

# 2) val PSNR
ax = axes[1]
ax.plot(val_iter, val_psnr, '-o', ms=3, color='darkgreen')
ax.axvline(val_iter[best_i], color='red', ls='--', lw=1)
ax.scatter([val_iter[best_i]], [val_psnr[best_i]], color='red', zorder=5,
           label=f'best {val_psnr[best_i]:.3f} @ {val_iter[best_i]//1000}k')
ax.set_title('Validation PSNR (dB)', fontweight='bold')
ax.set_xlabel('iteration'); ax.set_ylabel('PSNR (dB)'); ax.legend(); ax.grid(alpha=0.3)

# 3) val SSIM
ax = axes[2]
ax.plot(val_iter, val_ssim, '-o', ms=3, color='purple')
ax.set_title('Validation SSIM', fontweight='bold')
ax.set_xlabel('iteration'); ax.set_ylabel('SSIM'); ax.grid(alpha=0.3)

# mark progressive stage boundaries on all axes
for ax in axes:
    for x, lbl in stages[:-1]:
        ax.axvline(x, color='gray', ls=':', lw=0.8, alpha=0.6)

fig.suptitle('Holo Baseline Restormer — 300k training curves', fontweight='bold', y=1.02)
plt.tight_layout()
out = os.path.join(EXP, 'training_curves.png')
plt.savefig(out, dpi=130, bbox_inches='tight')
print('saved ->', os.path.abspath(out))
