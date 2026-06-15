import re, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP = '../experiments/Holo_Baseline_Restormer'
logs = sorted(glob.glob(os.path.join(EXP, 'train_*.log')))

iter_re = re.compile(r'iter:\s*([\d,]+).*?l_pix:\s*([0-9.eE+-]+)')
val_re  = re.compile(r'psnr:\s*([0-9.]+)\s*#\s*ssim:\s*([0-9.]+)')

train_iter, train_loss = [], []
val_map = {}
cur = 0
for lf in logs:
    with open(lf) as f:
        for line in f:
            m = iter_re.search(line)
            if m:
                cur = int(m.group(1).replace(',', ''))
                train_iter.append(cur); train_loss.append(float(m.group(2))); continue
            v = val_re.search(line)
            if v:
                val_map[cur] = float(v.group(1))

train_iter = np.array(train_iter); train_loss = np.array(train_loss)
val_iter = np.array(sorted(val_map))
val_psnr = np.array([val_map[i] for i in val_iter])

# derived validation loss from PSNR on [0,1] data:  MSE = 10^(-PSNR/10), RMSE = sqrt(MSE)
val_rmse = np.sqrt(10 ** (-val_psnr / 10.0))

# smooth train L1
w = 50
sm_i, sm_l = train_iter, train_loss
if len(train_loss) > w:
    sm_l = np.convolve(train_loss, np.ones(w)/w, mode='valid'); sm_i = train_iter[w-1:]

best_v = int(np.argmin(val_rmse))   # lowest val loss

fig, ax = plt.subplots(figsize=(12, 6.5))
ax.plot(train_iter, train_loss, color='tab:blue', alpha=0.18, lw=0.6)
ax.plot(sm_i, sm_l, color='tab:blue', lw=2.0, label='train loss  (L1 / MAE)')
ax.plot(val_iter, val_rmse, '-o', ms=3, color='tab:red',
        label='val loss  (RMSE from PSNR)')
ax.axvline(val_iter[best_v], color='red', ls='--', lw=1)
ax.annotate(f'min val loss @ {val_iter[best_v]//1000}k\n(PSNR {val_psnr[best_v]:.3f} dB)',
            xy=(val_iter[best_v], val_rmse[best_v]),
            xytext=(val_iter[best_v]-95000, val_rmse[best_v]+0.004),
            color='red', fontsize=9, arrowprops=dict(arrowstyle='->', color='red'))

for x, lbl in [(92000,'128->160'),(156000,'160->192'),(204000,'192->256')]:
    ax.axvline(x, color='gray', ls=':', lw=0.8, alpha=0.6)
    ax.text(x, ax.get_ylim()[1]*0.96, lbl, fontsize=7, color='gray', ha='center')

ax.set_xlabel('iteration')
ax.set_ylabel('loss / error  (lower = better)')
ax.grid(alpha=0.25)
ax.legend(loc='upper right')

note = ('Overfitting would show the RED val loss bottoming out then RISING\n'
        'while BLUE train loss keeps falling. Here the val loss keeps\n'
        'decreasing / plateauing (min @ 224k) -> NO overfitting.\n'
        'Note: val loss is RMSE derived from PSNR; train loss is L1 on\n'
        'patches -> compare TRENDS, not absolute values.')
ax.text(0.015, 0.04, note, transform=ax.transAxes, fontsize=8.5, va='bottom',
        bbox=dict(boxstyle='round', fc='lightyellow', ec='gray', alpha=0.9))

plt.title('Holo Baseline Restormer — train loss vs validation loss', fontweight='bold')
plt.tight_layout()
out = os.path.join(EXP, 'loss_vs_loss.png')
plt.savefig(out, dpi=130, bbox_inches='tight')
print(f'min val loss (RMSE) {val_rmse[best_v]:.5f} @ {val_iter[best_v]} (PSNR {val_psnr[best_v]:.4f})')
print(f'final val loss (RMSE) {val_rmse[-1]:.5f} @ {val_iter[-1]} (PSNR {val_psnr[-1]:.4f})')
print('saved ->', os.path.abspath(out))
