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
cur_iter = 0
for lf in logs:
    with open(lf) as f:
        for line in f:
            m = iter_re.search(line)
            if m:
                cur_iter = int(m.group(1).replace(',', ''))
                train_iter.append(cur_iter); train_loss.append(float(m.group(2)))
                continue
            v = val_re.search(line)
            if v:
                val_map[cur_iter] = (float(v.group(1)), float(v.group(2)))

val_iter = sorted(val_map)
val_psnr = np.array([val_map[i][0] for i in val_iter])
val_iter = np.array(val_iter)
train_iter = np.array(train_iter); train_loss = np.array(train_loss)

# smooth train loss (moving average)
w = 50
sm_iter, sm_loss = train_iter, train_loss
if len(train_loss) > w:
    sm_loss = np.convolve(train_loss, np.ones(w)/w, mode='valid')
    sm_iter = train_iter[w-1:]

# val "loss" proxy: MSE implied by PSNR on [0,1] data  (MSE = 10^(-PSNR/10))
val_mse = 10 ** (-val_psnr / 10.0)

best_i = int(np.argmax(val_psnr))

fig, ax1 = plt.subplots(figsize=(12, 6.5))

# left axis: training L1 loss
c1 = 'tab:blue'
ax1.plot(train_iter, train_loss, color=c1, alpha=0.18, lw=0.6)
ax1.plot(sm_iter, sm_loss, color=c1, lw=2.0, label='train L1 loss (smoothed)')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train L1 loss', color=c1)
ax1.tick_params(axis='y', labelcolor=c1)
ax1.grid(alpha=0.25)

# right axis: validation PSNR
ax2 = ax1.twinx()
c2 = 'tab:green'
ax2.plot(val_iter, val_psnr, '-o', ms=3, color=c2, label='val PSNR (dB)')
ax2.axvline(val_iter[best_i], color='red', ls='--', lw=1)
ax2.scatter([val_iter[best_i]], [val_psnr[best_i]], color='red', zorder=5)
ax2.annotate(f'best val {val_psnr[best_i]:.3f} dB @ {val_iter[best_i]//1000}k',
             xy=(val_iter[best_i], val_psnr[best_i]),
             xytext=(val_iter[best_i]-90000, val_psnr[best_i]-0.6),
             color='red', fontsize=9,
             arrowprops=dict(arrowstyle='->', color='red'))
ax2.set_ylabel('validation PSNR (dB)', color=c2)
ax2.tick_params(axis='y', labelcolor=c2)

# progressive stage boundaries
for x, lbl in [(92000,'128'),(156000,'160'),(204000,'192')]:
    ax1.axvline(x, color='gray', ls=':', lw=0.8, alpha=0.6)
    ax1.text(x, ax1.get_ylim()[1]*0.97, f'->{lbl}', fontsize=7, color='gray', ha='left')

# interpretation box
verdict = ('NO overfitting: train loss keeps falling while val PSNR\n'
           'rises then plateaus (no sustained decline). Final 33.40 dB,\n'
           'peak 33.47 dB @ 224k. Gap is the normal LR-anneal plateau.')
ax1.text(0.015, 0.04, verdict, transform=ax1.transAxes, fontsize=9,
         va='bottom', bbox=dict(boxstyle='round', fc='lightyellow', ec='gray', alpha=0.9))

lines1, lab1 = ax1.get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, lab1+lab2, loc='center right')

plt.title('Holo Baseline Restormer — train loss vs validation PSNR (overfitting check)',
          fontweight='bold')
plt.tight_layout()
out = os.path.join(EXP, 'overfit_check.png')
plt.savefig(out, dpi=130, bbox_inches='tight')
print('best val PSNR {:.4f} @ {}  | final {:.4f} @ {}'.format(
    val_psnr[best_i], val_iter[best_i], val_psnr[-1], val_iter[-1]))
print('saved ->', os.path.abspath(out))
