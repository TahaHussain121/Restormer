"""Side-by-side comparison of two training runs.

Overlays training loss, validation PSNR and validation SSIM for two
experiments so their *dynamics* can be compared on one figure.

    python plot_compare.py \
        --exp_a Holo_Baseline_Restormer           --label_a 'Exp1 noisy (mixup on)' \
        --exp_b Holo_Baseline_Restormer_verynoisy --label_b 'Exp2 verynoisy (mixup off)' \
        --out ../experiments/compare_noisy_vs_verynoisy.png

NOTE: absolute PSNR/SSIM are only comparable when both runs share the same
dataset and the same input noise level. Pass --note to stamp a caveat on the
figure when they do not.
"""
import re, glob, os, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--exp_a', default='Holo_Baseline_Restormer')
parser.add_argument('--exp_b', default='Holo_Baseline_Restormer_verynoisy')
parser.add_argument('--label_a', default=None)
parser.add_argument('--label_b', default=None)
parser.add_argument('--out', default='../experiments/compare_runs.png')
parser.add_argument('--note', default=None, help='caveat stamped under the title')
args = parser.parse_args()

iter_re = re.compile(r'iter:\s*([\d,]+).*?l_pix:\s*([0-9.eE+-]+)')
val_re = re.compile(r'psnr:\s*([0-9.]+)\s*#\s*ssim:\s*([0-9.]+)')


def parse(exp):
    logs = sorted(glob.glob(os.path.join('../experiments', exp, 'train_*.log')))
    t_it, t_loss, val = [], [], {}
    cur = 0
    for lf in logs:
        with open(lf) as f:
            for line in f:
                m = iter_re.search(line)
                if m:
                    cur = int(m.group(1).replace(',', ''))
                    t_it.append(cur)
                    t_loss.append(float(m.group(2)))
                    continue
                v = val_re.search(line)
                if v:
                    val[cur] = (float(v.group(1)), float(v.group(2)))
    v_it = sorted(val)
    print(f'{exp}: {len(logs)} log(s), {len(t_it)} train pts, {len(v_it)} val pts')
    return (t_it, t_loss, v_it,
            [val[i][0] for i in v_it], [val[i][1] for i in v_it])


A = parse(args.exp_a)
B = parse(args.exp_b)
la = args.label_a or args.exp_a
lb = args.label_b or args.exp_b
CA, CB = 'tab:blue', 'tab:red'


def smooth(x, y, w=50):
    if len(y) <= w:
        return x, y
    return x[w - 1:], np.convolve(y, np.ones(w) / w, mode='valid')


fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

# 1) training loss (smoothed, log y — the two runs differ by ~an order of magnitude)
ax = axes[0]
for (t_it, t_loss, *_), c, l in ((A, CA, la), (B, CB, lb)):
    ax.plot(*smooth(t_it, t_loss), color=c, lw=1.6, label=l)
ax.set_yscale('log')
ax.set_title('Training loss (L1, MA50)', fontweight='bold')
ax.set_xlabel('iteration'); ax.set_ylabel('l_pix (log)')
ax.legend(); ax.grid(alpha=0.3, which='both')

# 2) validation PSNR
ax = axes[1]
for (_, _, v_it, v_p, _), c, l in ((A, CA, la), (B, CB, lb)):
    bi = int(np.argmax(v_p))
    ax.plot(v_it, v_p, '-o', ms=2.5, color=c, label=f'{l}  (best {v_p[bi]:.2f} @ {v_it[bi]//1000}k)')
    ax.scatter([v_it[bi]], [v_p[bi]], color=c, edgecolor='k', zorder=5, s=45)
ax.set_title('Validation PSNR (dB)', fontweight='bold')
ax.set_xlabel('iteration'); ax.set_ylabel('PSNR (dB)')
ax.legend(loc='center right', fontsize=9); ax.grid(alpha=0.3)

# 3) validation SSIM
ax = axes[2]
for (_, _, v_it, _, v_s), c, l in ((A, CA, la), (B, CB, lb)):
    ax.plot(v_it, v_s, '-o', ms=2.5, color=c, label=l)
ax.set_title('Validation SSIM', fontweight='bold')
ax.set_xlabel('iteration'); ax.set_ylabel('SSIM')
ax.legend(loc='lower right', fontsize=9); ax.grid(alpha=0.3)

# progressive patch-size stage boundaries (shared by both runs)
for ax in axes:
    for x in (92000, 156000, 204000):
        ax.axvline(x, color='gray', ls=':', lw=0.8, alpha=0.6)

fig.suptitle(f'{la}  vs  {lb}', fontweight='bold', y=1.04)
if args.note:
    fig.text(0.5, 0.995, args.note, ha='center', va='top',
             fontsize=9.5, style='italic', color='darkred')
plt.tight_layout()
plt.savefig(args.out, dpi=130, bbox_inches='tight')
print('saved ->', os.path.abspath(args.out))

for (_, _, v_it, v_p, v_s), l in ((A, la), (B, lb)):
    bi = int(np.argmax(v_p))
    print(f'{l}: best PSNR {v_p[bi]:.4f} @ {v_it[bi]} | '
          f'final {v_p[-1]:.4f}/{v_s[-1]:.4f} @ {v_it[-1]}')
