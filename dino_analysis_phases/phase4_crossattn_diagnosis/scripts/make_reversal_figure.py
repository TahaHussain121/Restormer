"""The headline of Task B in one picture: the SAME weights are +2.00 dB over E0
at the training scale and -7.56 dB below it at full resolution.

READ-ONLY. Numbers are the ones printed by inference_interventions.py (crop128)
and by run_full256_lastckpt.sh via the project's own masked_metrics.py
(full256). Nothing is recomputed here.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'outputs')
# val split, n=339, uint16 evaluation path
CROP = {'E0-Fixed 268k': 19.816, 'crossattn 4k': 18.284, 'crossattn 178k': 21.812}
FULL = {'E0-Fixed 268k': 22.077, 'crossattn 4k': 18.563, 'crossattn 178k': 14.517}

fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
names = list(CROP)
colors = ['#4c4c4c', '#888888', '#c44e52']

ax = axes[0]
x = np.arange(2)
w = 0.26
for i, n in enumerate(names):
    v = [CROP[n], FULL[n]]
    b = ax.bar(x + (i - 1) * w, v, w, label=n, color=colors[i])
    for bb, vv in zip(b, v):
        ax.text(bb.get_x() + bb.get_width() / 2, vv + 0.15, f'{vv:.2f}',
                ha='center', fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['crop128 / val\n(the TRAINING scale)',
                    'full256 / val\n(the EVALUATION scale)'])
ax.set_ylabel('val PSNR (dB), uint16 path')
ax.set_ylim(0, 25.5)
ax.legend(fontsize=11, loc='lower left')
ax.grid(axis='y', alpha=0.3)
ax.set_title('Same weights, two protocols', fontsize=13)

ax = axes[1]
for i, n in enumerate(names):
    ax.plot([0, 1], [CROP[n] - CROP['E0-Fixed 268k'],
                     FULL[n] - FULL['E0-Fixed 268k']], 'o-',
            color=colors[i], lw=2.6, ms=11, label=n)
ax.axhline(0, color='black', lw=1.8)
ax.text(0.02, 0.35, 'better than E0', fontsize=10.5, color='#1a7f37')
ax.text(0.02, -0.9, 'worse than E0', fontsize=10.5, color='#c0392b')
ax.annotate('+1.996', xy=(0, 1.996), xytext=(0.08, 2.6), fontsize=12,
            fontweight='bold', color='#c44e52')
ax.annotate('-7.560', xy=(1, -7.56), xytext=(0.78, -6.6), fontsize=12,
            fontweight='bold', color='#c44e52')
ax.set_xticks([0, 1])
ax.set_xticklabels(['crop128 (128 px, 256 tokens)',
                    'full256 (256 px, 1024 tokens)'])
ax.set_ylabel('delta vs E0-Fixed (dB)')
ax.set_ylim(-9, 4)
ax.legend(fontsize=11, loc='lower left')
ax.grid(alpha=0.3)
ax.set_title("crossattn-render's final checkpoint reverses sign with scale;\n"
             'its 4k checkpoint does not', fontsize=13)

fig.suptitle('crossattn-render: the published failure is a checkpoint-selection '
             'and scale-transfer result, not a training failure\n'
             'selection used the training-time full-256 validation — the one '
             'regime this arm cannot do — and chose iteration 4,000',
             fontsize=13.5, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.88])
p = os.path.join(OUT, 'B_scale_reversal.png')
fig.savefig(p, dpi=130)
print('wrote', p)
