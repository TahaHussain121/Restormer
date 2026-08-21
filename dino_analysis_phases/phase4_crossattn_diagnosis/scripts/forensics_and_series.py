"""crossattn-render diagnosis, sections 0 and 1: was it killed or did it fail,
and what did the logged attention statistics actually do.

READ-ONLY. Parses SLURM/basicsr logs and the dino_stability CSVs that already
exist. Launches nothing, loads no checkpoint, writes only under
phase4_crossattn_diagnosis/outputs/.
"""

import csv
import glob
import json
import os
import re
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE4 = os.path.dirname(_HERE)
_REPO = os.path.dirname(os.path.dirname(_PHASE4))
OUT = os.path.join(_PHASE4, 'outputs')

ARMS = [
    ('E0-Fixed', 'Holo_E0_fixed128_baseline', '#4c4c4c'),
    ('addition-render', 'Holo_E1_addition_render_fixed128_spatial_B6_latent', '#2b7bba'),
    ('concat-render', 'Holo_concat_render_fixed128_spatial_B6_latent', '#2ca02c'),
    ('E1-noisy', 'Holo_E1_addition_noisy_fixed128_spatial_B6_latent', '#dd8452'),
    ('crossattn-render', 'Holo_crossattn_render_fixed128_spatial_B6_latent', '#c44e52'),
]
XAT = 'Holo_crossattn_render_fixed128_spatial_B6_latent'
UNIFORM_ENTROPY = float(np.log(256))
UNIFORM_DIAG = 1.0 / 256

ITER_RE = re.compile(r'iter:\s*([\d,]+)')
VAL_RE = re.compile(r'Validation ValSet.*?psnr:\s*([\d.]+)\s+.*?ssim:\s*([\d.]+)')
STAT_RE = re.compile(r'dino/(\w+):\s*([\d.eE+-]+)')


def read_log(exp):
    val, stats, it = {}, {}, None
    for f in sorted(glob.glob(os.path.join(_REPO, 'experiments', exp, 'train_*.log'))):
        with open(f) as fh:
            for ln in fh:
                m = ITER_RE.search(ln)
                if m:
                    it = int(m.group(1).replace(',', ''))
                if it is not None:
                    for k, v in STAT_RE.findall(ln):
                        stats.setdefault(k, {})[it] = float(v)
                v = VAL_RE.search(ln)
                if v and it is not None:
                    val[it] = (float(v.group(1)), float(v.group(2)))
    return val, stats


def series(d):
    ks = sorted(d)
    return np.array(ks, dtype=float), np.array([d[k] for k in ks], dtype=float)


def dedup(x, y):
    """Collapse a step-function series to its distinct measurements.

    attn stats are MEASURED every dino_attn_stats_freq forwards but PRINTED
    every print_freq iterations, so the log repeats the cached value. Only the
    first iteration carrying each new value is a real measurement.
    """
    keep = [0] + [i for i in range(1, len(y)) if y[i] != y[i - 1]]
    return x[keep], y[keep]


def sacct(jobs):
    try:
        out = subprocess.check_output(
            ['sacct', '-j', ','.join(jobs), '--noheader', '-P', '--format',
             'JobID,JobName,State,ExitCode,Elapsed,Timelimit,Start,End'],
            stderr=subprocess.DEVNULL).decode()
        return [l.split('|') for l in out.strip().splitlines() if l]
    except Exception as e:                                    # noqa: BLE001
        return [[f'sacct unavailable: {e}']]


def main():
    os.makedirs(OUT, exist_ok=True)
    record = {}

    # ---------------------------------------------------------- section 0 ---
    print('=' * 92)
    print('SECTION 0 — killed, diverged, or converged badly?')
    print('=' * 92)

    chain_dir = os.path.join(_REPO, 'experiments',
                             f'Phase3_chain_state_{XAT}')
    marks = sorted(os.listdir(chain_dir))
    print(f'\nchain state {chain_dir}:')
    for m in marks:
        print(f'  {m} = {open(os.path.join(chain_dir, m)).read().strip()!r}')
    print(f'  TRAINING_DONE present: {"TRAINING_DONE" in marks}')
    print(f'  CHAIN_ABORTED present: {"CHAIN_ABORTED" in marks}  '
          f'(the crash guard never fired)')
    record['chain_state'] = {m: open(os.path.join(chain_dir, m)).read().strip()
                             for m in marks}

    print('\nSLURM accounting:')
    for row in sacct(['1784331', '1784336', '1784337']):
        print('  ' + ' | '.join(row))
    record['sacct'] = sacct(['1784331', '1784336', '1784337'])

    chain_log = glob.glob(os.path.join(
        _REPO, 'dino_analysis_phases', 'phase3_restoration', 'results', XAT,
        'logs', 'chain_*.out'))
    for p in chain_log:
        txt = open(p).read()
        for ln in txt.splitlines():
            if 'CANCELLED' in ln or 'Elapsed runtime' in ln or \
                    'Requested resources' in ln or 'queued successor' in ln:
                print('  ' + ln.strip())
                record.setdefault('chain_log_lines', []).append(ln.strip())

    curves = {}
    for label, exp, color in ARMS:
        v, s = read_log(exp)
        curves[label] = (v, s)
    xv = curves['crossattn-render'][0]
    ks = sorted(xv)
    best = max(ks, key=lambda k: xv[k][0])
    print(f'\nvalidation (8-bit training-time path), crossattn-render: '
          f'{len(ks)} points, {ks[0]:,}..{ks[-1]:,}')
    print(f'  best        {xv[best][0]:.4f} dB at iteration {best:,}')
    print(f'  last        {xv[ks[-1]][0]:.4f} dB at iteration {ks[-1]:,}')
    late = [k for k in ks if k >= 44000]
    ly = np.array([xv[k][0] for k in late])
    lx = np.array(late, dtype=float)
    slope = np.polyfit(lx, ly, 1)[0] * 100000          # dB per 100k iterations
    print(f'  plateau 44k..{ks[-1] // 1000}k: mean {ly.mean():.4f} dB, '
          f'std {ly.std(ddof=1):.4f}, n={len(ly)}')
    print(f'  linear trend over that plateau: {slope:+.4f} dB per 100k iterations')
    record['val'] = {'best_iter': best, 'best': xv[best][0],
                     'last_iter': ks[-1], 'last': xv[ks[-1]][0],
                     'plateau_mean': float(ly.mean()),
                     'plateau_std': float(ly.std(ddof=1)),
                     'plateau_slope_db_per_100k': float(slope)}

    print('\nsame-iteration comparison against the arms that ran to 300k:')
    hdr = f'{"iter":>8}' + ''.join(f'{l:>18}' for l, _, _ in ARMS)
    print(hdr)
    for it in (4000, 20000, 44000, 92000, 120000, 176000):
        row = f'{it:>8}'
        for label, _, _ in ARMS:
            v = curves[label][0].get(it)
            row += f'{v[0]:>18.3f}' if v else f'{"--":>18}'
        print(row)

    # ---------------------------------------------------------- section 1 ---
    print('\n' + '=' * 92)
    print('SECTION 1 — the logged attention statistics')
    print('=' * 92)
    st = curves['crossattn-render'][1]
    ex, ey = dedup(*series(st['attn_entropy']))
    dx, dy = dedup(*series(st['attn_diag_mass']))
    print(f'\nreference: uniform entropy = ln(256) = {UNIFORM_ENTROPY:.4f} nats, '
          f'uniform diag_mass = 1/256 = {UNIFORM_DIAG:.5f}')
    print(f'measured every dino_attn_stats_freq=5000 forwards, printed every '
          f'1000 -> {len(ey)} distinct entropy measurements in '
          f'{len(st["attn_entropy"])} log lines')
    print(f'\n{"iter":>8}{"entropy":>12}{"diag_mass":>12}')
    for i in range(0, len(ex), max(1, len(ex) // 14)):
        print(f'{int(ex[i]):>8}{ey[i]:>12.4f}'
              f'{dy[min(i, len(dy) - 1)]:>12.4f}')
    print(f'{int(ex[-1]):>8}{ey[-1]:>12.4f}{dy[-1]:>12.4f}   <- last')
    print(f'\n  entropy   {ey[0]:.4f} -> {ey[-1]:.4f}  '
          f'({100 * (1 - ey[-1] / ey[0]):.1f}% of the starting value removed); '
          f'as a fraction of uniform: {ey[0] / UNIFORM_ENTROPY:.3f} -> '
          f'{ey[-1] / UNIFORM_ENTROPY:.3f}')
    print(f'  diag_mass {dy[0]:.5f} -> {dy[-1]:.5f}  '
          f'(uniform would be {UNIFORM_DIAG:.5f}, '
          f'fully diagonal 1.0); max over the run {dy.max():.4f}')
    print(f'  effective support 1/exp-entropy: a row of entropy {ey[-1]:.3f} '
          f'nats spreads over about exp({ey[-1]:.3f}) = {np.exp(ey[-1]):.2f} '
          f'of the 256 keys')
    record['attn'] = {'entropy_first': float(ey[0]), 'entropy_last': float(ey[-1]),
                      'diag_first': float(dy[0]), 'diag_last': float(dy[-1]),
                      'diag_max': float(dy.max()),
                      'n_measurements': int(len(ey))}

    print('\ninjection_ratio, all three fusion arms (dino_stability.csv):')
    ratios = {}
    for label, exp, _ in ARMS:
        p = os.path.join(_REPO, 'experiments', exp, 'dino_stability.csv')
        if not os.path.isfile(p):
            continue
        rows = list(csv.DictReader(open(p)))
        it = np.array([float(r['iter']) for r in rows])
        r = np.array([float(r['injection_ratio']) for r in rows])
        ln = np.array([float(r_['latent_norm']) for r_ in rows])
        pn = np.array([float(r_['projected_norm']) for r_ in rows])
        ratios[label] = (it, r, ln, pn)
        print(f'  {label:<18} n={len(r):>4}  ratio first {r[0]:.3f} '
              f'last {r[-1]:.3f} min {r.min():.3f} max {r.max():.3f}  |  '
              f'latent_norm {ln[0]:.0f} -> {ln[-1]:.0f}  '
              f'projected_norm {pn[0]:.0f} -> {pn[-1]:.0f}')
        record.setdefault('ratios', {})[label] = {
            'first': float(r[0]), 'last': float(r[-1]),
            'min': float(r.min()), 'max': float(r.max()),
            'latent_first': float(ln[0]), 'latent_last': float(ln[-1]),
            'projected_first': float(pn[0]), 'projected_last': float(pn[-1])}

    # ------------------------------------------------------------- figure ---
    fig, axes = plt.subplots(2, 2, figsize=(17, 11))

    ax = axes[0, 0]
    for label, _, color in ARMS:
        v = curves[label][0]
        if not v:
            continue
        k = np.array(sorted(v), dtype=float)
        y = np.array([v[int(i)][0] for i in k])
        ax.plot(k / 1000, y, color=color, lw=1.9, label=label,
                alpha=1.0 if label == 'crossattn-render' else 0.75)
    ax.axvline(176, color='#c44e52', ls='--', lw=1.6)
    ax.text(96, 17.0, 'scancel at 176k / 08:45:04\n(16 min before walltime,\n'
            'after 132k flat iterations)',
            color='#c44e52', fontsize=10.5, fontweight='bold')
    ax.plot([best / 1000], [xv[best][0]], 'o', ms=11, mfc='none',
            mec='#c44e52', mew=2.5)
    ax.set_xlabel('iteration (thousands)')
    ax.set_ylabel('validation PSNR (dB, 8-bit training-time path)')
    ax.set_title('Every other arm is still climbing at 176k.\n'
                 f'crossattn peaked at {best:,} and is flat at '
                 f'{ly.mean():.2f} dB for its last {len(ly)} val points',
                 fontsize=12.5)
    ax.legend(fontsize=10.5, loc='center right')
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(lx / 1000, ly, 'o-', color='#c44e52', lw=1.6, ms=5)
    fit = np.polyfit(lx, ly, 1)
    ax.plot(lx / 1000, np.polyval(fit, lx), '--', color='black', lw=2,
            label=f'trend {slope:+.3f} dB / 100k iterations')
    ax.fill_between(lx / 1000, ly.mean() - ly.std(ddof=1),
                    ly.mean() + ly.std(ddof=1), color='#c44e52', alpha=0.15,
                    label=f'mean {ly.mean():.2f} ± {ly.std(ddof=1):.2f} dB')
    ax.set_xlabel('iteration (thousands)')
    ax.set_ylabel('validation PSNR (dB)')
    ax.set_title('The plateau, on its own axis: 33 val points over 132k '
                 'iterations,\nno trend — it was not still improving when '
                 'it was stopped', fontsize=12.5)
    ax.legend(fontsize=10.5)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(ex / 1000, ey, 'o-', color='#c44e52', lw=2, ms=4,
            label='attn_entropy (measured)')
    ax.axhline(UNIFORM_ENTROPY, color='#888', ls='--', lw=1.7,
               label=f'uniform over 256 keys = {UNIFORM_ENTROPY:.3f}')
    ax.axhline(0, color='#333', ls=':', lw=1.5, label='one-hot = 0')
    ax.set_ylim(-0.3, 6)
    ax.set_xlabel('iteration (thousands)')
    ax.set_ylabel('row entropy (nats)', color='#c44e52')
    ax2 = ax.twinx()
    ax2.plot(dx / 1000, dy, 'o-', color='#2b7bba', lw=2, ms=4,
             label='attn_diag_mass (measured)')
    ax2.axhline(UNIFORM_DIAG, color='#2b7bba', ls='--', lw=1.5,
                label=f'uniform = {UNIFORM_DIAG:.4f}')
    ax2.axhline(1.0, color='#2b7bba', ls=':', lw=1.5,
                label='same-position routing = 1.0')
    ax2.set_ylim(-0.06, 1.15)
    ax2.set_ylabel('diagonal mass', color='#2b7bba')
    ax.set_title('Neither collapse-to-uniform nor collapse-to-diagonal:\n'
                 f'rows became near one-hot ({ey[-1]:.3f} nats) but only '
                 f'{dy[-1]:.2f} of the mass sits on the query\'s own position',
                 fontsize=12.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9.5, loc='upper right')
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for label in ('addition-render', 'concat-render', 'E1-noisy',
                  'crossattn-render'):
        if label not in ratios:
            continue
        it, r, _, _ = ratios[label]
        color = dict((l, c) for l, _, c in ARMS)[label]
        ax.plot(it / 1000, r, color=color, lw=1.9, label=label)
    ax.axhline(10.0, color='black', ls='--', lw=1.7,
               label='gate rule 1 cap = 10.0')
    ax.set_yscale('log')
    ax.set_xlabel('iteration (thousands)')
    ax.set_ylabel('injection_ratio  = ||injected|| / ||F||  (log)')
    ax.set_title('injection_ratio never distinguished the failing arm:\n'
                 'crossattn sits between addition and the pooled arm '
                 'the whole way', fontsize=12.5)
    ax.legend(fontsize=10.5)
    ax.grid(alpha=0.3)

    fig.suptitle('crossattn-render diagnosis — sections 0 and 1',
                 fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    p = os.path.join(OUT, 'S0_S1_curves_and_attention_stats.png')
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print('\nwrote', p)

    with open(os.path.join(OUT, 'forensics.json'), 'w') as f:
        json.dump(record, f, indent=2)
    print('wrote', os.path.join(OUT, 'forensics.json'))


if __name__ == '__main__':
    main()
