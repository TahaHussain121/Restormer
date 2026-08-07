"""Generate the 4k-iteration STABILITY GATE configs from the real arm configs.

Why this is generated rather than hand-written: the gate is only meaningful if
it runs the SAME model and data path as the real run. Deriving it from the real
yml means the two cannot drift apart -- the gate always inherits any change made
to the arm config, and only the run-length knobs are overridden here.

    python Deraining_Holo/make_gate_configs.py

Writes Deraining_Holo/Options/Holo_DINOv2_{arm}_GATE.yml for both arms.
See DEVLOG Step 22 for why this gate exists.
"""
import os

ARMS = ('lqDINO', 'renderDINO')
OPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Options')

HDR = """# =============================================================================
# GENERATED from Holo_DINOv2_{arm}_Restormer.yml -- do not hand-edit.
# Regenerate: python Deraining_Holo/make_gate_configs.py
#
# STABILITY GATE (DEVLOG Step 22). A 4000-iteration run whose ONLY purpose is to
# answer "is this run healthy?" before committing ~3 GPU-days to it. Identical to
# the full config except: total_iter 4000, val every 1000, print every 200, and
# its own experiment name so it never touches the real run's directory.
#
# PASS CRITERION: val PSNR at iter 4000 >= 18 dB.
# The Exp 2 baseline reached 19.62 dB at its first validation; the two FAILED
# unbounded-FiLM arms were at 3.22 and 7.80 dB at the same point. This gate would
# have caught that failure in ~30 minutes instead of ~20 GPU-hours.
# Also watch film_g_absmax (must stay <= film_gamma_scale) and film_g_std
# (near 0 => gamma is a constant rescale, not input-dependent guidance).
# =============================================================================
"""

# (find, replace) applied once each. Everything else is inherited verbatim.
OVERRIDES = [
    ('  total_iter: 300000', '  total_iter: 4000'),
    ('  val_freq: !!float 4e3', '  val_freq: !!float 1e3'),
    ('  print_freq: 1000', '  print_freq: 200'),
    ('  save_checkpoint_freq: !!float 2e3', '  save_checkpoint_freq: !!float 4e3'),
]


def build(arm):
    src = os.path.join(OPTS, f'Holo_DINOv2_{arm}_Restormer.yml')
    s = open(src).read()
    s = s[s.index('# general settings'):]          # drop the full run's header
    for a, b in [(f'name: Holo_DINOv2_{arm}_verynoisy_v2',
                  f'name: Holo_DINOv2_{arm}_GATE'),
                 (f'tb_logger_dir: tb_logger/Holo_DINOv2_{arm}_verynoisy_v2',
                  f'tb_logger_dir: tb_logger/Holo_DINOv2_{arm}_GATE')] + OVERRIDES:
        if a not in s:
            raise SystemExit(f'{src}: expected line not found -> {a!r}\n'
                             'The arm config changed shape; fix this script.')
        s = s.replace(a, b, 1)
    dst = os.path.join(OPTS, f'Holo_DINOv2_{arm}_GATE.yml')
    open(dst, 'w').write(HDR.format(arm=arm) + s)
    print('wrote', dst)


if __name__ == '__main__':
    for a in ARMS:
        build(a)
