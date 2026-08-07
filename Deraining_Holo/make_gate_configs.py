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
# STABILITY GATE (DEVLOG Steps 22/24). A 16000-iteration run whose ONLY purpose
# is to answer "is this run healthy?" before committing ~3 GPU-days. Identical to
# the full config except: total_iter 16000, val every 2000, print every 200, and
# its own experiment name so it never touches the real run's directory.
#
# Length is set by the FiLM schedule, not by taste: warmup 5k + ramp 5k means
# FiLM is only fully on from iter 10k, so a short gate would test nothing. 16k
# gives ~6k iterations of fully-on FiLM.
#
# TWO PASS CRITERIA:
#   1. final val PSNR >= 18 dB.
#   2. final val PSNR >= (best val during warmup) - 1 dB.
# (2) is the important one and is free: during warmup film_ramp = 0, so the
# validations at 2k/4k ARE the plain baseline. The gate therefore contains its
# own control, and criterion (2) asks exactly "does switching FiLM on make the
# model worse?" -- the question both previous attempts failed.
# Also watch film_g_absmax (must stay well BELOW film_gamma_scale -- pinned at
# the rail is the Step 24 failure) and film_g_std (near 0 => gamma is a constant
# rescale, not input-dependent guidance).
# =============================================================================
"""

# (find, replace) applied once each. Everything else is inherited verbatim.
OVERRIDES = [
    ('  total_iter: 300000', '  total_iter: 16000'),
    ('  val_freq: !!float 4e3', '  val_freq: !!float 2e3'),
    ('  print_freq: 1000', '  print_freq: 200'),
    ('  save_checkpoint_freq: !!float 2e3', '  save_checkpoint_freq: !!float 8e3'),
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
