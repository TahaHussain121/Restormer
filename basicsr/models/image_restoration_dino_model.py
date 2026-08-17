"""ImageCleanModel + explicit DINO regime switching and the E1 stability gate.

WHY A SEPARATE MODEL CLASS
  1. `RestormerDinoSpatial` needs to be told, EXPLICITLY, which regime it is in
     -- 'train128' (radar 128 -> DINO 224, train128 mean) or 'eval256'
     (radar 256 -> DINO 448, eval256 mean). The work order forbids inferring
     that from tensor size. Training always uses train128; the in-training
     validation loop always runs on FULL 256x256 images (the val dataset has no
     crop keys), so it always uses eval256. Both switches are made here, in one
     place, by name.
  2. The stability logging and the hard abort gate live here, because this is
     the only place that sees every optimizer step.

Everything else -- data feeding, loss, optimizer, validation loop, metrics --
is inherited unchanged from `ImageCleanModel`. E0 uses `ImageCleanModel`
directly, so the two arms share the identical training step.
"""

import datetime
import json
import os
import socket

import torch

from basicsr.models.image_restoration_model import ImageCleanModel
from basicsr.utils import get_root_logger


class StabilityAbort(RuntimeError):
    """Raised on an OPTIMIZATION / STABILITY FAILURE. Never caught here.

    This is deliberately fatal: the work order says a trigger is reported and
    investigated, NOT repaired by changing the architecture mid-run.
    """


class ImageCleanModelDinoSpatial(ImageCleanModel):

    def __init__(self, opt):
        super().__init__(opt)
        st = (opt.get('dino_stability') or {})
        self.stab_enabled = bool(st.get('enabled', True))
        # DIAGNOSTIC MODE. abort_on_trigger: false keeps every measurement and
        # every trigger evaluation but does NOT stop the run -- it exists so a
        # throwaway identity can answer "does injection_ratio plateau or
        # diverge?", which a fatal gate makes unobservable by construction.
        # The default is true, so the real E1 config is unaffected.
        self.stab_abort = bool(st.get('abort_on_trigger', True))
        self._stab_reported = set()
        self.stab_ratio_max = float(st.get('injection_ratio_max', 0.5))
        # WINDOW. The ratio-based rules are only meaningful once the branch has
        # finished switching on: P is zero-initialised, so injection_ratio MUST
        # rise from 0, and AdamW's first steps are ~lr per weight regardless of
        # gradient magnitude. Before this iteration the ratio is still measured,
        # logged and stored -- it simply does not trigger an abort. The NaN/Inf
        # hard-stop is NOT windowed and stays active from iteration 1.
        self.stab_ratio_start = int(st.get('ratio_rules_start_iter', 1))
        self.stab_ref_iter = int(st.get('reference_iter', 5000))
        self.stab_growth_max = float(st.get('growth_factor_max', 10.0))
        self.stab_log_freq = int(st.get('log_freq',
                                        opt['logger'].get('print_freq', 1000)))
        self.stab_devlog = st.get('devlog', None)

        exp_dir = opt['path'].get('experiments_root') or os.path.dirname(
            opt['path']['models'])
        self.stab_dir = exp_dir
        self.stab_csv = os.path.join(exp_dir, 'dino_stability.csv')
        self.stab_ref_path = os.path.join(exp_dir, 'dino_stability_reference.json')
        self.stab_ref_value = None
        if os.path.isfile(self.stab_ref_path):
            # survives a walltime kill + auto-resume: the ~5k reference must not
            # silently reset, or the >10x growth rule would never fire again.
            with open(self.stab_ref_path) as f:
                self.stab_ref_value = float(json.load(f)['injection_ratio'])

    # ------------------------------------------------------------------
    def _bare(self):
        return self.get_bare_model(self.net_g)

    # ------------------------------------------------------------------
    # training: ALWAYS the 128-crop regime and the train128 mean
    # ------------------------------------------------------------------
    def optimize_parameters(self, current_iter):
        self._bare().set_dino_mode('train128')
        super().optimize_parameters(current_iter)
        self._after_step(current_iter)

    def _after_step(self, current_iter):
        net = self._bare()
        stats = dict(net.last_dino_stats)
        if net.last_dino_mode != 'train128':
            raise StabilityAbort(
                f'iter {current_iter}: training step ran in dino mode '
                f'{net.last_dino_mode!r}, expected train128')

        self.log_dict['dino/latent_norm'] = stats['latent_norm']
        self.log_dict['dino/projected_norm'] = stats['projected_norm']
        self.log_dict['dino/injection_ratio'] = stats['injection_ratio']

        if not self.stab_enabled:
            return
        self._check_stability(current_iter, stats)

        if current_iter % self.stab_log_freq == 0:
            new = not os.path.isfile(self.stab_csv)
            with open(self.stab_csv, 'a') as f:
                if new:
                    f.write('iter,loss,lr,latent_norm,projected_norm,'
                            'injection_ratio\n')
                f.write(f"{current_iter},{self.log_dict.get('l_pix', float('nan')):.6e},"
                        f"{self.get_current_learning_rate()[0]:.6e},"
                        f"{stats['latent_norm']:.6e},{stats['projected_norm']:.6e},"
                        f"{stats['injection_ratio']:.6e}\n")

    # ------------------------------------------------------------------
    # the gate. Hard stop at ANY iteration; no soft recovery, no redesign.
    # ------------------------------------------------------------------
    def _check_stability(self, current_iter, stats):
        loss = self.log_dict.get('l_pix', float('nan'))

        def bad(x):
            return not torch.isfinite(torch.tensor(float(x))).item()

        if bad(loss):
            self._abort(current_iter, stats, f'non-finite training loss ({loss})',
                        rule='nan_loss')
        for k in ('latent_norm', 'projected_norm', 'injection_ratio'):
            if bad(stats[k]):
                self._abort(current_iter, stats, f'non-finite {k} ({stats[k]}); '
                            'this covers NaN/Inf in the DINO features, in the '
                            'projected prior and in the latent alike',
                            rule=f'nan_{k}')
        # ---- ratio-based rules: only inside the validity window --------------
        ratio_rules_active = current_iter >= self.stab_ratio_start
        if ratio_rules_active and stats['injection_ratio'] > self.stab_ratio_max:
            self._abort(current_iter, stats,
                        f'injection_ratio {stats["injection_ratio"]:.4f} > '
                        f'{self.stab_ratio_max} (abort rule 1)', rule='rule1')
        if not ratio_rules_active:
            return          # still measured and logged, just not enforced yet

        if self.stab_ref_value is None and current_iter >= self.stab_ref_iter:
            self.stab_ref_value = stats['injection_ratio']
            with open(self.stab_ref_path, 'w') as f:
                json.dump({'iter': current_iter,
                           'injection_ratio': self.stab_ref_value,
                           'recorded': datetime.datetime.now().astimezone().isoformat()},
                          f, indent=2)
            get_root_logger().info(
                f'[dino] stable injection_ratio reference at iter '
                f'{current_iter}: {self.stab_ref_value:.6e}')
        elif self.stab_ref_value is not None and self.stab_ref_value > 0:
            growth = stats['injection_ratio'] / self.stab_ref_value
            if growth > self.stab_growth_max:
                self._abort(current_iter, stats,
                            f'injection_ratio grew {growth:.1f}x vs the '
                            f'{self.stab_ref_iter}-iter reference '
                            f'{self.stab_ref_value:.4e} (abort rule 2)',
                            rule='rule2')

    def _abort(self, current_iter, stats, reason, rule='unspecified'):
        net = self._bare()
        rec = {
            'event': 'OPTIMIZATION / STABILITY FAILURE',
            'experiment': self.opt['name'],
            'iter': current_iter,
            'reason': reason,
            'loss': self.log_dict.get('l_pix'),
            'stats': stats,
            'reference': self.stab_ref_value,
            'dino_block_1indexed': getattr(net, 'dino_block_1indexed', None),
            'dino_mode': getattr(net, 'last_dino_mode', None),
            'mean_buffer': getattr(net, 'last_mean_key', None),
            'time': datetime.datetime.now().astimezone().isoformat(),
            'hostname': socket.gethostname(),
        }
        # ARCHIVE COPY, never overwritten: one file per (rule, iteration, run).
        # The un-suffixed file below is only a FLAG for the chain driver; this
        # is the record, and a later event can never destroy an earlier one.
        stamp = f'{rule}_iter{current_iter}_job{os.environ.get("SLURM_JOB_ID", "local")}'

        if not self.stab_abort:
            # DIAGNOSTIC MODE: measure and record, do not stop. Reported once
            # per rule so a 5000-iteration run does not produce 5000 lines.
            rec['event'] = 'STABILITY TRIGGER (diagnostic mode, run continues)'
            path = os.path.join(self.stab_dir, f'STABILITY_TRIGGER_{stamp}.json')
            if rule not in self._stab_reported:
                self._stab_reported.add(rule)
                with open(path, 'w') as f:
                    json.dump(rec, f, indent=2)
                get_root_logger().warning(
                    f'[dino] STABILITY TRIGGER ({rule}) at iter {current_iter}: '
                    f'{reason} -- diagnostic mode, continuing')
            return

        archive = os.path.join(self.stab_dir, f'STABILITY_FAILURE_{stamp}.json')
        with open(archive, 'w') as f:            # the record: unique, permanent
            json.dump(rec, f, indent=2)
        flag = os.path.join(self.stab_dir, 'STABILITY_FAILURE.json')
        with open(flag, 'w') as f:               # the flag the chain driver reads
            json.dump(rec, f, indent=2)
        get_root_logger().error(f'OPTIMIZATION / STABILITY FAILURE: {reason}')
        get_root_logger().error(f'archived to {archive}')

        if self.stab_devlog and os.path.isfile(self.stab_devlog):
            with open(self.stab_devlog, 'a') as f:
                f.write(f'\n\n## {rec["time"]} — OPTIMIZATION / STABILITY '
                        f'FAILURE (automatic)\n\n'
                        f'- experiment: `{rec["experiment"]}`\n'
                        f'- iteration: {current_iter}\n'
                        f'- reason: {reason}\n'
                        f'- loss: {rec["loss"]}\n'
                        f'- latent_norm: {stats["latent_norm"]:.6e}, '
                        f'projected_norm: {stats["projected_norm"]:.6e}, '
                        f'injection_ratio: {stats["injection_ratio"]:.6e}\n'
                        f'- reference injection_ratio: {self.stab_ref_value}\n'
                        f'- host: {rec["hostname"]}\n\n'
                        f'This is an OPTIMIZATION / STABILITY failure. It is NOT '
                        f'evidence that the DINO prior is useless, and it must '
                        f'not be answered by redesigning the architecture. '
                        f'Investigate first; any change needs a new experiment '
                        f'identity.\n')
        raise StabilityAbort(f'{rec["event"]} at iter {current_iter}: {reason}')

    # ------------------------------------------------------------------
    # validation: ALWAYS full 256x256 images -> the eval256 regime
    # ------------------------------------------------------------------
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img,
                           rgb2bgr, use_image):
        net = self._bare()
        previous = net.dino_mode
        net.set_dino_mode('eval256')
        try:
            return super().nondist_validation(dataloader, current_iter,
                                              tb_logger, save_img, rgb2bgr,
                                              use_image)
        finally:
            net.set_dino_mode(previous)
