from basicsr.models.image_restoration_model import ImageCleanModel


class ImageCleanModelDINO(ImageCleanModel):
    """ImageCleanModel + FiLM warmup/ramp control + modulation logging.

    Identical to ImageCleanModel in every training respect -- same loss, same
    optimizer, same schedule, same validation. It adds two things, both driven
    by DEVLOG Steps 22 and 24:

    1. WARMUP/RAMP. Before each step it sets net_g.film_ramp from current_iter:
       0 while iter < film_warmup_iters, then linear 0->1 over film_ramp_iters,
       then 1. At 0 the arch skips the DINO/FiLM block entirely, so the head
       receives no gradient and stays at its zero init while the backbone
       trains. This removes the race that killed the first two attempts: with a
       random backbone, the fastest loss reduction available to a random FiLM
       head is FiLM's scale degeneracy, so it went straight there and saturated.
       The ramp value persists into validation, so a validation during warmup is
       exactly the baseline model -- a useful built-in control.

    2. LOGGING. Copies RestormerDINO.last_film_stats (film_g_absmax,
       film_b_absmax, film_g_std, film_ramp) into log_dict so the modulation is
       visible in the log and TensorBoard every print_freq. The first attempt
       trained 73k iterations with |gamma| reaching 325 and nothing reported it.

    Config (train section):
        film_warmup_iters (int): FiLM fully off for this many iters. Default 0.
        film_ramp_iters (int):   linear ramp length after warmup. Default 0.

    Use with network_g.type: RestormerDINO.
    """

    def _set_film_ramp(self, current_iter):
        net = self.get_bare_model(self.net_g)
        if not hasattr(net, 'set_film_ramp'):
            return
        t = self.opt.get('train', {}) or {}
        w = net.film_ramp_at(current_iter,
                             int(t.get('film_warmup_iters', 0) or 0),
                             int(t.get('film_ramp_iters', 0) or 0))
        net.set_film_ramp(w)

    def _log_film_stats(self):
        net = self.get_bare_model(self.net_g)
        stats = getattr(net, 'last_film_stats', None)
        if stats:
            for k, v in stats.items():
                self.log_dict[k] = float(v)

    def optimize_parameters(self, current_iter):
        self._set_film_ramp(current_iter)
        super().optimize_parameters(current_iter)
        self._log_film_stats()
