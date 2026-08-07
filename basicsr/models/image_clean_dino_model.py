from basicsr.models.image_restoration_model import ImageCleanModel


class ImageCleanModelDINO(ImageCleanModel):
    """ImageCleanModel that also LOGS the FiLM modulation magnitude.

    Identical to ImageCleanModel in every training respect -- same loss, same
    optimizer, same schedule, same validation. The only addition is that after
    each step it copies RestormerDINO.last_film_stats into log_dict, so
    |gamma|max, |beta|max and the across-batch std of gamma appear in the
    training log and TensorBoard.

    This exists because of DEVLOG Step 22: the first E1 attempt trained for 73k
    iterations with |gamma| growing to 325 and nothing ever reported it. The
    training loss looked healthy the whole time. Modulation magnitude is now an
    observable, not something you only discover by loading a checkpoint.

    Use with network_g.type: RestormerDINO.
    """

    def _log_film_stats(self):
        net = self.get_bare_model(self.net_g)
        stats = getattr(net, 'last_film_stats', None)
        if stats:
            for k, v in stats.items():
                self.log_dict[k] = float(v)

    def optimize_parameters(self, current_iter):
        super().optimize_parameters(current_iter)
        self._log_film_stats()
