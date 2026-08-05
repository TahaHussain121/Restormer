from collections import OrderedDict

import torch

from basicsr.models.image_restoration_model import ImageCleanModel


class ImageCleanModelRender(ImageCleanModel):
    """ImageCleanModel that also feeds a separate 'dino' image to net_g.

    Identical to ImageCleanModel except the batch carries a 'dino' tensor (the
    aligned render, from Dataset_PairedImage_uint16_Render) which is passed to
    net_g as dino_img so DINO looks at the render instead of the LQ input.
    Everything else -- loss, optimizer, schedule, validation padding -- is
    inherited unchanged.

    Use with network_g.type: RestormerDINO (its forward accepts dino_img).
    """

    def feed_train_data(self, data):
        super().feed_train_data(data)
        self.dino_img = data['dino'].to(self.device) if 'dino' in data else None

    def feed_data(self, data):
        super().feed_data(data)
        self.dino_img = data['dino'].to(self.device) if 'dino' in data else None

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        preds = self.net_g(self.lq, dino_img=self.dino_img)
        if not isinstance(preds, list):
            preds = [preds]
        self.output = preds[-1]

        loss_dict = OrderedDict()
        l_pix = 0.
        for pred in preds:
            l_pix += self.cri_pix(pred, self.gt)
        loss_dict['l_pix'] = l_pix

        l_pix.backward()
        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def nonpad_test(self, img=None):
        # img is the (possibly reflect-padded) LQ; the render goes unpadded --
        # DINO resizes to its own input size anyway.
        if img is None:
            img = self.lq
        dino_img = getattr(self, 'dino_img', None)
        net = getattr(self, 'net_g_ema', self.net_g)
        net.eval()
        with torch.no_grad():
            pred = net(img, dino_img=dino_img)
        if isinstance(pred, list):
            pred = pred[-1]
        self.output = pred
        if not hasattr(self, 'net_g_ema'):
            self.net_g.train()
