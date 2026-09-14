"""The fixed residual refiner: a small U-Net, specified in advance, not tuned.

    input  concat(X, Y0)                      2 channels, X first
    E1     2 -> 16 -> 16                      full resolution
    pool   2x2 average
    E2     16 -> 32 -> 32                     1/2
    pool   2x2 average
    B      32 -> 64 -> 64                     1/4
    up     bilinear to E2 size, align_corners=False; concat [up(B), E2] = 96
    D2     96 -> 32 -> 32
    up     bilinear to E1 size, align_corners=False; concat [up(D2), E1] = 48
    D1     48 -> 16 -> 16
    out    1x1, 16 -> 1                       SIGNED residual, no activation

Every block is conv3x3(bias) - ReLU - conv3x3(bias) - ReLU, stride 1, padding 1.
No normalisation, attention or extra branches. Hidden convolutions are
Kaiming-normal (fan_in, ReLU) with zero bias; the output convolution has ZERO
weight AND bias, so at initialisation delta == 0 exactly and Y == Y0.

Expected trainable parameters: 118,129 (verified in smoke_refiner.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

EXPECTED_PARAMS = 118_129


def _block(cin, cmid, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cmid, 3, 1, 1, bias=True), nn.ReLU(),
        nn.Conv2d(cmid, cout, 3, 1, 1, bias=True), nn.ReLU())


class ResidualRefinerUNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.e1 = _block(2, 16, 16)
        self.e2 = _block(16, 32, 32)
        self.bottleneck = _block(32, 64, 64)
        self.d2 = _block(96, 32, 32)
        self.d1 = _block(48, 16, 16)
        self.out = nn.Conv2d(16, 1, 1, bias=True)
        self.pool = nn.AvgPool2d(2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in',
                                        nonlinearity='relu')
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def residual(self, x, y0):
        """delta = Refiner(concat(X, Y0)). x, y0: [B,1,H,W] float32."""
        if x.shape != y0.shape or x.shape[1] != 1:
            raise ValueError(f'x {tuple(x.shape)} / y0 {tuple(y0.shape)}')
        inp = torch.cat([x, y0], dim=1)
        e1 = self.e1(inp)
        e2 = self.e2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        u2 = F.interpolate(b, size=e2.shape[-2:], mode='bilinear',
                           align_corners=False)
        d2 = self.d2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(d2, size=e1.shape[-2:], mode='bilinear',
                           align_corners=False)
        d1 = self.d1(torch.cat([u1, e1], dim=1))
        return self.out(d1)

    def forward(self, x, y0):
        """Returns (Y_final, delta). Y_final is UNBOUNDED; clipping and
        quantisation happen only at evaluation, identically for E0."""
        delta = self.residual(x, y0)
        return y0 + delta, delta


def count_params(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
