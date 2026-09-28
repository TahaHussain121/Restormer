"""The fixed completion U-Net. Specified in advance, not tuned.

It is the SAME topology as the existing 118,129-parameter refiner
(dino_analysis_phases/refiner_e0/refiner_arch.py), with one change: the input
has THREE channels instead of two.

    input  concat(X, Y_in, M)                 3 channels, in that order
    E1     3 -> 16 -> 16                      full resolution
    pool   2x2 average
    E2     16 -> 32 -> 32                     1/2
    pool   2x2 average
    B      32 -> 64 -> 64                     1/4
    up     bilinear (align_corners=False) + concat E2 (96) -> D2 96 -> 32 -> 32
    up     bilinear + concat E1 (48)          -> D1 48 -> 16 -> 16
    out    1x1, 16 -> 1                       SIGNED residual, no activation

Every block conv3x3(bias)-ReLU-conv3x3(bias)-ReLU. Kaiming-normal (fan_in,
ReLU) hidden weights, zero biases, ZERO output weight and bias, so at
initialisation delta == 0 and Y_out == Y_in exactly.

The correction is applied ONLY inside the mask:

    Y_out = Y_in + M * delta

so Y_out == Y_in exactly wherever M == 0, by construction and not by a penalty.

Parameters: 118,273 trainable (the 2->3 input channel adds 16*9 = 144 weights
to the first convolution). The existing refiners have 118,129.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

EXPECTED_PARAMS = 118_273


def _block(cin, cmid, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cmid, 3, 1, 1, bias=True), nn.ReLU(),
        nn.Conv2d(cmid, cout, 3, 1, 1, bias=True), nn.ReLU())


class MaskedCompletionUNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.e1 = _block(3, 16, 16)
        self.e2 = _block(16, 32, 32)
        self.bottleneck = _block(32, 64, 64)
        self.d2 = _block(96, 32, 32)
        self.d1 = _block(48, 16, 16)
        self.out = nn.Conv2d(16, 1, 1, bias=True)
        self.pool = nn.AvgPool2d(2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def residual(self, x, y_in, m):
        """delta = Net(concat(X, Y_in, M)); all tensors [B,1,H,W] float32."""
        if not (x.shape == y_in.shape == m.shape) or x.shape[1] != 1:
            raise ValueError(f'x {tuple(x.shape)} y {tuple(y_in.shape)} m {tuple(m.shape)}')
        inp = torch.cat([x, y_in, m], dim=1)
        e1 = self.e1(inp)
        e2 = self.e2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        u2 = F.interpolate(b, size=e2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = self.d2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(d2, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        d1 = self.d1(torch.cat([u1, e1], dim=1))
        return self.out(d1)

    def forward(self, x, y_in, m):
        """Returns (Y_out, delta_masked). Y_out is UNBOUNDED; clipping and
        quantisation happen only at evaluation. Outside the mask Y_out is
        bit-identical to Y_in (multiplication by M == 0 then addition of 0)."""
        delta = self.residual(x, y_in, m) * m
        return y_in + delta, delta


def count_params(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
