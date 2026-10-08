"""Networks: U-Net (measured images -> absorption/phase) and constant-input MLPs for the
nuisance parameters (Zernike pupil coefficients, per-image shifts, per-image intensity factors)."""
import torch
import torch.nn as nn


def _block(ci, co, bn=True):
    layers = [nn.Conv2d(ci, co, 3, padding=1, padding_mode="reflect")]
    if bn: layers.append(nn.BatchNorm2d(co))
    layers += [nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(co, co, 3, padding=1, padding_mode="reflect")]
    if bn: layers.append(nn.BatchNorm2d(co))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class UNet(nn.Module):
    """Encoder-decoder with skip connections. Output: 2 channels (raw absorption, raw phase)."""

    def __init__(self, cin, base=16, depth=4, bn=True):
        super().__init__()
        ch = [base * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList([_block(cin if i == 0 else ch[i - 1], ch[i], bn) for i in range(depth)])
        self.mid = _block(ch[depth - 1], ch[depth], bn)
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([_block(2 * ch[i], ch[i], bn) for i in reversed(range(depth))])
        self.head = nn.Conv2d(ch[0], 2, 1)
        nn.init.normal_(self.head.weight, std=1e-3); nn.init.zeros_(self.head.bias)

    def forward(self, x):
        skips = []
        for e in self.enc:
            x = e(x); skips.append(x); x = nn.functional.max_pool2d(x, 2)
        x = self.mid(x)
        for u, d, s in zip(self.up, self.dec, reversed(skips)):
            x = d(torch.cat([u(x), s], 1))
        return self.head(x)


class ParamMLP(nn.Module):
    """MLP with constant input '1' (as in the slide). out = init + scale * f(1); the last layer is
    zero-initialised so optimisation starts exactly at `init`."""

    def __init__(self, n_out, init=None, scale=1.0, hidden=64, depth=2):
        super().__init__()
        layers, d = [], 1
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.Tanh()]; d = hidden
        last = nn.Linear(d, n_out); nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
        self.f = nn.Sequential(*layers, last)
        self.register_buffer("init", torch.zeros(n_out) if init is None else torch.as_tensor(init, dtype=torch.float32))
        self.scale = scale

    def forward(self):
        return self.init + self.scale * self.f(torch.ones(1, 1))[0]
