"""Hybrid autoencoder: ResNet-18 (VAE) encoder f() and a four-layer CNN decoder g()."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = {"cifar100": (0.5071, 0.4865, 0.4409), "tinyimagenet": (0.4802, 0.4481, 0.3975)}
STD = {"cifar100": (0.2673, 0.2564, 0.2762), "tinyimagenet": (0.2770, 0.2691, 0.2821)}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                          nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18Encoder(nn.Module):
    """ResNet-18 with a 3x3 stem (as in MFCL for 32x32 / 64x64 inputs) and Gaussian heads.

    Takes images in [0, 1]; returns (mu, logvar) of q(z|x). Classification uses mu.
    """

    def __init__(self, latent_dim: int, dataset: str, width: int = 64):
        super().__init__()
        self.register_buffer("mean", torch.tensor(MEAN[dataset]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD[dataset]).view(1, 3, 1, 1))
        self.conv1 = nn.Conv2d(3, width, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        layers, cin = [], width
        for i, (mult, stride) in enumerate([(1, 1), (2, 2), (4, 2), (8, 2)]):
            cout = width * mult
            layers.append(nn.Sequential(BasicBlock(cin, cout, stride), BasicBlock(cout, cout, 1)))
            cin = cout
        self.layers = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(cin, latent_dim)
        self.fc_logvar = nn.Linear(cin, latent_dim)

    def forward(self, x):
        x = (x - self.mean) / self.std
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.layers(h)
        h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class ConvDecoder(nn.Module):
    """Four-layer decoder: FC -> [up, conv] x (n_up - 1) -> up, conv -> sigmoid.

    For 32x32 outputs with latent 307 and 64 base channels this has ~1.41M parameters,
    matching the 1.4M decoder reported in Table 3 of the paper.
    """

    def __init__(self, latent_dim: int, image_size: int, ch: int = 64):
        super().__init__()
        n_up = int(round(math.log2(image_size / 8)))
        assert n_up in (2, 3) and 8 * 2 ** n_up == image_size
        self.ch = ch
        self.fc = nn.Linear(latent_dim, ch * 8 * 8)
        self.bn0 = nn.BatchNorm2d(ch)
        up = lambda: nn.Upsample(scale_factor=2, mode="nearest")  # noqa: E731
        blocks = [up(), nn.Conv2d(ch, 2 * ch, 3, 1, 1), nn.BatchNorm2d(2 * ch),
                  nn.LeakyReLU(0.2, inplace=True),
                  up(), nn.Conv2d(2 * ch, ch, 3, 1, 1), nn.BatchNorm2d(ch),
                  nn.LeakyReLU(0.2, inplace=True)]
        if n_up == 3:  # 64x64 outputs need a third doubling before the output conv
            blocks.append(up())
        blocks += [nn.Conv2d(ch, 3, 3, 1, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*blocks)

    def forward(self, z):
        h = self.fc(z).view(-1, self.ch, 8, 8)
        return self.net(self.bn0(h))


class HybridAutoencoder(nn.Module):
    def __init__(self, dataset: str, image_size: int, latent_dim: int, dec_ch: int = 64):
        super().__init__()
        self.encoder = ResNet18Encoder(latent_dim, dataset)
        self.decoder = ConvDecoder(latent_dim, image_size, dec_ch)
        self.latent_dim = latent_dim

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
