"""Network architectures used by AHR and the baselines.

* MNIST: a dense encoder with two hidden layers of 400 ReLU units (as in
  van de Ven et al., 2021) and its mirror image as the decoder.
* SVHN / CIFAR-10 / CIFAR-100: a CIFAR-style ResNet-32 encoder (He et al., 2016;
  the backbone used by FACIL / Masana et al., 2022) and a decoder made of three
  transposed-convolution layers. The decoder has ~1.4M parameters for a 307-d
  latent, matching the decoder size the paper reports for CIFAR-100.

Encoders take images in ``[0, 1]``; dataset normalisation happens inside the
encoder so that decoded images (also in ``[0, 1]``) can be fed straight back.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _fp32(fn, x):
    """Run ``fn`` in float32 even inside a bf16 autocast region. Used for the maps
    into and out of the latent space, whose precision matters for memorisation."""
    with torch.autocast(x.device.type, enabled=False):
        return fn(x.float())


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


# --------------------------------------------------------------------------- #
# Dense networks (MNIST)
# --------------------------------------------------------------------------- #
class MLPBackbone(nn.Module):
    """784 -> 400 -> 400 (ReLU). ``out_dim`` = 400."""

    def __init__(self, in_shape=(1, 28, 28), hidden=400):
        super().__init__()
        n_in = in_shape[0] * in_shape[1] * in_shape[2]
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(n_in, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.out_dim = hidden

    def forward(self, x):
        return self.net(x)


class MLPDecoder(nn.Module):
    """Mirror of the dense encoder: latent -> 400 -> 400 -> 784 (sigmoid)."""

    def __init__(self, latent_dim, out_shape=(1, 28, 28), hidden=400):
        super().__init__()
        self.out_shape = out_shape
        n_out = out_shape[0] * out_shape[1] * out_shape[2]
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, n_out))

    def forward(self, z):
        return torch.sigmoid(self.net(z)).view(-1, *self.out_shape)


# --------------------------------------------------------------------------- #
# ResNet-32 (CIFAR variant, 0.46M parameters)
# --------------------------------------------------------------------------- #
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + res, inplace=True)


class ResNet32Backbone(nn.Module):
    """ResNet-32 feature extractor (3 stages x 5 basic blocks, 16/32/64 channels).

    ``pool`` controls the spatial size kept after the last stage: 1 gives the
    usual global-average-pooled 64-d feature (used by the softmax baselines),
    4 keeps a 64x4x4 map, which AHR's latent head uses so that the latent code
    retains the spatial information the decoder needs for reconstruction.
    """

    def __init__(self, in_channels=3, mean=None, std=None, pool=1, num_blocks=5):
        super().__init__()
        self.norm = Normalize(mean, std) if mean is not None else nn.Identity()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.inplanes = 16
        self.layer1 = self._make_layer(16, num_blocks, 1)
        self.layer2 = self._make_layer(32, num_blocks, 2)
        self.layer3 = self._make_layer(64, num_blocks, 2)
        self.pool = nn.AdaptiveAvgPool2d(pool)
        self.out_dim = 64 * pool * pool
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride):
        layers = [BasicBlock(self.inplanes, planes, stride)]
        self.inplanes = planes
        layers += [BasicBlock(planes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.norm(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer3(self.layer2(self.layer1(x)))
        return torch.flatten(self.pool(x), 1)


class ConvDecoder(nn.Module):
    """Three transposed-convolution layers: latent -> C x 32 x 32 (sigmoid).

    latent --Linear--> 192x4x4 --ConvT--> 128x8x8 --ConvT--> 64x16x16 --ConvT--> Cx32x32.
    ``width`` scales the channel counts (width=1 gives ~1.47M params for m=307).
    """

    def __init__(self, latent_dim, out_channels=3, out_size=32, width=1.0):
        super().__init__()
        c1, c2, c3 = int(192 * width), int(128 * width), int(64 * width)
        self.base = out_size // 8
        self.c1 = c1
        self.fc = nn.Linear(latent_dim, c1 * self.base * self.base)
        self.deconv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c1, c2, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c2, c3, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c3, out_channels, 4, 2, 1))

    def forward(self, z):
        h = _fp32(self.fc, z).view(-1, self.c1, self.base, self.base)
        return torch.sigmoid(self.deconv(h))


class SpatialHead(nn.Module):
    """1x1 convolution of the ResNet feature map to ``c`` channels, flattened: a
    spatially laid-out latent (e.g. 8x8x5 = 320 numbers for a ~307-d budget)."""

    def __init__(self, in_channels, c, init_std=1e-3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, c, 1)
        nn.init.normal_(self.conv.weight, std=init_std)
        nn.init.zeros_(self.conv.bias)

    def forward(self, f):
        return torch.flatten(self.conv(f), 1)


class SpatialConvDecoder(nn.Module):
    """Three convolutional layers from a c x 8 x 8 latent to C x 32 x 32:
    Conv3x3(c->w1) -> ConvT(w1->w2, x2) -> ConvT(w2->C, x2)."""

    def __init__(self, c, out_channels=3, base=8, w1=384, w2=192):
        super().__init__()
        self.c, self.base = c, base
        self.net = nn.Sequential(
            nn.Conv2d(c, w1, 3, 1, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(w1, w2, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(w2, out_channels, 4, 2, 1))

    def forward(self, z):
        h = _fp32(lambda t: t, z).view(-1, self.c, self.base, self.base)
        return torch.sigmoid(self.net(h))


# --------------------------------------------------------------------------- #
# Model wrappers
# --------------------------------------------------------------------------- #
class GaussianBlur(nn.Module):
    """Fixed depthwise Gaussian low-pass filter (no parameters)."""

    def __init__(self, channels, sigma):
        super().__init__()
        r = max(1, int(round(2 * sigma)))
        t = torch.arange(-r, r + 1, dtype=torch.float32)
        g = torch.exp(-t ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        k = (g[:, None] * g[None, :]).expand(channels, 1, 2 * r + 1, 2 * r + 1).clone()
        self.register_buffer("kernel", k)
        self.pad, self.channels = r, channels

    def forward(self, x):
        x = F.pad(x, (self.pad,) * 4, mode="replicate")
        return F.conv2d(x, self.kernel.to(x.dtype), groups=self.channels)


class Encoder(nn.Module):
    """Backbone followed by a linear map to the ``latent_dim``-d latent space.

    ``input_blur > 0`` puts a fixed Gaussian low-pass in front of the backbone so that
    the high-frequency detail that distinguishes real from decoded images is not
    visible to the encoder (see REPRODUCTION.md).
    """

    def __init__(self, backbone, latent_dim, head_init_std=1e-3, input_blur=0.0, channels=3):
        super().__init__()
        self.blur = GaussianBlur(channels, input_blur) if input_blur > 0 else nn.Identity()
        self.backbone = backbone
        self.head = nn.Linear(backbone.out_dim, latent_dim)
        # Start with (almost) all inputs mapped close to the origin so that the latent
        # scale is set by the CCE placement (RFA) rather than by the backbone's
        # arbitrary initial activation scale (which is ~500 for ResNet-32).
        nn.init.normal_(self.head.weight, std=head_init_std)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return _fp32(self.head, self.backbone(self.blur(x)))


class HybridAutoencoder(nn.Module):
    """HAE: encoder phi (classification in latent space) + decoder psi (replay)."""

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)


class IncrementalClassifier(nn.Module):
    """Backbone + a softmax head that grows as new classes arrive (baselines)."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.fc = None

    def add_classes(self, n_new):
        old = self.fc
        n_old = 0 if old is None else old.out_features
        dev = next(self.backbone.parameters()).device
        fc = nn.Linear(self.backbone.out_dim, n_old + n_new).to(dev)
        if old is not None:
            with torch.no_grad():
                fc.weight[:n_old] = old.weight
                fc.bias[:n_old] = old.bias
        self.fc = fc

    def features(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.fc(self.backbone(x))


DATASET_STATS = {
    "mnist": ((0.1307,), (0.3081,)),
    "svhn": ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),
}


def make_backbone(dataset, in_shape, pool=1):
    if dataset == "mnist":
        return MLPBackbone(in_shape)
    mean, std = DATASET_STATS[dataset]
    return ResNet32Backbone(in_shape[0], mean, std, pool=pool)


def make_hae(dataset, in_shape, latent_dim, decoder_width=1.0, enc_pool=4, input_blur=0.0,
             latent_kind="vector"):
    if dataset != "mnist" and latent_kind == "spatial":
        c = max(1, round(latent_dim / 64))  # 8x8 spatial latent
        backbone = make_backbone(dataset, in_shape, pool=8)
        backbone.pool = nn.Identity()
        backbone.forward = _feature_map_forward.__get__(backbone)
        enc = Encoder(backbone, c * 64, input_blur=input_blur, channels=in_shape[0])
        enc.head = SpatialHead(64, c)
        dec = SpatialConvDecoder(c, in_shape[0], in_shape[1] // 4,
                                 int(384 * decoder_width), int(192 * decoder_width))
        return HybridAutoencoder(enc, dec)
    if dataset == "mnist":
        enc = Encoder(MLPBackbone(in_shape), latent_dim, input_blur=input_blur, channels=in_shape[0])
        dec = MLPDecoder(latent_dim, in_shape)
    else:
        enc = Encoder(make_backbone(dataset, in_shape, pool=enc_pool), latent_dim,
                      input_blur=input_blur, channels=in_shape[0])
        dec = ConvDecoder(latent_dim, in_shape[0], in_shape[1], width=decoder_width)
    return HybridAutoencoder(enc, dec)


def _feature_map_forward(self, x):
    """ResNet forward that returns the last feature map (no pooling / flattening)."""
    x = self.norm(x)
    x = F.relu(self.bn1(self.conv1(x)), inplace=True)
    return self.layer3(self.layer2(self.layer1(x)))


def n_params(module):
    return sum(p.numel() for p in module.parameters())
