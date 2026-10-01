"""Models used in the paper: VGG9 (FedMA, ~3.49M params) and ResNet9 (FetchSGD, ~6.6M params)."""
import torch
import torch.nn as nn


def conv_bn(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                         nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class VGG9(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn(3, 32), conv_bn(32, 64), nn.MaxPool2d(2),
            conv_bn(64, 128), conv_bn(128, 128), nn.MaxPool2d(2), nn.Dropout(0.05),
            conv_bn(128, 256), conv_bn(256, 256), nn.MaxPool2d(2), nn.Dropout(0.1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(4096, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class Residual(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(conv_bn(c, c), conv_bn(c, c))

    def forward(self, x):
        return x + self.body(x)


class ResNet9(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.net = nn.Sequential(
            conv_bn(3, 64),
            conv_bn(64, 128), nn.MaxPool2d(2), Residual(128),
            conv_bn(128, 256), nn.MaxPool2d(2),
            conv_bn(256, 512), nn.MaxPool2d(2), Residual(512),
            nn.AdaptiveMaxPool2d(1), nn.Flatten(),
            nn.Linear(512, num_classes, bias=False),
        )

    def forward(self, x):
        return self.net(x) * 0.125


class SmallCNN(nn.Module):
    """Tiny model for quick smoke tests."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(conv_bn(3, 16), nn.MaxPool2d(2), conv_bn(16, 32), nn.MaxPool2d(2),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, num_classes))

    def forward(self, x):
        return self.net(x)


MODELS = {"vgg9": VGG9, "resnet9": ResNet9, "smallcnn": SmallCNN}


def build_model(name, num_classes):
    return MODELS[name](num_classes)


# --- flat state helpers: parameters + floating point buffers (BN running stats) ---

def state_tensors(model):
    return [t for t in model.state_dict(keep_vars=True).values() if t.dtype.is_floating_point]


@torch.no_grad()
def get_flat(model):
    return torch.cat([t.detach().reshape(-1) for t in state_tensors(model)])


@torch.no_grad()
def set_flat(model, vec):
    offset = 0
    for t in state_tensors(model):
        n = t.numel()
        t.data.copy_(vec[offset:offset + n].view_as(t))
        offset += n
