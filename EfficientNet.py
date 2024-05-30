import torch
import torch.nn as nn
import torch.nn.functional as F

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride, kernel_size):
        super(MBConv, self).__init__()
        self.stride = stride
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (self.stride == 1 and in_channels == out_channels)

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(SiLU())

        # Depthwise convolution layer
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(SiLU())

        # Squeeze-and-excitation layer
        se_ratio = 0.25
        se_dim = max(1, int(in_channels * se_ratio))
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Conv2d(hidden_dim, se_dim, kernel_size=1))
        layers.append(SiLU())
        layers.append(nn.Conv2d(se_dim, hidden_dim, kernel_size=1))
        layers.append(nn.Sigmoid())

        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        identity = x
        x = self.block(x)
        if self.use_residual:
            x += identity
        return x

class FusedMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride, kernel_size):
        super(FusedMBConv, self).__init__()
        self.stride = stride
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (self.stride == 1 and in_channels == out_channels)

        layers = []
        if expand_ratio != 1:
            # Expansion and depthwise convolution layer
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(SiLU())
        else:
            hidden_dim = in_channels
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(SiLU())

        # Projection layer
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        identity = x
        x = self.block(x)
        if self.use_residual:
            x += identity
        return x

class EfficientNetV2S(nn.Module):
    def __init__(self, num_classes=1000):
        super(EfficientNetV2S, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            SiLU()
        )

        self.blocks = nn.Sequential(
            FusedMBConv(24, 24, expand_ratio=1, stride=1, kernel_size=3),
            FusedMBConv(24, 48, expand_ratio=4, stride=2, kernel_size=3),
            FusedMBConv(48, 64, expand_ratio=4, stride=2, kernel_size=3),
            MBConv(64, 128, expand_ratio=4, stride=2, kernel_size=3),
            MBConv(128, 160, expand_ratio=6, stride=1, kernel_size=3),
            MBConv(160, 256, expand_ratio=6, stride=2, kernel_size=3)
        )

        self.head = nn.Sequential(
            nn.Conv2d(256, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

model = EfficientNetV2S(num_classes=1000)
x = torch.randn(1, 3, 224, 224)  
output = model(x)
