# model.py
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """
    '정상' Normal Map을 복원하기 위한 U-Net 기반 AutoEncoder.
    입력/출력: (B, 3, 256, 256), 값 범위: [-1, 1]
    """
    def __init__(self, n_channels: int = 3, n_classes: int = 3):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder
        self.inc = DoubleConv(n_channels, 64)

        self.down1 = nn.ModuleList([nn.MaxPool2d(2), DoubleConv(64, 128)])
        self.down2 = nn.ModuleList([nn.MaxPool2d(2), DoubleConv(128, 256)])
        self.down3 = nn.ModuleList([nn.MaxPool2d(2), DoubleConv(256, 512)])
        self.down4 = nn.ModuleList([nn.MaxPool2d(2), DoubleConv(512, 1024)])  # Bottleneck

        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.final_activation = nn.Tanh()  # [-1, 1]

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)          # (B, 64, 256, 256)
        x2 = self.down1[0](x1)
        x2 = self.down1[1](x2)    # (B, 128, 128, 128)

        x3 = self.down2[0](x2)
        x3 = self.down2[1](x3)    # (B, 256, 64, 64)

        x4 = self.down3[0](x3)
        x4 = self.down3[1](x4)    # (B, 512, 32, 32)

        x5 = self.down4[0](x4)
        x5 = self.down4[1](x5)    # (B, 1024, 16, 16) - Bottleneck

        # Decoder + Skip
        x = self.up1(x5)          # (B, 512, 32, 32)
        x = torch.cat([x4, x], dim=1)
        x = self.conv1(x)

        x = self.up2(x)           # (B, 256, 64, 64)
        x = torch.cat([x3, x], dim=1)
        x = self.conv2(x)

        x = self.up3(x)           # (B, 128, 128, 128)
        x = torch.cat([x2, x], dim=1)
        x = self.conv3(x)

        x = self.up4(x)           # (B, 64, 256, 256)
        x = torch.cat([x1, x], dim=1)
        x = self.conv4(x)

        x = self.outc(x)          # (B, 3, 256, 256)
        logits = self.final_activation(x)  # [-1, 1]
        return logits
