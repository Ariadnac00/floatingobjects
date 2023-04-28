import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import CenterCrop


def get_UNet_model(input_channels, hidden_channels, n_classes, BN):
    model = UNet(input_channels=input_channels,
                 hidden_channels=hidden_channels,
                 n_classes=n_classes, BN=BN)
    return model


def get_deepUNet_model(input_channels, hidden_channels, n_classes, BN):
    model = DeepUNet(input_channels, hidden_channels, n_classes, BN)
    return model

class DobleConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, BN=False):
        super(DobleConv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.BN = BN
        if self.BN:
            self.bn1 = nn.BatchNorm2d(mid_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        if self.BN:
            x = self.ReLU(self.bn1(self.conv1(x)))
            x = self.ReLU(self.bn2(self.conv2(x)))
        else:
            x = self.ReLU(self.conv1(x))
            x = self.ReLU(self.conv2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, BN):
        super(Encoder, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.doble_conv = DobleConv(in_channels, mid_channels, out_channels, BN)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.doble_conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, mid_channels1, mid_channels2, out_channels, BN):
        super(Decoder, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, mid_channels1, kernel_size=2, stride=2)
        self.doble_conv = DobleConv(mid_channels1 * 2, mid_channels2, out_channels, BN)

    def forward(self, x, conc_layer):
        x = self.up(x)
        conc_layer = CenterCrop(size=(x.size()[2], x.size()[3]))(conc_layer)
        x = torch.cat([x, conc_layer], dim=1)
        x = self.doble_conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, input_channels, hidden_channels, n_classes, BN):
        super(UNet, self).__init__()
        self.doble_conv = DobleConv(input_channels, hidden_channels, hidden_channels, BN)

        self.Encoder1 = Encoder(hidden_channels, hidden_channels * 2, hidden_channels * 2, BN)
        self.Encoder2 = Encoder(hidden_channels * 2, hidden_channels * 4, hidden_channels * 4, BN)
        self.Encoder3 = Encoder(hidden_channels * 4, hidden_channels * 8, hidden_channels * 8, BN)
        self.Encoder4 = Encoder(hidden_channels * 8, hidden_channels * 16, hidden_channels * 16, BN)

        self.Decoder1 = Decoder(hidden_channels * 16, hidden_channels * 8, hidden_channels * 8, hidden_channels * 8, BN)
        self.Decoder2 = Decoder(hidden_channels * 8, hidden_channels * 4, hidden_channels * 4, hidden_channels * 4, BN)
        self.Decoder3 = Decoder(hidden_channels * 4, hidden_channels * 2, hidden_channels * 2, hidden_channels * 2, BN)
        self.Decoder4 = Decoder(hidden_channels * 2, hidden_channels, hidden_channels, hidden_channels, BN)
        self.out = nn.Conv2d(hidden_channels, n_classes, kernel_size=1)
        self.ReLU = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.doble_conv(x)
        x2 = self.Encoder1(x1)
        x3 = self.Encoder2(x2)
        x4 = self.Encoder3(x3)
        x = self.Encoder4(x4)

        x = self.Decoder1(x, x4)
        x = self.Decoder2(x, x3)
        x = self.Decoder3(x, x2)
        x = self.Decoder4(x, x1)
        x = self.out(x)

        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.maxpool(x)


class Classifier(nn.Module):
    def __init__(self, n_classes):
        super(Classifier, self).__init__()

        self.conv1 = nn.Conv2d(512, 4096, 7)
        self.conv2 = nn.Conv2d(4096, 4096, 1)
        self.conv3 = nn.Conv2d(4096, n_classes, 1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d()

    def forward(self, x):
        x = self.dropout(self.relu(self.conv1(x)))
        x = self.dropout(self.relu(self.conv2(x)))
        x = self.conv3(x)

        return x



class DownBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, BN):
        super(DownBlock, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.doble_conv = DobleConv(in_channels, mid_channels, out_channels, BN)

    def forward(self, x):
        x = self.maxpool(x)
        return self.doble_conv(x) + x


class UpBlock(nn.Module):
    def __init__(self, in_channels, mid_channels1, mid_channels2, out_channels, BN):
        super(UpBlock, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, mid_channels1, kernel_size=2, stride=2)
        self.doble_conv = DobleConv(mid_channels1 * 2, mid_channels2, out_channels, BN)

    def forward(self, x, conc_layer):
        x1 = self.up(x)
        conc_layer = CenterCrop(size=(x1.size()[2], x1.size()[3]))(conc_layer)
        x = torch.cat([x1, conc_layer], dim=1)
        return self.doble_conv(x) + x1


class DeepUNet(nn.Module):
    def __init__(self, input_channels, hidden_channels, n_classes, BN):
        super(DeepUNet, self).__init__()

        self.doble_conv = DobleConv(input_channels, 2 * hidden_channels, hidden_channels, BN)
        self.down_block1 = DownBlock(hidden_channels, 2 * hidden_channels, hidden_channels, BN)
        self.down_block2 = DownBlock(hidden_channels, 2 * hidden_channels, hidden_channels, BN)
        self.down_block3 = DownBlock(hidden_channels, 2 * hidden_channels, hidden_channels, BN)
        self.down_block4 = DownBlock(hidden_channels, 2 * hidden_channels, hidden_channels, BN)

        self.up_block1 = UpBlock(hidden_channels, hidden_channels, 2 * hidden_channels, hidden_channels, BN)
        self.up_block2 = UpBlock(hidden_channels, hidden_channels, 2 * hidden_channels, hidden_channels, BN)
        self.up_block3 = UpBlock(hidden_channels, hidden_channels, 2 * hidden_channels, hidden_channels, BN)
        self.up_block4 = UpBlock(hidden_channels, hidden_channels, 2 * hidden_channels, hidden_channels, BN)
        self.out = nn.Conv2d(hidden_channels, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.doble_conv(x)
        x2 = self.down_block1(x1)
        x3 = self.down_block2(x2)
        x4 = self.down_block3(x3)
        y = self.down_block4(x4)

        y = self.up_block1(y, x4)
        y = self.up_block2(y, x3)
        y = self.up_block3(y, x2)
        y = self.up_block4(y, x1)

        x, x1, x2, x3, x4 = None, None, None, None, None

        return self.out(y)


"""
from torchinfo import summary
model = get_UNet_model(padding=True, input_channels=3, hidden_channels=32, n_classes=1)
summary(model, (1, 3, 128, 128))
"""
"""
from torchinfo import summary
model = get_FCN32s_model(3, 1)
summary(model, (1, 3, 256, 256))
"""
