import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop


##### GENERAL ####
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


#### UNET ####
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


#### DEEPUNET ####
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


#### DEEPUNET FE ####
class Decoder_residual(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, BN):
        super(Decoder_residual, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.doble_conv = DobleConv(in_channels, mid_channels, out_channels, BN)

    def forward(self, x, sum_layer):
        x = self.up(x)
        sum_layer = CenterCrop(size=(x.size()[2], x.size()[3]))(sum_layer)
        x = x + sum_layer
        x = self.doble_conv(x)
        return x


class FeatureExtractor_conc(nn.Module):
    def __init__(self, in_channels, out_channels, BN):
        super(FeatureExtractor_conc, self).__init__()

        self.doble_conv = DobleConv(in_channels, in_channels * 2, in_channels * 2, BN)
        self.Encoder1 = Encoder(in_channels * 2, in_channels * 4, in_channels * 4, BN)
        self.Encoder2 = Encoder(in_channels * 4, in_channels * 8, in_channels * 8, BN)

        self.Decoder1 = Decoder(in_channels * 8, in_channels * 4, in_channels * 4, in_channels * 4, BN)
        self.Decoder2 = Decoder(in_channels * 4, in_channels * 2, in_channels * 2, in_channels * 2, BN)
        self.out = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)

    def forward(self, x):

        x1 = self.doble_conv(x)
        x2 = self.Encoder1(x1)
        x = self.Encoder2(x2)

        x = self.Decoder1(x, x2)
        x = self.Decoder2(x, x1)
        x = self.out(x)

        return x


class FeatureExtractor_sum(nn.Module):
    def __init__(self, in_channels, out_channels, BN):
        super(FeatureExtractor_sum, self).__init__()

        self.doble_conv = DobleConv(in_channels, in_channels * 2, in_channels * 2, BN)
        self.Encoder1 = Encoder(in_channels * 2, in_channels * 4, in_channels * 4, BN)
        self.Encoder2 = Encoder(in_channels * 4, in_channels * 8, in_channels * 8, BN)
        self.Encoder3 = Encoder(in_channels * 8, in_channels * 8, in_channels * 8, BN)

        self.Decoder1 = Decoder_residual(in_channels * 8, in_channels * 8, in_channels * 4, BN)
        self.Decoder2 = Decoder_residual(in_channels * 4, in_channels * 4, in_channels * 2, BN)
        self.Decoder3 = Decoder_residual(in_channels * 2, in_channels * 2, in_channels, BN)
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.doble_conv(x)
        x2 = self.Encoder1(x1)
        x3 = self.Encoder2(x2)
        x = self.Encoder3(x3)

        x = self.Decoder1(x, x3)
        x = self.Decoder2(x, x2)
        x = self.Decoder3(x, x1)
        x = self.out(x)

        return x


