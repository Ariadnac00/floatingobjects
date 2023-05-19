import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop
from .parts import *
bands =  ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]

def get_UNet_model(input_channels, hidden_channels, n_classes, BN):
    model = UNet(input_channels=input_channels,
                 hidden_channels=hidden_channels,
                 n_classes=n_classes, BN=BN)
    return model

def get_deepUNet_model(input_channels, hidden_channels, n_classes, BN):
    model = DeepUNet(input_channels, hidden_channels, n_classes, BN)
    return model

def get_DeepUNet_with_fe(input_channels, hidden_channels, n_classes, BN, feature_extractor, bandas):
    model = DeepUNet_with_fe(input_channels, hidden_channels, n_classes, BN, feature_extractor, bandas)
    return model

def get_SUMNet_model(input_channels, hidden_channels, n_classes, BN):
    model = SUMNet(input_channels, hidden_channels, n_classes, BN)
    return model

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

class DeepUNet_with_fe(nn.Module):
    def __init__(self, input_channels, hidden_channels, n_classes, BN, feature_extractor, bandas):
        super(DeepUNet_with_fe, self).__init__()

        self.bandas = bandas
        if feature_extractor == 'conc':
            self.feature_extractor = FeatureExtractor_conc(len(bandas), 1, BN)
        else:
            self.feature_extractor = FeatureExtractor_sum(len(bandas), 1, BN)
        self.deepunet = DeepUNet(input_channels + 1, hidden_channels, n_classes, BN)


    def forward(self, x):
        indexes = [bands.index(band) for band in self.bandas]
        y = x[:, indexes, :, :]
        y = self.feature_extractor(y)
        x = torch.cat([x, y], dim=1)
        x = self.deepunet(x)
        return x


class SUMNet(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, BN):
        super(SUMNet, self).__init__()

        self.doble_conv = DobleConv(in_channels, hidden_channels, hidden_channels, BN)
        self.Encoder1 = Encoder(hidden_channels, hidden_channels * 2, hidden_channels * 2, BN)
        self.Encoder2 = Encoder(hidden_channels * 2, hidden_channels * 4, hidden_channels * 4, BN)
        self.Encoder3 = Encoder(hidden_channels * 4, hidden_channels * 8, hidden_channels * 8, BN)
        self.Encoder4 = Encoder(hidden_channels * 8, hidden_channels * 8, hidden_channels * 8, BN)

        self.Decoder1 = Decoder_residual(hidden_channels * 8, hidden_channels * 8, hidden_channels * 4, BN)
        self.Decoder2 = Decoder_residual(hidden_channels * 4, hidden_channels * 4, hidden_channels * 2, BN)
        self.Decoder3 = Decoder_residual(hidden_channels * 2, hidden_channels * 2, hidden_channels, BN)
        self.Decoder4 = Decoder_residual(hidden_channels, hidden_channels, out_channels, BN)

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

        return x


"""
from torchinfo import summary
model = get_UNet_model(padding=True, input_channels=3, hidden_channels=32, n_classes=1)
summary(model, (1, 3, 128, 128))
"""