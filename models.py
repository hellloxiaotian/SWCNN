import torch
import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                                bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding,
                                bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out


class DnCNN_RL(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN_RL, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                                bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding,
                                bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return x + out


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
        layers.append(nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        layers.append(nn.ReLU(inplace=True))
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        return x


class HN(nn.Module):

    def __init__(self, in_channels=3, out_channels=3):
        """Initializes U-Net."""

        super(HN, self).__init__()

        # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv6, upsample5
        self._block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block4 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block5 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block6 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):

        # Encoder
        pool1 = self._block1(x)  # 2
        pool2 = self._block2(pool1)  # 1
        pool3 = self._block2(pool2)  # 1
        pool4 = self._block2(pool3)  # 1
        pool5 = self._block2(pool4)  # 1

        # Decoder
        upsample5 = self._block3(pool5)  # 2
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self._block4(concat5)  # 3
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self._block5(concat4)  # 3
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block5(concat3)  # 3
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block5(concat2)  # 3
        concat1 = torch.cat((upsample1, x), dim=1)

        # Final activation
        return self._block6(concat1)  # 3


from batchrenorm import BatchRenorm2d


class UpNet(nn.Module):

    def __init__(self):
        super(UpNet, self).__init__()
        layers = [nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
                  BatchRenorm2d(64),
                  nn.ReLU()]

        for i in range(15):
            layers.append(nn.Conv2d(64, 64, 3, 1, 1))
            layers.append(BatchRenorm2d(64))
            layers.append(nn.ReLU())

        layers.append(nn.Conv2d(64, 1, 3, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out


class DownNet(nn.Module):

    def __init__(self):
        super(DownNet, self).__init__()
        layers = [nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
                  BatchRenorm2d(64),
                  nn.ReLU()]

        for i in range(7):
            layers.append(nn.Conv2d(64, 64, 3, 1, padding=2, dilation=2))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(64, 64, 3, 1, 1))
        layers.append(BatchRenorm2d(64))
        layers.append(nn.ReLU())
        for i in range(6):
            layers.append(nn.Conv2d(64, 64, 3, 1, padding=2, dilation=2))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(64, 64, 3, 1, 1))
        layers.append(BatchRenorm2d(64))
        layers.append(nn.ReLU())

        layers.append(nn.Conv2d(64, 3, 3, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out


import basicblock as B


class UNetRes(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode='strideconv',
                 upsample_mode='convtranspose'):
        super(UNetRes, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(
            *[B.ResBlock(nc[0], nc[0], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(
            *[B.ResBlock(nc[1], nc[1], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(
            *[B.ResBlock(nc[2], nc[2], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body = B.sequential(
            *[B.ResBlock(nc[3], nc[3], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'),
                                  *[B.ResBlock(nc[2], nc[2], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'),
                                  *[B.ResBlock(nc[1], nc[1], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'),
                                  *[B.ResBlock(nc[0], nc[0], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)

        return x