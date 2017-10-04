from bluntools.layers import *


class ResNeXt_128(nn.Module):
    def __init__(self):
        super(ResNeXt_128, self).__init__()
        # MaxPool
        self.lhs_1x = nn.Sequential(ConvBNReLU(3, 64), ResNeXt(64, 64))
        self.lhs_2x = nn.Sequential(nn.MaxPool2d(2, stride=2), ResNeXt(64, 128))
        self.lhs_4x = nn.Sequential(nn.MaxPool2d(2, stride=2), ResNeXt(128, 256))
        self.lhs_8x = nn.Sequential(nn.MaxPool2d(2, stride=2), ResNeXt(256, 512))

        self.bottom = nn.Sequential(nn.MaxPool2d(2, stride=2), ResNeXt(512, 512))

        # BilinearUp
        self.rhs_8x = UpConcat(512, 512)
        self.rhs_4x = UpConcat(512, 256)
        self.rhs_2x = UpConcat(256, 128)
        self.rhs_1x = UpConcat(128, 64)

        # Classify
        self.classify = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        lhs_1x = self.lhs_1x(x)
        lhs_2x = self.lhs_2x(lhs_1x)
        lhs_4x = self.lhs_4x(lhs_2x)
        lhs_8x = self.lhs_8x(lhs_4x)

        bottom = self.bottom(lhs_8x)

        rhs_8x = self.rhs_8x(lhs_8x, bottom)
        rhs_4x = self.rhs_4x(lhs_4x, rhs_8x)
        rhs_2x = self.rhs_2x(lhs_2x, rhs_4x)
        rhs_1x = self.rhs_1x(lhs_1x, rhs_2x)

        return self.classify(rhs_1x)


class DualPath_128(nn.Module):
    def __init__(self):
        super(DualPath_128, self).__init__()
        # MaxPool
        self.lhs_1x = nn.Sequential(ConvBNReLU(3, 64), DualBlock(64, 64))
        self.lhs_2x = nn.Sequential(nn.MaxPool2d(2, stride=2), DualBlock(64, 128))
        self.lhs_4x = nn.Sequential(nn.MaxPool2d(2, stride=2), DualBlock(128, 256))
        self.lhs_8x = nn.Sequential(nn.MaxPool2d(2, stride=2), DualBlock(256, 512))

        self.bottom = nn.Sequential(nn.MaxPool2d(2, stride=2), DualBlock(512, 512))

        # BilinearUp
        self.rhs_8x = UpConcat(512, 512)
        self.rhs_4x = UpConcat(512, 256)
        self.rhs_2x = UpConcat(256, 128)
        self.rhs_1x = UpConcat(128, 64)

        # Classify
        self.classify = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        lhs_1x = self.lhs_1x(x)
        lhs_2x = self.lhs_2x(lhs_1x)
        lhs_4x = self.lhs_4x(lhs_2x)
        lhs_8x = self.lhs_8x(lhs_4x)

        bottom = self.bottom(lhs_8x)

        rhs_8x = self.rhs_8x(lhs_8x, bottom)
        rhs_4x = self.rhs_4x(lhs_4x, rhs_8x)
        rhs_2x = self.rhs_2x(lhs_2x, rhs_4x)
        rhs_1x = self.rhs_1x(lhs_1x, rhs_2x)

        return self.classify(rhs_1x)


