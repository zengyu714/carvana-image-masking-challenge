from nets.layers import *


class DeepAtrous_128(nn.Module):
    def __init__(self):
        super(DeepAtrous_128, self).__init__()
        self.in_fit = nn.Sequential(ConvReLU(3, 32), ConvReLU(32, 32))

        self.lhs_1x = DownAtrous(32, 64)
        self.lhs_2x = DownAtrous(64, 128)
        self.lhs_4x = DownAtrous(128, 256)
        self.lhs_8x = DownAtrous(256, 256)

        self.rhs_8x = UpBlock(256, 256)
        self.rhs_4x = UpBlock(256, 128)
        self.rhs_2x = UpBlock(128, 64)
        self.rhs_1x = UpBlock(64, 32)

        self.classify = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        lhs_1x, pool = self.lhs_1x(self.in_fit(x))
        lhs_2x, pool = self.lhs_2x(pool)
        lhs_4x, pool = self.lhs_4x(pool)
        lhs_8x, _ = self.lhs_8x(pool)

        _, up = self.rhs_8x(lhs_8x, pool)
        _, up = self.rhs_4x(lhs_4x, up)
        _, up = self.rhs_2x(lhs_2x, up)
        rhs_1x, _ = self.rhs_1x(lhs_1x, up)

        return self.classify(rhs_1x)


class DeepAtrous_HD(nn.Module):
    def __init__(self):
        super(DeepAtrous_HD, self).__init__()
        self.in_fit = nn.Sequential(ConvReLU(3, 4), ConvReLU(4, 4))

        self.lhs_1x = DownAtrous(4, 8)
        self.lhs_2x = DownAtrous(8, 16)
        self.lhs_4x = DownAtrous(16, 32)
        self.lhs_8x = DownAtrous(32, 64)
        self.lhs_16x = DownAtrous(64, 128)
        self.lhs_32x = DownAtrous(128, 256)
        self.lhs_64x = DownAtrous(256, 256)

        self.rhs_64x = UpBlock(256, 256)
        self.rhs_32x = UpBlock(256, 128)
        self.rhs_16x = UpBlock(128, 64)
        self.rhs_8x = UpBlock(64, 32)
        self.rhs_4x = UpBlock(32, 16)
        self.rhs_2x = UpBlock(16, 8)
        self.rhs_1x = UpBlock(8, 4)

        self.classify = nn.Conv2d(8, 2, kernel_size=1)

    def forward(self, x):
        lhs_1x, pool = self.lhs_1x(self.in_fit(x))
        lhs_2x, pool = self.lhs_2x(pool)
        lhs_4x, pool = self.lhs_4x(pool)
        lhs_8x, pool = self.lhs_8x(pool)
        lhs_16x, pool = self.lhs_16x(pool)
        lhs_32x, pool = self.lhs_32x(pool)
        lhs_64x, _ = self.lhs_64x(pool)

        _, up = self.rhs_64x(lhs_64x, pool)
        _, up = self.rhs_32x(lhs_32x, up)
        _, up = self.rhs_16x(lhs_16x, up)
        _, up = self.rhs_8x(lhs_8x, up)
        _, up = self.rhs_4x(lhs_4x, up)
        _, up = self.rhs_2x(lhs_2x, up)
        rhs_1x, _ = self.rhs_1x(lhs_1x, up)

        return self.classify(rhs_1x)


# Experiment
# ----------------------------------------------------------------------
class Atrous_128(nn.Module):
    def __init__(self):
        super(Atrous_128, self).__init__()
        # MaxPool
        self.lhs_1x = nn.Sequential(ConvReLU(3, 64), AtrousBlock(64, 64))
        self.lhs_2x = nn.Sequential(nn.MaxPool2d(2, stride=2), AtrousBlock(64, 128))
        self.lhs_4x = nn.Sequential(nn.MaxPool2d(2, stride=2), AtrousBlock(128, 256))  # w/h: 32

        self.bottom = nn.Sequential(nn.MaxPool2d(2, stride=2), AtrousBlock(256, 256))

        # BilinearUp
        self.rhs_4x = UpConcat(256, 256)
        self.rhs_2x = UpConcat(256, 128)
        self.rhs_1x = UpConcat(128, 64)

        # Classify
        self.classify = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        lhs_1x = self.lhs_1x(x)
        lhs_2x = self.lhs_2x(lhs_1x)
        lhs_4x = self.lhs_4x(lhs_2x)

        bottom = self.bottom(lhs_4x)

        rhs_4x = self.rhs_4x(lhs_4x, bottom)
        rhs_2x = self.rhs_2x(lhs_2x, rhs_4x)
        rhs_1x = self.rhs_1x(lhs_1x, rhs_2x)

        return self.classify(rhs_1x)


class Atrous_256(nn.Module):
    """Reference: https://arxiv.org/pdf/1706.05587.pdf"""

    def __init__(self):
        super(Atrous_256, self).__init__()
        # MaxPool
        self.lhs_1x = nn.Sequential(ConvReLU(3, 32), AtrousBlock(32, 32))
        self.lhs_2x = nn.Sequential(nn.MaxPool2d(2, stride=2), AtrousBlock(32, 64))
        self.lhs_4x = nn.Sequential(nn.MaxPool2d(2, stride=2), AtrousBlock(64, 128))
        self.lhs_8x = nn.Sequential(nn.MaxPool2d(2, stride=2), AtrousBlock(128, 256))  # w/h: 32

        self.bottom = nn.Sequential(nn.MaxPool2d(2, stride=2), AtrousBlock(256, 256))

        # BilinearUp
        self.rhs_8x = UpConcat(256, 256)
        self.rhs_4x = UpConcat(256, 128)
        self.rhs_2x = UpConcat(128, 64)
        self.rhs_1x = UpConcat(64, 32)

        # Classify
        self.classify = nn.Conv2d(32, 2, kernel_size=1)

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


class Atrous_1024(nn.Module):
    """Reference: https://arxiv.org/pdf/1706.05587.pdf"""

    def __init__(self):
        super(Atrous_1024, self).__init__()
        # MaxPool
        self.lhs_1x = nn.Sequential(ConvReLU(3, 8), AtrousBlock(8, 8))
        self.lhs_2x = nn.Sequential(nn.MaxPool2d(2, stride=2), AtrousBlock(8, 16))
        self.lhs_4x = nn.Sequential(nn.MaxPool2d(2, stride=2), AtrousBlock(16, 64))
        self.lhs_8x = nn.Sequential(nn.MaxPool2d(2, stride=2), AtrousBlock(64, 128))
        self.lhs_16x = nn.Sequential(nn.MaxPool2d(2, stride=2), AtrousBlock(128, 256))
        self.lhs_32x = nn.Sequential(nn.MaxPool2d(2, stride=2), AtrousBlock(256, 512))  # w/h: 32

        self.bottom = nn.Sequential(nn.MaxPool2d(2, stride=2), AtrousBlock(512, 512))

        # BilinearUp
        self.rhs_32x = UpConcat(512, 512)
        self.rhs_16x = UpConcat(512, 256)
        self.rhs_8x = UpConcat(256, 128)
        self.rhs_4x = UpConcat(128, 64)
        self.rhs_2x = UpConcat(64, 16)
        self.rhs_1x = UpConcat(16, 8)

        # Classify
        self.classify = nn.Conv2d(8, 2, kernel_size=1)

    def forward(self, x):
        lhs_1x = self.lhs_1x(x)
        lhs_2x = self.lhs_2x(lhs_1x)
        lhs_4x = self.lhs_4x(lhs_2x)
        lhs_8x = self.lhs_8x(lhs_4x)
        lhs_16x = self.lhs_16x(lhs_8x)
        lhs_32x = self.lhs_32x(lhs_16x)

        bottom = self.bottom(lhs_32x)

        rhs_32x = self.rhs_32x(lhs_32x, bottom)
        rhs_16x = self.rhs_16x(lhs_16x, rhs_32x)
        rhs_8x = self.rhs_8x(lhs_8x, rhs_16x)
        rhs_4x = self.rhs_4x(lhs_4x, rhs_8x)
        rhs_2x = self.rhs_2x(lhs_2x, rhs_4x)
        rhs_1x = self.rhs_1x(lhs_1x, rhs_2x)

        return self.classify(rhs_1x)


class AtrousMLs_1024(nn.Module):
    def __init__(self):
        super(AtrousMLs_1024, self).__init__()
        # MaxPool
        self.lhs_1x = nn.Sequential(ConvReLU(3, 8), AtrousBlock(8, 8))
        self.lhs_2x = nn.Sequential(nn.MaxPool2d(2, stride=2), AtrousBlock(8, 16))
        self.lhs_4x = nn.Sequential(nn.MaxPool2d(2, stride=2), AtrousBlock(16, 32))
        self.lhs_8x = nn.Sequential(nn.MaxPool2d(2, stride=2), AtrousBlock(32, 64))
        self.lhs_16x = nn.Sequential(nn.MaxPool2d(2, stride=2), AtrousBlock(64, 128))
        self.lhs_32x = nn.Sequential(nn.MaxPool2d(2, stride=2), AtrousBlock(128, 256))  # w/h: 32

        self.bottom = nn.Sequential(nn.MaxPool2d(2, stride=2), AtrousBlock(256, 256))

        # BilinearUp
        self.rhs_32x = UpConcat(256, 256)
        self.rhs_16x = UpConcat(256, 128)
        self.rhs_8x = UpConcat(128, 64)
        self.rhs_4x = UpConcat(64, 32)
        self.rhs_2x = UpConcat(32, 16)
        self.rhs_1x = UpConcat(16, 8)

        # Classify
        self.classify_64x = nn.Conv2d(256, 2, kernel_size=1)
        self.classify_32x = nn.Conv2d(256, 2, kernel_size=1)
        self.classify_16x = nn.Conv2d(128, 2, kernel_size=1)
        self.classify_8x = nn.Conv2d(64, 2, kernel_size=1)
        self.classify_4x = nn.Conv2d(32, 2, kernel_size=1)
        self.classify_2x = nn.Conv2d(16, 2, kernel_size=1)
        self.classify_1x = nn.Conv2d(8, 2, kernel_size=1)

    def forward(self, x):
        lhs_1x = self.lhs_1x(x)
        lhs_2x = self.lhs_2x(lhs_1x)
        lhs_4x = self.lhs_4x(lhs_2x)
        lhs_8x = self.lhs_8x(lhs_4x)
        lhs_16x = self.lhs_16x(lhs_8x)
        lhs_32x = self.lhs_32x(lhs_16x)

        bottom = self.bottom(lhs_32x)

        rhs_32x = self.rhs_32x(lhs_32x, bottom)
        rhs_16x = self.rhs_16x(lhs_16x, rhs_32x)
        rhs_8x = self.rhs_8x(lhs_8x, rhs_16x)
        rhs_4x = self.rhs_4x(lhs_4x, rhs_8x)
        rhs_2x = self.rhs_2x(lhs_2x, rhs_4x)
        rhs_1x = self.rhs_1x(lhs_1x, rhs_2x)

        return (self.classify_1x(rhs_1x),
                self.classify_2x(rhs_2x),
                self.classify_4x(rhs_4x),
                self.classify_8x(rhs_8x),
                self.classify_16x(rhs_16x),
                self.classify_32x(rhs_32x),
                self.classify_64x(bottom))
