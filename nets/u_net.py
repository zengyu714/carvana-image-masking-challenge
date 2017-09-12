from nets.layers import *

"""
Maxpool and BilinearUpsample
"""


class UNet_128(nn.Module):
    """Input: [batch_size, 3, 128, 128]"""

    def __init__(self):
        super(UNet_128, self).__init__()
        # MaxPool
        self.lhs_1x = nn.Sequential(ConvReLU(3, 64), ConvReLU(64, 64))
        self.lhs_2x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(64, 128), ConvReLU(128, 128))
        self.lhs_4x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(128, 256), ConvReLU(256, 256))
        self.lhs_8x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(256, 512), ConvReLU(512, 512))

        # Bottom 16x <==> [batch_size, 3, 8, 8]
        self.bottom = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(512, 1024), ConvReLU(1024, 1024))

        # BilinearUp
        self.rhs_8x = UpConcat(1024, 512)
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


class UNet_256(nn.Module):
    """Input: [batch_size, 3, 256, 256]"""

    def __init__(self):
        super(UNet_256, self).__init__()
        # MaxPool
        self.lhs_1x = nn.Sequential(ConvReLU(3, 32), ConvReLU(32, 32))
        self.lhs_2x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(32, 64), ConvReLU(64, 64))
        self.lhs_4x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(64, 128), ConvReLU(128, 128))
        self.lhs_8x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(128, 256), ConvReLU(256, 256))
        self.lhs_16x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(256, 512), ConvReLU(512, 512))

        # Bottom 32x <==> [batch_size, 3, 8, 8]
        self.bottom = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(512, 1024), ConvReLU(1024, 1024))

        # BilinearUp
        self.rhs_16x = UpConcat(1024, 512)
        self.rhs_8x = UpConcat(512, 256)
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
        lhs_16x = self.lhs_16x(lhs_8x)

        bottom = self.bottom(lhs_16x)

        rhs_16x = self.rhs_16x(lhs_16x, bottom)
        rhs_8x = self.rhs_8x(lhs_8x, rhs_16x)
        rhs_4x = self.rhs_4x(lhs_4x, rhs_8x)
        rhs_2x = self.rhs_2x(lhs_2x, rhs_4x)
        rhs_1x = self.rhs_1x(lhs_1x, rhs_2x)

        return self.classify(rhs_1x)


class UNet_512(nn.Module):
    """Input: [batch_size, 3, 512, 512]"""

    def __init__(self):
        super(UNet_512, self).__init__()
        # MaxPool
        self.lhs_1x = nn.Sequential(ConvReLU(3, 16), ConvReLU(16, 16))
        self.lhs_2x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(16, 32), ConvReLU(32, 32))
        self.lhs_4x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(32, 64), ConvReLU(64, 64))
        self.lhs_8x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(64, 128), ConvReLU(128, 128))
        self.lhs_16x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(128, 256), ConvReLU(256, 256))
        self.lhs_32x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(256, 512), ConvReLU(512, 512))

        # Bottom 64x <==> [batch_size, 3, 8, 8]
        self.bottom = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(512, 1024), ConvReLU(1024, 1024))

        # BilinearUp
        self.rhs_32x = UpConcat(1024, 512)
        self.rhs_16x = UpConcat(512, 256)
        self.rhs_8x = UpConcat(256, 128)
        self.rhs_4x = UpConcat(128, 64)
        self.rhs_2x = UpConcat(64, 32)
        self.rhs_1x = UpConcat(32, 16)

        # Classify
        self.classify = nn.Conv2d(16, 2, kernel_size=1)

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


class UNet_1024(nn.Module):
    """Input: [batch_size, 3, 1024, 1024]"""

    def __init__(self):
        super(UNet_1024, self).__init__()
        # MaxPool
        self.lhs_1x = nn.Sequential(ConvReLU(3, 8), ConvReLU(8, 8))
        self.lhs_2x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(8, 16), ConvReLU(16, 16))
        self.lhs_4x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(16, 32), ConvReLU(32, 32))
        self.lhs_8x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(32, 64), ConvReLU(64, 64))
        self.lhs_16x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(64, 128), ConvReLU(128, 128))
        self.lhs_32x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(128, 256), ConvReLU(256, 256))
        self.lhs_64x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(256, 512), ConvReLU(512, 512))

        # Bottom 128x
        self.bottom = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(512, 1024), ConvReLU(1024, 1024))

        # BilinearUp
        self.rhs_64x = UpConcat(1024, 512)
        self.rhs_32x = UpConcat(512, 256)
        self.rhs_16x = UpConcat(256, 128)
        self.rhs_8x = UpConcat(128, 64)
        self.rhs_4x = UpConcat(64, 32)
        self.rhs_2x = UpConcat(32, 16)
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
        lhs_64x = self.lhs_64x(lhs_32x)

        bottom = self.bottom(lhs_64x)

        rhs_64x = self.rhs_64x(lhs_64x, bottom)
        rhs_32x = self.rhs_32x(lhs_32x, rhs_64x)
        rhs_16x = self.rhs_16x(lhs_16x, rhs_32x)
        rhs_8x = self.rhs_8x(lhs_8x, rhs_16x)
        rhs_4x = self.rhs_4x(lhs_4x, rhs_8x)
        rhs_2x = self.rhs_2x(lhs_2x, rhs_4x)
        rhs_1x = self.rhs_1x(lhs_1x, rhs_2x)

        return self.classify(rhs_1x)


class UNetReLU_128(nn.Module):
    """Input: [batch_size, 3, 128, 128] + Non-linear activations"""

    def __init__(self, act=nn.ReLU()):
        super(UNetReLU_128, self).__init__()
        # MaxPool
        self.lhs_1x = nn.Sequential(ConvReLU(3, 64), ConvReLU(64, 64))
        self.lhs_2x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(64, 128, relu=act),
                                    ConvReLU(128, 128, relu=act))
        self.lhs_4x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(128, 256, relu=act),
                                    ConvReLU(256, 256, relu=act))
        self.lhs_8x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(256, 512, relu=act),
                                    ConvReLU(512, 512, relu=act))

        # Bottom 16x <==> [batch_size, 3, 8, 8]
        self.bottom = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(512, 1024, relu=act),
                                    ConvReLU(1024, 1024, relu=act))

        # BilinearUp
        self.rhs_8x = UpConcat(1024, 512)
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


class UNetShallow_128(nn.Module):
    """Input: [batch_size, 3, 128, 128]"""

    def __init__(self):
        super(UNetShallow_128, self).__init__()
        # MaxPool
        self.lhs_1x = nn.Sequential(ConvReLU(3, 64), ConvReLU(64, 64))
        self.lhs_2x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(64, 128), ConvReLU(128, 128))
        self.lhs_4x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(128, 256), ConvReLU(256, 256))

        # Bottom 8x <==> [batch_size, 3, 8, 8]
        self.bottom = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(256, 256), ConvReLU(256, 256))

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


class UNetMultiLoss_128(nn.Module):
    """Input: [batch_size, 3, 128, 128] + Loss Refinement"""

    def __init__(self):
        super(UNetMultiLoss_128, self).__init__()
        # MaxPool
        self.lhs_1x = nn.Sequential(ConvReLU(3, 64), ConvReLU(64, 64))
        self.lhs_2x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(64, 128), ConvReLU(128, 128))
        self.lhs_4x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(128, 256), ConvReLU(256, 256))
        self.lhs_8x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(256, 512), ConvReLU(512, 512))

        # Bottom 16x <==> [batch_size, 3, 8, 8]
        self.bottom = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(512, 512), ConvReLU(512, 512))

        # BilinearUp
        self.rhs_8x = UpConcat(512, 512)
        self.rhs_4x = UpConcat(512, 256)
        self.rhs_2x = UpConcat(256, 128)
        self.rhs_1x = UpConcat(128, 64)

        # Classify
        self.classify_8x = nn.Conv2d(512, 2, kernel_size=1)
        self.classify_4x = nn.Conv2d(256, 2, kernel_size=1)
        self.classify_2x = nn.Conv2d(128, 2, kernel_size=1)
        self.classify_1x = nn.Conv2d(64, 2, kernel_size=1)

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

        return (self.classify_1x(rhs_1x),
                self.classify_2x(rhs_2x),
                self.classify_4x(rhs_4x),
                self.classify_8x(rhs_8x))
