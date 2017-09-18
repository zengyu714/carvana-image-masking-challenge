from nets.layers import *

"""
Maxpool and BilinearUpsample
"""


class DeepUNet_128(nn.Module):
    """Input: [batch_size, 3, 128, 128]"""

    def __init__(self):
        super(DeepUNet_128, self).__init__()
        self.in_fit = nn.Sequential(ConvReLU(3, 32), ConvReLU(32, 32))

        self.lhs_1x = DownBlock(32, 64)
        self.lhs_2x = DownBlock(64, 128)
        self.lhs_4x = DownBlock(128, 256)
        self.lhs_8x = DownBlock(256, 512)
        self.lhs_16x = DownBlock(512, 512)

        self.rhs_16x = UpBlock(512, 512)
        self.rhs_8x = UpBlock(512, 256)
        self.rhs_4x = UpBlock(256, 128)
        self.rhs_2x = UpBlock(128, 64)
        self.rhs_1x = UpBlock(64, 32)

        self.classify = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        lhs_1x, pool = self.lhs_1x(self.in_fit(x))
        lhs_2x, pool = self.lhs_2x(pool)
        lhs_4x, pool = self.lhs_4x(pool)
        lhs_8x, pool = self.lhs_8x(pool)
        lhs_16x, _ = self.lhs_16x(pool)

        _, up = self.rhs_16x(lhs_16x, pool)
        _, up = self.rhs_8x(lhs_8x, up)
        _, up = self.rhs_4x(lhs_4x, up)
        _, up = self.rhs_2x(lhs_2x, up)
        rhs_1x, _ = self.rhs_1x(lhs_1x, up)

        return self.classify(rhs_1x)


class DeepUMLs_128(nn.Module):
    """
    Deep-UNet-Multiple-Loss
    Input: [batch_size, 3, 128, 128]
    """

    def __init__(self):
        super(DeepUMLs_128, self).__init__()
        self.in_fit = nn.Sequential(ConvReLU(3, 32), ConvReLU(32, 32))

        self.lhs_1x = DownBlock(32, 64)
        self.lhs_2x = DownBlock(64, 128)
        self.lhs_4x = DownBlock(128, 256)
        self.lhs_8x = DownBlock(256, 512)
        self.lhs_16x = DownBlock(512, 512)

        self.rhs_16x = UpBlock(512, 512)
        self.rhs_8x = UpBlock(512, 256)
        self.rhs_4x = UpBlock(256, 128)
        self.rhs_2x = UpBlock(128, 64)
        self.rhs_1x = UpBlock(64, 32)

        self.classify_16x = nn.Conv2d(512, 2, kernel_size=1)
        self.classify_8x = nn.Conv2d(512, 2, kernel_size=1)
        self.classify_4x = nn.Conv2d(256, 2, kernel_size=1)
        self.classify_2x = nn.Conv2d(128, 2, kernel_size=1)
        self.classify_1x = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        lhs_1x, pool = self.lhs_1x(self.in_fit(x))
        lhs_2x, pool = self.lhs_2x(pool)
        lhs_4x, pool = self.lhs_4x(pool)
        lhs_8x, pool = self.lhs_8x(pool)
        lhs_16x, _ = self.lhs_16x(pool)

        rhs_16x, up = self.rhs_16x(lhs_16x, pool)
        rhs_8x, up = self.rhs_8x(lhs_8x, up)
        rhs_4x, up = self.rhs_4x(lhs_4x, up)
        rhs_2x, up = self.rhs_2x(lhs_2x, up)
        rhs_1x, _ = self.rhs_1x(lhs_1x, up)

        return (self.classify_1x(rhs_1x),
                self.classify_2x(rhs_2x),
                self.classify_4x(rhs_4x),
                self.classify_8x(rhs_8x),
                self.classify_16x(rhs_16x))


class DeepUNet_HD(nn.Module):
    """Input: [batch_size, 3, 1280, 1918]"""

    def __init__(self):
        super(DeepUNet_HD, self).__init__()
        self.in_fit = nn.Sequential(ConvReLU(3, 4), ConvReLU(4, 4))

        self.lhs_1x = DownBlock(4, 8)
        self.lhs_2x = DownBlock(8, 16)
        self.lhs_4x = DownBlock(16, 32)
        self.lhs_8x = DownBlock(32, 64)
        self.lhs_16x = DownBlock(64, 128)
        self.lhs_32x = DownBlock(128, 256)
        self.lhs_64x = DownBlock(256, 512)
        self.lhs_128x = DownBlock(512, 512)

        self.rhs_128x = UpBlock(512, 512)
        self.rhs_64x = UpBlock(512, 256)
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
        lhs_64x, pool = self.lhs_64x(pool)
        lhs_128x, _ = self.lhs_128x(pool)

        _, up = self.rhs_128x(lhs_128x, pool)
        _, up = self.rhs_64x(lhs_64x, up)
        _, up = self.rhs_32x(lhs_32x, up)
        _, up = self.rhs_16x(lhs_16x, up)
        _, up = self.rhs_8x(lhs_8x, up)
        _, up = self.rhs_4x(lhs_4x, up)
        _, up = self.rhs_2x(lhs_2x, up)
        rhs_1x, _ = self.rhs_1x(lhs_1x, up)

        return self.classify(rhs_1x)


class DeepUMLs_HD(nn.Module):
    """
    Deep-UNet-Multiple-Loss
    Input: [batch_size, 3, 1280, 1918]
    """

    def __init__(self):
        super(DeepUMLs_HD, self).__init__()
        self.in_fit = nn.Sequential(ConvReLU(3, 4), ConvReLU(4, 4))

        self.lhs_1x = DownBlock(4, 8)
        self.lhs_2x = DownBlock(8, 16)
        self.lhs_4x = DownBlock(16, 32)
        self.lhs_8x = DownBlock(32, 64)
        self.lhs_16x = DownBlock(64, 128)
        self.lhs_32x = DownBlock(128, 256)
        self.lhs_64x = DownBlock(256, 512)
        self.lhs_128x = DownBlock(512, 512)

        self.rhs_128x = UpBlock(512, 512)
        self.rhs_64x = UpBlock(512, 256)
        self.rhs_32x = UpBlock(256, 128)
        self.rhs_16x = UpBlock(128, 64)
        self.rhs_8x = UpBlock(64, 32)
        self.rhs_4x = UpBlock(32, 16)
        self.rhs_2x = UpBlock(16, 8)
        self.rhs_1x = UpBlock(8, 4)

        self.classify_128x = nn.Conv2d(512, 2, kernel_size=1)
        self.classify_32x = nn.Conv2d(256, 2, kernel_size=1)
        self.classify_8x = nn.Conv2d(64, 2, kernel_size=1)
        self.classify_1x = nn.Conv2d(8, 2, kernel_size=1)

    def forward(self, x):
        lhs_1x, pool = self.lhs_1x(self.in_fit(x))
        lhs_2x, pool = self.lhs_2x(pool)
        lhs_4x, pool = self.lhs_4x(pool)
        lhs_8x, pool = self.lhs_8x(pool)
        lhs_16x, pool = self.lhs_16x(pool)
        lhs_32x, pool = self.lhs_32x(pool)
        lhs_64x, pool = self.lhs_64x(pool)
        lhs_128x, _ = self.lhs_128x(pool)

        rhs_128x, up = self.rhs_128x(lhs_128x, pool)
        _, up = self.rhs_64x(lhs_64x, up)
        rhs_32x, up = self.rhs_32x(lhs_32x, up)
        _, up = self.rhs_16x(lhs_16x, up)
        rhs_8x, up = self.rhs_8x(lhs_8x, up)
        _, up = self.rhs_4x(lhs_4x, up)
        _, up = self.rhs_2x(lhs_2x, up)
        rhs_1x, _ = self.rhs_1x(lhs_1x, up)

        return (self.classify_1x(rhs_1x),
                self.classify_8x(rhs_8x),
                self.classify_32x(rhs_32x),
                self.classify_128x(rhs_128x))


# Experiment
# ----------------------------------------------------------------------
class UNet_128(nn.Module):
    """Input: [batch_size, 3, 128, 128]"""

    def __init__(self):
        super(UNet_128, self).__init__()
        # MaxPool
        self.lhs_1x = nn.Sequential(ConvReLU(3, 128), ConvReLU(128, 128))
        self.lhs_2x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(128, 256), ConvReLU(256, 256))
        self.lhs_4x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(256, 512), ConvReLU(512, 512))
        self.lhs_8x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(512, 1024), ConvReLU(1024, 1024))

        # Bottom 16x <==> [batch_size, 3, 8, 8]
        self.bottom = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(1024, 1024), ConvReLU(1024, 1024))

        # BilinearUp
        self.rhs_8x = UpConcat(1024, 1024)
        self.rhs_4x = UpConcat(1024, 512)
        self.rhs_2x = UpConcat(512, 256)
        self.rhs_1x = UpConcat(256, 128)

        # Classify
        self.classify = nn.Conv2d(128, 2, kernel_size=1)

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
        self.lhs_1x = nn.Sequential(ConvReLU(3, 64), ConvReLU(64, 64))
        self.lhs_2x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(64, 128), ConvReLU(128, 128))
        self.lhs_4x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(128, 256), ConvReLU(256, 256))
        self.lhs_8x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(256, 512), ConvReLU(512, 512))
        self.lhs_16x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(512, 1024), ConvReLU(1024, 1024))

        # Bottom 32x <==> [batch_size, 3, 8, 8]
        self.bottom = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(1024, 1024), ConvReLU(1024, 1024))

        # BilinearUp
        self.rhs_16x = UpConcat(1024, 1024)
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
        self.lhs_1x = nn.Sequential(ConvReLU(3, 64), ConvReLU(64, 64))
        self.lhs_2x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(64, 128), ConvReLU(128, 128))
        self.lhs_4x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(128, 256), ConvReLU(256, 256))
        self.lhs_8x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(256, 512), ConvReLU(512, 512))
        self.lhs_16x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(512, 1024), ConvReLU(1024, 1024))

        # Bottom 32x <==> [batch_size, 3, 16, 16]
        self.bottom = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(1024, 1024), ConvReLU(1024, 1024))

        # BilinearUp
        self.rhs_16x = UpConcat(1024, 1024)
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
        lhs_16x = self.lhs_16x(lhs_8x)

        bottom = self.bottom(lhs_16x)

        rhs_16x = self.rhs_16x(lhs_16x, bottom)
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

        # Bottom 64x <==> [batch_size, 3, 16, 16]
        self.bottom = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(256, 512), ConvReLU(512, 512))

        # BilinearUp
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

        bottom = self.bottom(lhs_32x)

        rhs_32x = self.rhs_32x(lhs_32x, bottom)
        rhs_16x = self.rhs_16x(lhs_16x, rhs_32x)
        rhs_8x = self.rhs_8x(lhs_8x, rhs_16x)
        rhs_4x = self.rhs_4x(lhs_4x, rhs_8x)
        rhs_2x = self.rhs_2x(lhs_2x, rhs_4x)
        rhs_1x = self.rhs_1x(lhs_1x, rhs_2x)

        return self.classify(rhs_1x)


class UNet_HD(nn.Module):
    """Input: [batch_size, 3, 1280, 1918]"""

    def __init__(self):
        super(UNet_HD, self).__init__()
        # MaxPool
        self.lhs_1x = nn.Sequential(ConvReLU(3, 8), ConvReLU(8, 8))
        self.lhs_2x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(8, 16), ConvReLU(16, 16))
        self.lhs_4x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(16, 32), ConvReLU(32, 32))
        self.lhs_8x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(32, 64), ConvReLU(64, 64))
        self.lhs_16x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(64, 128), ConvReLU(128, 128))
        self.lhs_32x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(128, 256), ConvReLU(256, 256))

        # Bottom 64x <==> [batch_size, 3, 16, 16]
        self.bottom = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(256, 512), ConvReLU(512, 512))

        # BilinearUp
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

        bottom = self.bottom(lhs_32x)

        rhs_32x = self.rhs_32x(lhs_32x, bottom)
        rhs_16x = self.rhs_16x(lhs_16x, rhs_32x)
        rhs_8x = self.rhs_8x(lhs_8x, rhs_16x)
        rhs_4x = self.rhs_4x(lhs_4x, rhs_8x)
        rhs_2x = self.rhs_2x(lhs_2x, rhs_4x)
        rhs_1x = self.rhs_1x(lhs_1x, rhs_2x)

        return self.classify(rhs_1x)


class UNetReLU_128(nn.Module):
    """Input: [batch_size, 3, 128, 128] + Non-linear activations"""

    def __init__(self, act=nn.ReLU()):
        super(UNet_128, self).__init__()
        # MaxPool
        self.lhs_1x = nn.Sequential(ConvReLU(3, 128), ConvReLU(128, 128))
        self.lhs_2x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(128, 256, relu=act),
                                    ConvReLU(256, 256, relu=act))
        self.lhs_4x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(256, 512, relu=act),
                                    ConvReLU(512, 512, relu=act))
        self.lhs_8x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(512, 1024, relu=act),
                                    ConvReLU(1024, 1024, relu=act))

        # Bottom 16x <==> [batch_size, 3, 8, 8]
        self.bottom = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(1024, 1024, relu=act),
                                    ConvReLU(1024, 1024, relu=act))

        # BilinearUp
        self.rhs_8x = UpConcat(1024, 1024)
        self.rhs_4x = UpConcat(1024, 512)
        self.rhs_2x = UpConcat(512, 256)
        self.rhs_1x = UpConcat(256, 128)

        # Classify
        self.classify = nn.Conv2d(128, 2, kernel_size=1)

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


class UNetUMLs_1024(nn.Module):
    """Input: [batch_size, 3, 1024, 1024]"""

    def __init__(self):
        super(UNetUMLs_1024, self).__init__()
        # MaxPool
        self.lhs_1x = nn.Sequential(ConvReLU(3, 8), ConvReLU(8, 8))
        self.lhs_2x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(8, 16), ConvReLU(16, 16))
        self.lhs_4x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(16, 32), ConvReLU(32, 32))
        self.lhs_8x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(32, 64), ConvReLU(64, 64))
        self.lhs_16x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(64, 128), ConvReLU(128, 128))
        self.lhs_32x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(128, 256), ConvReLU(256, 256))

        # Bottom 64x <==> [batch_size, 3, 16, 16]
        self.bottom = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvReLU(256, 512), ConvReLU(512, 512))

        # BilinearUp
        self.rhs_32x = UpConcat(512, 256)
        self.rhs_16x = UpConcat(256, 128)
        self.rhs_8x = UpConcat(128, 64)
        self.rhs_4x = UpConcat(64, 32)
        self.rhs_2x = UpConcat(32, 16)
        self.rhs_1x = UpConcat(16, 8)

        # Classify
        self.classify_64x = nn.Conv2d(512, 2, kernel_size=1)
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
