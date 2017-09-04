from nets.layers import *


class UNet_128(nn.Module):
    """Input: [batch_size, 3, 128, 128]"""

    def __init__(self):
        super(UNet_128, self).__init__()
        # Conv
        self.lhs_1x = ConvReLU(3, 64)
        self.lhs_2x = nn.Sequential(ConvReLU(64, 128, stride=2), ConvReLU(128, 128))
        self.lhs_4x = nn.Sequential(ConvReLU(128, 256, stride=2), ConvX3(256, 256))
        self.lhs_8x = nn.Sequential(ConvReLU(256, 512, stride=2), ConvX3(512, 512))

        # Bottom 16x <==> [batch_size, 3, 8, 8]
        self.bottom = nn.Sequential(ConvReLU(512, 1024, stride=2), ConvX3(1024, 1024))

        # ConvTranspose
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
        # Conv
        self.lhs_1x = ConvReLU(3, 32)
        self.lhs_2x = nn.Sequential(ConvReLU(32, 64, stride=2), ConvReLU(64, 64))
        self.lhs_4x = nn.Sequential(ConvReLU(64, 128, stride=2), ConvReLU(128, 128))
        self.lhs_8x = nn.Sequential(ConvReLU(128, 256, stride=2), ConvX3(256, 256))
        self.lhs_16x = nn.Sequential(ConvReLU(256, 512, stride=2), ConvX3(512, 512))

        # Bottom 32x <==> [batch_size, 3, 8, 8]
        self.bottom = nn.Sequential(ConvReLU(512, 1024, stride=2), ConvX3(1024, 1024))

        # ConvTranspose
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
        # Conv
        self.lhs_1x = ConvReLU(3, 16)
        self.lhs_2x = nn.Sequential(ConvReLU(16, 32, stride=2), ConvReLU(32, 32))
        self.lhs_4x = nn.Sequential(ConvReLU(32, 64, stride=2), ConvReLU(64, 64))
        self.lhs_8x = nn.Sequential(ConvReLU(64, 128, stride=2), ConvReLU(128, 128))
        self.lhs_16x = nn.Sequential(ConvReLU(128, 256, stride=2), ConvX3(256, 256))
        self.lhs_32x = nn.Sequential(ConvReLU(256, 512, stride=2), ConvX3(512, 512))

        # Bottom 64x <==> [batch_size, 3, 8, 8]
        self.bottom = nn.Sequential(ConvReLU(512, 1024, stride=2), ConvX3(1024, 1024))

        # ConvTranspose
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
        # Conv
        self.lhs_1x = ConvReLU(3, 8)
        self.lhs_2x = nn.Sequential(ConvReLU(8, 16, stride=2), ConvReLU(16, 16))
        self.lhs_4x = nn.Sequential(ConvReLU(16, 32, stride=2), ConvX3(32, 32))
        self.lhs_8x = nn.Sequential(ConvReLU(32, 64, stride=2), ConvX3(64, 64))
        self.lhs_16x = nn.Sequential(ConvReLU(64, 128, stride=2), ConvX3(128, 128))
        self.lhs_32x = nn.Sequential(ConvReLU(128, 256, stride=2), ConvX3(256, 256))
        self.lhs_64x = nn.Sequential(ConvReLU(256, 512, stride=2), ConvX3(512, 512))

        # Bottom 128x
        self.bottom = nn.Sequential(ConvReLU(512, 1024, stride=2), ConvX3(1024, 1024))

        # ConvTranspose
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
