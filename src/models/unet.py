import torch
import logging

from models.common import ConvBlock


logger = logging.getLogger(__name__)


class Encoder(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.build()

    def build(self):
        self.conv = ConvBlock(self.in_channels, self.out_channels, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, input_tensor):
        cx = self.conv(input_tensor)
        px = self.pool(cx)

        return px, cx


class Decoder(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.build()

    def build(self):
        self.pose = torch.nn.ConvTranspose2d(
            self.in_channels,
            self.out_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv = ConvBlock(
            self.in_channels,
            self.out_channels,
            padding=1,
        )

    def forward(self, input_tensor, skip):
        x = self.pose(input_tensor)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)

        return x


class UNet(torch.nn.Module):

    def __init__(self, in_channels, num_classes, steps=(64, 128, 256, 512)):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.steps = steps

        self.build()

    def build(self):
        self.middle = ConvBlock(self.steps[-1], self.steps[-1] * 2, padding=1)
        self.head = torch.nn.Conv2d(self.steps[0], self.num_classes, kernel_size=1)
        self.encoders = []
        self.decoders = []

        next_step = self.in_channels
        for step in self.steps:
            e = Encoder(next_step, step)
            self.encoders.append(e)
            next_step = step

        self.encoders = torch.nn.ModuleList(self.encoders)

        next_step = next_step * 2
        for step in sorted(self.steps, reverse=True):
            d = Decoder(next_step, step)
            self.decoders.append(d)
            next_step = step

        self.decoders = torch.nn.ModuleList(self.decoders)

    def forward(self, input_tensor):
        skip_conn = []
        x = input_tensor

        for encoder in self.encoders:
            p0, c0 = encoder(x)
            x = p0
            skip_conn.append(c0)

        u0 = self.middle(x)

        for i, decoder in enumerate(self.decoders):
            idx = len(self.steps) - 1 - i
            u0 = decoder(u0, skip_conn[idx])

        x = self.head(u0)

        return x
