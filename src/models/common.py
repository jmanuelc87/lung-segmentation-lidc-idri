import torch


class ConvBlock(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        self.build()

    def build(self):
        self.conv1 = torch.nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=self.bias,
        )
        self.norm1 = torch.nn.BatchNorm2d(self.out_channels)
        self.relu1 = torch.nn.ReLU(inplace=False)

    def forward(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.norm1(x)
        x = self.relu1(x)

        return x
