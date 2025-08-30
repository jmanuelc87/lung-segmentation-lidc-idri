import torch
import pytest
import logging

from torchinfo import summary

from models.common import ConvBlock
from models.unet import Encoder, Decoder, UNet

logger = logging.getLogger(__name__)


def test_common_convblock():
    in_channels = 3
    out_channels = 64

    conv = ConvBlock(in_channels, out_channels)

    tensor = torch.rand((16, in_channels, 6, 6))

    out_tensor = conv(tensor)

    logger.info(out_tensor.shape)

    assert out_tensor.shape[1] == out_channels


def test_unet_encoder():
    in_channels = 3
    out_channels = 64

    encoder = Encoder(in_channels, out_channels)

    tensor = torch.rand((16, in_channels, 12, 12))

    p0, c0 = encoder(tensor)

    logger.info("%s, %s", p0.shape, c0.shape)

    assert p0.shape[1] == out_channels
    assert p0.shape[2] == 6
    assert p0.shape[3] == 6


def test_unet_decoder():
    decoder = Decoder(32, 16)

    tensor = torch.rand((16, 32, 12, 12))
    skip = torch.rand((16, 16, 24, 24))

    out_tensor = decoder(tensor, skip)

    logger.info(out_tensor.shape)

    assert out_tensor.shape[1] == 16
    assert out_tensor.shape[2] == 24
    assert out_tensor.shape[3] == 24


def test_unet_model():
    unet = UNet(in_channels=1, num_classes=1)

    input_size = (16, 1, 384, 384)

    tensor = torch.rand(input_size)

    model_stats = summary(unet, input_size)

    logger.info("%s", model_stats)

    out_tensor = unet(tensor)

    logger.info(out_tensor.shape)
