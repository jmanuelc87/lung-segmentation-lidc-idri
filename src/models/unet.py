import keras

from keras.layers import (
    Activation,
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    Conv2DTranspose,
    Concatenate,
)
from models.common import ConvBlock


@keras.saving.register_keras_serializable()
class Encoder(keras.layers.Layer):

    def __init__(
        self,
        *,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs,
    ):
        super().__init__(
            activity_regularizer=activity_regularizer,
            trainable=trainable,
            dtype=dtype,
            autocast=autocast,
            name=name,
        )

        self.filters = kwargs["filters"]
        self.act = kwargs["activation"]
        self.dropout = kwargs["dropout"]
        self.dilation = kwargs["dilation"]

    def build(self, input_shape):
        self.conv1 = ConvBlock(
            filters=self.filters,
            activation=self.act,
            dropout=self.dropout,
            dilation=self.dilation,
        )
        self.pool1 = MaxPooling2D((2, 2))
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "activation": self.act,
                "dropout": self.dropout,
                "dilation": self.dilation,
            }
        )
        return config

    def call(self, x):
        c = self.conv1(x)
        p = self.pool1(c)

        return p, c


@keras.saving.register_keras_serializable()
class Decoder(keras.layers.Layer):

    def __init__(
        self,
        *,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs,
    ):
        super().__init__(
            activity_regularizer=activity_regularizer,
            trainable=trainable,
            dtype=dtype,
            autocast=autocast,
            name=name,
        )

        self.filters = kwargs["filters"]
        self.act = kwargs["activation"]
        self.dropout = kwargs["dropout"]
        self.dilation = kwargs["dilation"]

    def build(self, input_shape):
        self.trans = Conv2DTranspose(self.filters, 2, strides=2, padding="same")
        self.concat = Concatenate()
        self.conv1 = ConvBlock(
            filters=self.filters,
            activation=self.act,
            dropout=self.dropout,
            dilation=self.dilation,
        )
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "activation": self.act,
                "dropout": self.dropout,
                "dilation": self.dilation,
            }
        )
        return config

    def call(self, x, skip):
        x = self.trans(x)
        x = self.concat([x, skip])
        x = self.conv1(x)

        return x


class UNet(keras.Model):

    def __init__(self, num_classes, activation="relu"):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.activation = activation
        self.encoder1 = Encoder(
            filters=64,
            activation=self.activation,
            dropout=0.1,
            dilation=1,
        )
        self.encoder2 = Encoder(
            filters=128,
            activation=self.activation,
            dropout=0.1,
            dilation=1,
        )
        self.encoder3 = Encoder(
            filters=256,
            activation=self.activation,
            dropout=0.1,
            dilation=1,
        )
        self.encoder4 = Encoder(
            filters=512,
            activation=self.activation,
            dropout=0.1,
            dilation=1,
        )

        self.bottleneck = ConvBlock(
            filters=1024,
            activation=self.activation,
            dropout=0.1,
            dilation=2,
        )

        self.decoder1 = Decoder(
            filters=512,
            activation=self.activation,
            dropout=0.1,
            dilation=1,
        )
        self.decoder2 = Decoder(
            filters=256,
            activation=self.activation,
            dropout=0.1,
            dilation=1,
        )
        self.decoder3 = Decoder(
            filters=128,
            activation=self.activation,
            dropout=0.1,
            dilation=1,
        )
        self.decoder4 = Decoder(
            filters=64,
            activation=self.activation,
            dropout=0.1,
            dilation=1,
        )

        self.conv1x1 = Conv2D(
            self.num_classes,
            1,
            padding="same",
            use_bias=False,
        )

        self.head = Activation("sigmoid")

    def call(self, input_tensor):
        p1, c1 = self.encoder1(input_tensor)
        p2, c2 = self.encoder2(p1)
        p3, c3 = self.encoder3(p2)
        p4, c4 = self.encoder4(p3)

        b0 = self.bottleneck(p4)

        u0 = self.decoder1(b0, c4)
        u1 = self.decoder2(u0, c3)
        u2 = self.decoder3(u1, c2)
        u3 = self.decoder4(u2, c1)

        out = self.conv1x1(u3)
        out = self.head(out)

        return out
