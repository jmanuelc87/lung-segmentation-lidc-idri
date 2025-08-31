import keras

from keras.layers import Conv2D, BatchNormalization, Dropout, Activation, Add


@keras.saving.register_keras_serializable()
class ConvBlock(keras.layers.Layer):

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
        self.conv1 = Conv2D(
            self.filters,
            3,
            padding="same",
            kernel_initializer="he_normal",
            dilation_rate=self.dilation,
            use_bias=False,
        )
        self.conv2 = Conv2D(
            self.filters,
            3,
            padding="same",
            kernel_initializer="he_normal",
            dilation_rate=self.dilation,
            use_bias=False,
        )
        self.conv3 = Conv2D(
            self.filters,
            1,
            padding="same",
            use_bias=False,
        )
        self.batc1 = BatchNormalization()
        self.batc2 = BatchNormalization()
        self.batc3 = BatchNormalization()
        self.drop1 = Dropout(self.dropout)
        self.acti1 = Activation(self.act)
        self.acti2 = Activation(self.act)
        self.acti3 = Activation(self.act)
        self.add1 = Add()
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
        x_in = x
        x = self.conv1(x)
        x = self.batc1(x)
        x = self.acti1(x)

        if self.dropout > 0:
            x = self.drop1(x)

        x = self.conv2(x)
        x = self.batc2(x)

        if x_in.shape[-1] != self.filters:
            x_in = self.conv3(x_in)
            x_in = self.batc3(x_in)

        x = self.add1([x, x_in])
        x = self.acti3(x)

        return x
