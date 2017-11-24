from keras import layers, models
from keras.optimizers import Adam

"""https://github.com/zhixuhao/unet"""


class UNet(object):
    def __init__(self, source_shape, num_class):
        self.source_shape = source_shape
        self.num_class = num_class
        self.model = None
        self.create_model(source_shape, num_class)
        self.compile()
        print("UNet initialized")

    def _CommonConv(self, filters, name, kernel_size = 3):
        return layers.Conv2D(filters=filters, kernel_size=kernel_size, activation="relu", padding="same", name=name)

    def _CommonPool(self):
        return layers.MaxPooling2D(pool_size = 2)

    def _CommonDeconv(self, filters, name, inputs):
        up = layers.UpSampling2D(size = 2)(inputs)
        return self._CommonConv(filters, name, kernel_size = 2)(up)

    def _CommonMerge(self, a, b):
        return layers.Concatenate(axis = 3)([a, b])

    def create_model(self, source_shape, num_class):
        inputs = layers.Input(shape = source_shape)#256x256

        conv1 = self._CommonConv(32, "conv1_1")(inputs)#254x254
        conv1 = self._CommonConv(32, "conv1_2")(conv1)#252x252
        pool1 = self._CommonPool()(conv1)#126x126

        conv2 = self._CommonConv(64, "conv2_1")(pool1)#124x124
        conv2 = self._CommonConv(64, "conv2_2")(conv2)#122x122
        pool2 = self._CommonPool()(conv2)#61x61

        conv3 = self._CommonConv(128, "conv3_1")(pool2)#59x59
        conv3 = self._CommonConv(128, "conv3_2")(conv3)#57x57

        up4 = self._CommonDeconv(64, "up4", conv3)
        merge4 = self._CommonMerge(conv2, up4)
        conv4 = self._CommonConv(64, "conv4_1")(merge4)
        conv4 = self._CommonConv(64, "conv4_2")(conv4)

        up5 = self._CommonDeconv(32, "up5", conv4)
        merge5 = self._CommonMerge(conv1, up5)
        conv5 = self._CommonConv(32, "conv5_1")(merge5)
        conv5 = self._CommonConv(32, "conv5_2")(conv5)

        conv6 = layers.Conv2D(filters=num_class, kernel_size=1, activation="sigmoid")(conv5)
        outputs = layers.Reshape((256, 256))(conv6)

        self.model = models.Model(inputs = inputs, outputs = outputs)
        print("UNet constructed")

    def compile(self):
        self.model.compile(optimizer = Adam(lr = 1e-4), loss = "binary_crossentropy", metrics = ["accuracy"])
        print("UNet compiled")
