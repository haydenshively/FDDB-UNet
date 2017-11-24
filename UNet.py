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
        #return layers.merge([a, b], mode = "concat", concat_axis = 3)#DEPRECATED
        return layers.Concatenate(axis = 3)([a, b])

    def create_model(self, source_shape, num_class):
        inputs = layers.Input(shape = source_shape)

        conv1 = self._CommonConv(32, "conv1_1")(inputs)
        conv1 = self._CommonConv(32, "conv1_2")(conv1)
        pool1 = self._CommonPool()(conv1)

        conv2 = self._CommonConv(64, "conv2_1")(pool1)
        conv2 = self._CommonConv(64, "conv2_2")(conv2)
        pool2 = self._CommonPool()(conv2)

        conv3 = self._CommonConv(128, "conv3_1")(pool2)
        conv3 = self._CommonConv(128, "conv3_2")(conv3)
        pool3 = self._CommonPool()(conv3)

        conv4 = self._CommonConv(256, "conv4_1")(pool3)
        conv4 = self._CommonConv(256, "conv4_2")(conv4)
        drop4 = layers.Dropout(0.5)(conv4)
        pool4 = self._CommonPool()(drop4)

        conv5 = self._CommonConv(512, "conv5_1")(pool4)
        conv5 = self._CommonConv(512, "conv5_2")(conv5)
        drop5 = layers.Dropout(0.5)(conv5)

        up6 = self._CommonDeconv(256, "up6", drop5)
        merge6 = self._CommonMerge(drop4, up6)
        conv6 = self._CommonConv(256, "conv6_1")(merge6)
        conv6 = self._CommonConv(256, "conv6_2")(conv6)

        up7 = self._CommonDeconv(128, "up7", conv6)
        merge7 = self._CommonMerge(conv3, up7)
        conv7 = self._CommonConv(128, "conv7_1")(merge7)
        conv7 = self._CommonConv(128, "conv7_2")(conv7)

        up8 = self._CommonDeconv(64, "up8", conv7)
        merge8 = self._CommonMerge(conv2, up8)
        conv8 = self._CommonConv(64, "conv8_1")(merge8)
        conv8 = self._CommonConv(64, "conv8_2")(conv8)

        up9 = self._CommonDeconv(32, "up9", conv8)
        merge9 = self._CommonMerge(conv1, up9)
        conv9 = self._CommonConv(32, "conv9_1")(merge9)
        conv9 = self._CommonConv(32, "conv9_2")(conv9)

        conv10 = layers.Conv2D(filters=num_class, kernel_size=1, activation="sigmoid")(conv9)

        self.model = models.Model(inputs = inputs, outputs = conv10)
        print("UNet constructed")

    def compile(self):
        self.model.compile(optimizer = Adam(lr = 1e-4), loss = "binary_crossentropy", metrics = ["accuracy"])
        print("UNet compiled")
