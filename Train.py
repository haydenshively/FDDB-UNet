import numpy as np
import cv2

from UNet import UNet
from keras.callbacks import ModelCheckpoint

def load_data():
    inputs = np.load("finalImages.npy")
    outputs = np.load("finalMasks.npy")
    training_input = inputs[:2000]
    training_input = np.expand_dims(training_input, axis = -1)
    desired_output = outputs[:2000]
    desired_output = np.expand_dims(desired_output, axis = -1)
    testing_input = inputs[2000:]
    testing_input = np.expand_dims(testing_input, axis = -1)
    return training_input, desired_output, testing_input



training_input, desired_output, testing_input = load_data()

unet = UNet((256, 256, 1), 1)

model_checkpoint = ModelCheckpoint("unet.hdf5", monitor = "loss", verbose = 1, save_best_only = True)
unet.model.fit(training_input, desired_output, batch_size=10, epochs=1, verbose=1, validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])

unet.model.save("unet taught.h5")
