from keras import models
import numpy as np

def load_data():
    inputs = np.load("finalImages.npy")[2800:]
    outputs = np.load("finalMasks.npy")[2800:]
    testing_input = np.expand_dims(inputs, axis = -1)
    desired_output = np.expand_dims(outputs, axis = -1)
    return testing_input, desired_output

unet = models.load_model("unet taught.h5")

inputs, outputs = load_data()

predicts = unet.predict(inputs, verbose=1)

np.save("predictions.npy", predicts)
np.save("comparisons.npy", outputs)
