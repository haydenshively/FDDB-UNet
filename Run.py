from keras import models
import numpy as np

def data(amount = None):
    images = np.load("Dataset Silo/images.npy")
    highlights = np.load("Dataset Silo/highlights.npy")
    if amount is not None:
        images = images[:amount]
        highlights = highlights[:amount]
    return images, highlights

unet = models.load_model("Model/unet.h5")

images, highlights = data(50)

predictions = unet.predict(images, verbose = 1)

np.save("Results/predictions.npy", predictions)
np.save("Results/truths.npy", highlights)
