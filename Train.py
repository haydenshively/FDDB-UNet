import numpy as np
import cv2

from UNet import UNet

def data(amount = None):
    images = np.load("Dataset Silo/images.npy")
    highlights = np.load("Dataset Silo/highlights.npy")
    if amount is not None:
        images = images[:amount]
        highlights = highlights[:amount]
    return images, highlights


images, highlights = data(220)

unet = UNet((256, 256, 3), 1)

unet.model.fit(images, highlights, batch_size = 4, epochs = 1, verbose = 1, validation_split = 0.1, shuffle = True)

unet.model.save("Model/unet.h5")
