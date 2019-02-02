"""
TRAIN
"""

if __name__ == '__main__':
    from unet import UNet
    from util import load_npy_data

    raw, highlighted = load_npy_data(amount = 220)

    unet = UNet((256, 256, 3), 1)
    unet.model.fit(raw, highlighted,
                   batch_size = 4,
                   epochs = 1,
                   verbose = 1,
                   validation_split = 0.1,
                   shuffle = True)

    unet.model.save("models/unet.h5")
