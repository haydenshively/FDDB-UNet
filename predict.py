"""
PREDICT
Runs the trained model on a subset of the data and saves results to a file
"""

if __name__ == '__main__':
    from keras import models
    from numpy import save
    from util import load_npy_data

    raw, highlighted = load_npy_data(amount = 50)

    unet = models.load_model("models/unet.h5")
    predictions = unet.predict(raw, verbose = 1)

    save('Results/truths.npy', highlighted)
    save('Results/predictions.npy', predictions)
