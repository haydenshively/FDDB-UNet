import numpy as np
import cv2
import time

predictions = np.load("Results/predictions.npy")
truths = np.load("Results/truths.npy")


for prediction, truth in zip(predictions, truths):
    cv2.imshow("prediction", prediction)
    cv2.imshow("truth", truth)

    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break

    time.sleep(.5)

cv2.destroyAllWindows()
