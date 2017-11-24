import numpy as np
import cv2

source = np.load("predictions.npy")
goal = np.load("comparisons.npy")

print(goal.shape)

image1 = source[3,:,:,0]*255
goal1 = goal[3,:,:,0]*255

while True:
    cv2.imshow("img", image1.astype("uint8"))
    cv2.imshow("act", goal1.astype("uint8"))
    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break

cv2.destroyAllWindows()
