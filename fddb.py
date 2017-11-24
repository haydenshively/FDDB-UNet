import numpy as np
import cv2
import os


class Directory(object):
    def __init__(self, path_string):
        self.path_string = path_string
        self.directory = os.fsencode(path_string)
    def ls(self):
        return os.listdir(self.directory)
    def pathTo(self, file):
        return self.path_string + os.fsdecode(file)


info_dir = Directory("FDDB_Labels/")
img_dir_str = "FDDB_Images/"

"""-------------------------------"""
paths = []
faces = []

group = []
for doc in info_dir.ls():
    with open(info_dir.pathTo(doc)) as text:
        doc = text.readlines()

    for line in doc:
        if "/" in line:
            if len(group) is not 0:
                faces.append(group)
                group = []
            paths.append(img_dir_str + line[:-1] + ".jpg")
        elif "  " in line:
            group.append(line.split(" ")[:5])

faces.append(group)# have to add the final group since if "/" doesn't run again
"""-------------------------------"""

shrink_iters = 0
divisor = 2**shrink_iters

finalImages = np.zeros((len(paths), 256, 256), dtype = "float32")
finalMasks = np.zeros((len(paths), 256, 256), dtype = "float32")

count = 0
for path, face in zip(paths, faces):
    img = cv2.imread(path)

    for i in range(shrink_iters):
        img = cv2.pyrDown(img)

    mask = np.zeros_like(img)

    for individual in face:
        maj_axis, min_axis, ang, x, y = individual
        x = int(float(x))
        y = int(float(y))
        maj_axis = int(float(maj_axis))
        min_axis = int(float(min_axis))
        ang = int(float(ang)*180/3.14)
        cv2.ellipse(mask, (x//divisor, y//divisor), (maj_axis//divisor, min_axis//divisor), ang, 0, 360, 255, -1)

    try:
        finalImages[count] = cv2.cvtColor(img[:256, :256], cv2.COLOR_BGR2GRAY).astype("float32")/255.
        finalMasks[count] = mask[:256, :256, 0].astype("float32")/255.
        count += 1
    except ValueError:
        continue


np.save("finalImages.npy", finalImages)
np.save("finalMasks.npy", finalMasks)
