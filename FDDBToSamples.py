import numpy as np
import cv2
import os


class Directory(object):
    def __init__(self, pathString):
        self.pathString = pathString
        self.directory = os.fsencode(pathString)
    def ls(self):
        return os.listdir(self.directory)
    def pathTo(self, file):
        return self.pathString + os.fsdecode(file)


annotationsDirectory = Directory("FDDB_Labels/")
imagePathString = "FDDB_Images/"

"""This section organizes data from the annotations directory.
---paths to images are placed in allPaths
---face descriptors are placed in allFaces, and grouped if there is more than one face in an image
   Everything is still just strings."""

allPaths = []
allFaces = []

imageFaces = []
for doc in annotationsDirectory.ls():
    with open(annotationsDirectory.pathTo(doc)) as text:
        doc = text.readlines()

    for line in doc:
        if "/" in line:
            if len(imageFaces) is not 0:
                allFaces.append(imageFaces)
                imageFaces = []
            allPaths.append(imagePathString + line[:-1] + ".jpg")
        elif "  " in line:
            imageFaces.append(line.split(" ")[:5])

allFaces.append(imageFaces)# have to add the final imageFaces since if "/" doesn't run again

"""This section processes the strings in allPaths and allFaces with OpenCV.
   Numpy then saves 2 files, 1 with raw images and the other with face regions highlighted.
   Size is 256x256."""

shrinkIterations = 0
divisor = 2**shrinkIterations

images = np.zeros((len(allPaths), 256, 256, 3), dtype = "float32")
highlights = np.zeros((len(allPaths), 256, 256), dtype = "float32")

count = 0
for path, imageFaces in zip(allPaths, allFaces):
    image = cv2.imread(path)# obtain actual image at specified path

    for i in range(shrinkIterations):# shrink the image
        image = cv2.pyrDown(image)

    highlight = np.zeros_like(image)# prepare for highlighting

    for face in imageFaces:
        face = [float(i) for i in face]# convert from strings to floats
        face[2] = face[2]*180/3.14# convert from radians to degrees
        maj_axis, min_axis, angle, x, y = [int(i) for i in face]# get ints to make OpenCV happy

        cv2.ellipse(highlight, (x//divisor, y//divisor), (maj_axis//divisor, min_axis//divisor), angle, 0, 360, (255, 255, 255), -1)

    try:
        images[count] = image[:256, :256].astype("float32")/255.
        highlights[count] = highlight[:256, :256, 0].astype("float32")/255.
        count += 1
    except ValueError:
        continue


np.save("Datasets/raw.npy", images[:count])
np.save("Datasets/highlighted.npy", highlights[:count])
