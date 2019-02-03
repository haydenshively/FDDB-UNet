# FDDB UNet

This project represents my first foray into the world of neural networks. Planning to incorporate this code into my Aslan project, I had hoped that the UNet architecture would be able to highlight image regions that contained faces. However, FDDB contained many faces located in the center of the image, and the network learned to highlight just that area. Obviously this wasn't very useful, so I abandoned this project and began work on FDDB-MobileNet. If I ever come back to it, the first thing I'll do is implement dataset augmentation, which may solve the center-highlighting issue.  
  
**FDDBToSamples.py**:  
converts the raw FDDB files into more convenient numpy files  
  
**unet.py**:  
defines a UNet architecture in terms of Keras' "Sequential Model" API  
  
**train.py**:  
trains a model using the numpy files and UNet architecture from the code above  
  
**predict.py**:  
runs the trained model on a portion of the dataset and saves results to a file  
  
**see_results.py**:  
displays the saved results  
  
## Prerequisites
Keras, OpenCV, numpy  
Dataset: http://vis-www.cs.umass.edu/fddb/
