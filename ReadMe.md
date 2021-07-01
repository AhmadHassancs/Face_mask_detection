#################################################################################################################################
########################## Real-Time Face Mask Detector with Python, OpenCV, Keras  #############################################
#################################################################################################################################

During pandemic COVID-19, WHO has made wearing masks compulsory to protect against this deadly virus.
In this tutorial we will develop a machine learning project – Realtime Face Mask Detector with Python.
We will build a real-time system to detect whether the person on the webcam is wearing a mask or not.
We will train the face mask detector model using Keras and OpenCV.
The dataset we are working on consists of 1376 images with 690 images containing images of people wearing
masks and 686 images with people without masks.

################################################################################
Download the dataset: https://data-flair.training/blogs/download-face-mask-data/
################################################################################

In this machine learning project for beginners, we will use Jupyter Notebook for the development.
We are going to build this project in two parts. In the first part, we will write a python script using
Keras to train face mask detector model.
In the second part, we test the results in a real-time webcam using OpenCV.

###############
Neural Network:
###############
          This convolution network consists of two pairs of Conv and MaxPool layers to extract features from the dataset.
          Which is then followed by a Flatten and Dropout layer to convert the data in 1D and ensure overfitting.
          And then two Dense layers for classification.
########
Summary:
########
          In this project, we have developed a deep learning model for face mask detection using Python, Keras, and OpenCV.
          We developed the face mask detector model for detecting whether person is wearing a mask or not.
          We have trained the model using Keras with network architecture. Training the model is the first part of this
          project and testing using webcam using OpenCV is the second part.
