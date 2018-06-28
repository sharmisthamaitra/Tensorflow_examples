# Tensorflow_examples
Repository to store Tensorflow code examples and Tensorflow projects

This directory contains Tensorflow code examples for the understanding of Machine Learning with TensorFlow.

This directory also contains the necessary modules for my personal project on Tensorflow:
Image classification with Convolutional neural network using Tensorflow. 
The source code for this project is Image_Classification_with_tensorflow.py. It employs Convolutional Neural Network architecture to classify color images in the CIFAR-10 dataset into 10 possible classes. 
The CIFAR-10 dataset contains 60,000 color images of animals and inanimate objects. Each image is 32x32 pixels. An image can belong to one of the following classes: 
          'airplane',
          'automobile',
          'bird',
          'cat',
          'deer',
          'dog',
          'frog',
          'horse',
          'ship',
          'truck'
Convolutional neural network (CNN) is a time tested approach for image classification problems in machine learning. 
CNNs work by exploring different parts of the input image with small filters. By varying the number and placement of 
convolutional and pooling layers, and their parameters like kernel size, strides, number of filters, it is possible to 
improve the performance of the model.

Following mechanism of data flow is employed in CNN . Input images -> Dropout(shutoff)some neurons -> convolution_layer_1 ->
convolution_layer_2 -> pooling_layer_3 -> convolution_layer_4 -> pooling_layer_5 -> pool5_flat ->
fully_connected_layer_1 -> fully_connected_layer_2
#Applying 2 conv layers before a pool to build up better representations of the data. Applying pooling early on has a disadvantage that it quickly loses all of the spatial information of the data.

#32 filters applied in convolution_layer_1, image size is 32x32. 64 filters applied in convolution_layer_2, image size reduced to 16x16. After applying pooling_layer_3 image size is 8x8 , number of filters is 64. 128 filters applied in 
convolution_layer_4, image size is 3x3. After applying pooling_layer_5 image size is 2x2 , number of filters is 128.
So progressively the size of the image is getting smaller but its depth is increasing, meaning more features about the image is being loaded into the model.  
