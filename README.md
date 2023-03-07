# Classification-of-VOC-2007-dataset-using-Convolution-Neural-Network
Implementation of a convolution neural network to classify VOC 2007 dataset

I had initially used the simple classifier and noticed very poor map score and the loss was also large. Hence, I had taken inspiration from VGG16 network and build a convent consisting of three layers followed by two fully connected layer. Each layer consisted of two convolutional 2d layer each with applied rely and batch normalisation followed by a max pooling layer and a dropout layer.
