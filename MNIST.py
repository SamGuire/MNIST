import numpy as np
from keras.datasets import mnist
from network import Network
import matplotlib.pyplot as plt

def vectorized_result(j) :
    result = np.zeros((10,1))
    result[j] = 1.0
    return result

(train_images, train_labels),(test_images,test_labels) = mnist.load_data()



train_images = train_images.reshape((60000,784))
train_images = train_images.astype('float32')/255.0
train_labels = [vectorized_result(j) for j in train_labels]

train_data = list(zip(train_images,train_labels))

test_images = test_images.reshape((10000,784))
test_images = test_images.astype('float32')/255.0


test_labels = [vectorized_result(j) for j in test_labels]


test_data = list(zip(test_images,test_labels))

# The hidden layer (value of index 1 of network array) can be manipulated to increase accuracy at the cost of runtime
network = Network([784,10,10])
network.stochastic_gradient_descent(10,128,1,train_data,test_data) 
