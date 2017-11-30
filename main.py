import os
import gzip
import numpy as np

from urllib.request import urlretrieve

import NeuralNetwork as nn

def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
    print("Downloading %s" % filename)
    urlretrieve(source + filename, filename)
        
def load_mnist_images(filename):
    if not os.path.exists(filename):
        download(filename)
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 784)
    return data / np.float32(256)

def load_mnist_labels(filename):
    if not os.path.exists(filename):
        download(filename)
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    n = len(data)
    no = max(data)
    labels = np.zeros(shape=[n,no+1], dtype=np.float32)
    for i in range(n):
        vec = np.zeros(shape=[no+1], dtype=np.float32)
        vec[data[i]] = 1
        labels[i] = vec
    return labels

def main():
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    num_train_images = 100
    num_test_images = 10000
    image_size = 28 * 28

    max_epochs = 20
    learn_rate = 0.008
    cross_error = 0.005

    num_input = image_size
    num_hidden = 200
    num_output = 10

    train_Data = np.zeros(shape=[num_train_images,num_input+num_output], dtype=np.float32)
    test_Data = np.zeros(shape=[num_test_images,num_input+num_output], dtype=np.float32)

    for i in range(num_train_images):
        train_Data[i] = np.append(X_train[i], y_train[i])


    for i in range(num_test_images):
        test_Data[i] = np.append(X_test[i], y_test[i])

    print(train_Data.shape)
    print(test_Data.shape)

    nw = nn.NeuralNetwork(num_input, num_hidden, num_output)
    
    nw.train(train_Data, max_epochs, learn_rate, cross_error)

    print("Train: ", nw.accuracy(train_Data), " Test:", nw.accuracy(test_Data))

if __name__ == '__main__':
    main()