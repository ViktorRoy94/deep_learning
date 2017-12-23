import os
import gzip
import sys
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
    args = sys.argv
    if '-h' in args:
        print(" -h help")
        print(" -n numer train images")
        print(" -t numer test images")
        print(" -s numer hidden layer nodes")
        print(" -l learn rate ")

    num_train_images = 1000
    num_test_images = 100
    image_size = 28 * 28

    max_epochs = 15
    learn_rate = 0.005
    cross_error = 0.005

    num_input = image_size
    num_hidden = 100
    num_output = 10

    if '-n' in args:
        idx = args.index('-n')
        num_train_images = int(args[idx+1])
    if '-t' in args:
        idx = args.index('-t')
        num_test_images = int(args[idx+1])
    if '-s' in args:
        idx = args.index('-s')
        num_hidden = int(args[idx+1])
    if '-l' in args:
        idx = args.index('-l')
        learn_rate = float(args[idx+1])

    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    t_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    t_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    X_train = X_train[0:num_train_images]
    t_train = t_train[0:num_train_images]

    X_test = X_test[0:num_test_images]
    t_test = t_test[0:num_test_images]

    nw = nn.NeuralNetwork(num_input, num_hidden, num_output)
    
    nw.train(X_train, t_train, max_epochs, learn_rate, cross_error)

    print("Train: ", nw.accuracy(X_train, t_train), "% Test:", nw.accuracy(X_test, t_test), "%")

if __name__ == '__main__':
    main()