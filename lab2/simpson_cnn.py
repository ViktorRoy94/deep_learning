import mxnet as mx
import logging

import cv2
import glob
import numpy as np

map_characters = {0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon',
        2: 'bart_simpson', 3: 'charles_montgomery_burns', 4: 'chief_wiggum',
        5: 'comic_book_guy', 6: 'edna_krabappel', 7: 'homer_simpson',
        8: 'kent_brockman', 9: 'krusty_the_clown', 10: 'lisa_simpson',
        11: 'marge_simpson', 12: 'milhouse_van_houten', 13: 'moe_szyslak',
        14: 'ned_flanders', 15: 'nelson_muntz', 16: 'principal_skinner',
        17: 'sideshow_bob'}

pic_size = 28

def load_pictures(path, BGR):
	print("Load data from pictures ...")
    pics = []
    labels = []
    for class_simpson, char in map_characters.items():
        pictures = [k for k in glob.glob(path + '%s/*' % char)]
        nb_pic = len(pictures)
        print("    ", char, "  ", nb_pic)
        for pic in np.random.choice(pictures, nb_pic):
            a = cv2.imread(pic)
            if BGR:
                a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
            a = cv2.resize(a, (pic_size,pic_size)).astype('float32') / 255
            pics.append(a)
            labels.append(class_simpson)
    print("Done")
    return np.array(pics), np.array(labels)

def split_data(X, y, percent):
    n = len(X)
    rand_indicies = np.arange(n)
    np.random.shuffle(rand_indicies)
    X = X[rand_indicies]
    y = y[rand_indicies]
    index = int(n * percent / 100)
    return X[:index], X[index:], y[:index], y[index:]

def write_data_to_file(X_train, X_test, y_train, y_test):
    print("Write data to files ...")
    np.save("X_train", X_train)
    np.save("X_test", X_test)
    np.save("y_train", y_train)
    np.save("y_test", y_test)
    print("Done")

def load_data_from_file():
    print("Load data from files ...")
    X_train = np.load("X_train.npy")
    X_test = np.load("X_test.npy")
    y_train = np.load("y_train.npy")
    y_test = np.load("y_test.npy")
    print("Done")
    return X_train, X_test, y_train, y_test

def run_train_and_test(X_train, X_test, y_train, y_test):
	print("X_train.shape =", X_train.shape)
	print("y_train.shape =", y_train.shape)
	print("X_test.shape =", X_test.shape)
	print("y_test.shape =", y_test.shape)

	logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

	# Initialize deep model
	batch_size = 10
	train_iter = mx.io.NDArrayIter(X_train, 
	                               y_train,
	                               batch_size, shuffle = True)
	val_iter = mx.io.NDArrayIter(X_test,
	                             y_test,
	                             batch_size)
	# Symbol model
	data = mx.sym.var('data')
	# 1st convolution layer
	conv_1 = mx.sym.Convolution(data = data, kernel = [2, 2], num_filter = 20)
	tanh_1 = mx.sym.Activation(data = conv_1, act_type = "tanh")
	pool_1 = mx.sym.Pooling(data = tanh_1, pool_type = "max", kernel = [2, 2], stride = [1, 1])
	# 1st full layer
	flatten = mx.sym.Flatten(data = pool_1)
	fc_1 = mx.sym.FullyConnected(data = flatten, num_hidden = 300)
	tanh_2 = mx.sym.Activation(data = fc_1, act_type = "tanh")
	# 2d full layer
	fc_2 = mx.sym.FullyConnected(data = tanh_2, num_hidden = 18)

	cnn = mx.sym.SoftmaxOutput(data = fc_2, name = 'softmax')

	# Train
	cnn_model = mx.mod.Module(symbol = cnn, context = mx.cpu())
	cnn_model.fit(train_iter,  # train data
	                 eval_data = val_iter,  # validation data
	                 optimizer = 'sgd',  # use SGD to train
	                 optimizer_params = {'learning_rate':0.1},  # use fixed learning rate
	                 eval_metric = 'acc',  # report accuracy during training
	                 batch_end_callback = mx.callback.Speedometer(batch_size, 100), # output progress for each 100 data batches
	                 num_epoch = 100)  # train for at most 100 dataset passes

	# Test
	test_iter = mx.io.NDArrayIter(X_test,
	                              y_test,
	                              batch_size)
	acc = mx.metric.Accuracy()
	cnn_model.score(test_iter, acc)
	print(acc)

X,y = load_pictures('characters/', True)
X_train, X_test, y_train, y_test = split_data(X, y, 85)
write_data_to_file(X_train, X_test, y_train, y_test)

#Load data from files
X_train, X_test, y_train, y_test = load_data_from_file()

#Train and test
run_train_and_test(X_train, X_test, y_train, y_test)

