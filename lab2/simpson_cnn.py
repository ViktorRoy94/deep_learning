import mxnet as mx
import logging
import os.path
import time
import read_write_data as io

def timer(f):
	def tmp(*args, **kwargs):
		t1 = time.time()
		res = f(*args, **kwargs)
		t2 = time.time()
		print("Function time = %f" % (t2-t1))
		return res
	return tmp

@timer
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
	                 num_epoch = 10)  # train for at most 100 dataset passes

	# Test
	test_iter = mx.io.NDArrayIter(X_test,
	                              y_test,
	                              batch_size)
	acc = mx.metric.Accuracy()
	cnn_model.score(test_iter, acc)
	print(acc)

def main():
    if not os.path.isfile('X_train.npy'):
        X,y = io.load_pictures('characters/', True)
        X_train, X_test, y_train, y_test = io.split_data(X, y, 85)
        io.write_data_to_file(X_train, X_test, y_train, y_test)
    else:
        X_train, X_test, y_train, y_test = io.load_data_from_file()

    #Train and test
    run_train_and_test(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()