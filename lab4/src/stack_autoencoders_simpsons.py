import mxnet as mx
import logging
import os.path
import numpy as np
import time		
import glob	
import cv2	  
import read_write_data as io
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

def timer(str_):
    def deco(f):
        def wrapper(*args, **kwargs):
            t1 = time.time()
            res = f(*args, **kwargs)
            t2 = time.time()
            print(str_ + '%f' % (t2-t1) + 'c')
            return res
        return wrapper
    return deco

def metric_MSE_for_autoencoder1(label, pred):
	label = label.reshape(-1, 28*28*3)
	return np.mean((label - pred) ** 2)
eval_metric1 = mx.metric.create(metric_MSE_for_autoencoder1)

def metric_MSE_for_autoencoder2(label, pred):
	label = label.reshape(-1, 400)
	return np.mean((label - pred) ** 2)
eval_metric2 = mx.metric.create(metric_MSE_for_autoencoder2)

def my_predict(img_array, mod):
    mod.forward(Batch([img_array]))
    features = mod.get_outputs()[0].asnumpy()
    return features

@timer("time train autoencoder 1 = ")
def autoencoder_one(X_train, batch_size):
	# Construct autoencoder 1
	train_iter1 = mx.io.NDArrayIter(X_train,
									X_train,
									batch_size, shuffle = True)						
	data = mx.sym.var('data')
	data = mx.sym.flatten(data = data)
	fc1 = mx.symbol.FullyConnected(data = data, num_hidden = 400, name = 'full_1')
	encoder = mx.sym.Activation(data = fc1, act_type = 'relu', name = 'relu')

	decoder = mx.symbol.FullyConnected(data = encoder, num_hidden = 3*28*28, name = 'decoder')
	out_autoencoder = mx.sym.LinearRegressionOutput(data = decoder, name = 'softmax')

	# Train autoencoder 1
	autoencoder1_model = mx.mod.Module(symbol = out_autoencoder, context = mx.cpu())
	autoencoder1_model.fit(train_iter1,  # train data
							optimizer = 'sgd',  # use SGD to train
							optimizer_params = {'learning_rate':0.01},  # use fixed learning rate
							eval_metric = eval_metric1,  # report mse during training
							num_epoch = 1) 

							
	return autoencoder1_model		

def forward_autoencoder_one(autoencoder1_model, X_train):
	
	arg_params, aux_params = autoencoder1_model.get_params()
	output = autoencoder1_model.symbol.get_internals()['relu_output']

	fe_mod = mx.mod.Module(symbol = output, context = mx.cpu(), label_names = None)
	fe_mod.bind(for_training = False, data_shapes=[('data', (1,3,28,28))])
	fe_mod.set_params(arg_params, aux_params)

	X = []
	train_iter = mx.io.NDArrayIter(X_train)	
	for batch in train_iter:
		img_array = batch.data[0]
		features = my_predict(img_array, fe_mod)
		X.append(features)
	
	X = np.array(X)			
	input_for_autoencoder2 = np.reshape(X, (X.shape[0], 400))
	
	return input_for_autoencoder2

@timer("time train autoencoder 2 = ")
def autoencoder_two(input_for_autoencoder2, batch_size):
	# Construct autoencoder 2
	train_iter2 = mx.io.NDArrayIter(input_for_autoencoder2,
									input_for_autoencoder2,
									batch_size, shuffle = True)						

	data = mx.sym.var('data')
	data = mx.sym.flatten(data = data)
	fc1 = mx.symbol.FullyConnected(data = data, num_hidden = 400, name = 'full_2')
	encoder = mx.sym.Activation(data = fc1, act_type = 'relu')

	decoder = mx.symbol.FullyConnected(data = encoder, num_hidden = 400, name = 'decoder')
	out_autoencoder = mx.sym.LinearRegressionOutput(data = decoder, name = 'softmax')

	# Train autoencoder 2
	autoencoder2_model = mx.mod.Module(symbol = out_autoencoder, context = mx.cpu())
	autoencoder2_model.fit(train_iter2,  # train data
							optimizer = 'sgd',  # use SGD to train
							optimizer_params = {'learning_rate':0.01},  # use fixed learning rate
							eval_metric = eval_metric2,  # report mse during training
							num_epoch = 1)
	return autoencoder2_model
	
@timer("time train fcnn = ")
def run_train_and_test_fcnn(X_train, X_test, y_train, y_test, batch_size, autoencoder1, autoencoder2):

	arg_params, aux_params = autoencoder1.get_params()
	new_args = dict({k:arg_params[k] for k in arg_params if 'full_1' in k})
	arg_params, aux_params = autoencoder2.get_params()
	new_args.update(dict({k:arg_params[k] for k in arg_params if 'full_2' in k}))
	
	train_iter = mx.io.NDArrayIter(X_train,
									y_train,
									batch_size, shuffle = True)						
	test_iter = mx.io.NDArrayIter(X_test,
								  y_test,
								  batch_size)
							 
	data = mx.sym.var('data')
	data = mx.sym.flatten(data = data)
	fc1 = mx.symbol.FullyConnected(data = data, num_hidden = 400, name = 'full_1')
	relu1 = mx.sym.Activation(data = fc1, act_type = 'relu')
	fc2 = mx.symbol.FullyConnected(data = relu1, num_hidden = 400, name = 'full_2')
	relu2 = mx.sym.Activation(data = fc2, act_type = 'relu')
	fc3 = mx.symbol.FullyConnected(data = relu2, num_hidden = 18)
	out = mx.sym.SoftmaxOutput(data = fc2, name = 'softmax')

	fcnn_model = mx.mod.Module(symbol = out, context = mx.cpu())
	fcnn_model.fit(train_iter,  # train data
               eval_data = test_iter,  # validation data
               optimizer = 'sgd',  # use SGD to train
			   arg_params = new_args,
			   allow_missing = True,
			   initializer = mx.init.Xavier(rnd_type = 'gaussian', factor_type = "in", magnitude = 2),
               optimizer_params = {'learning_rate':0.01},  # use fixed learning rate
               eval_metric = 'acc',  # report cross-entropy during training
               num_epoch = 1)  # train for at most 10 dataset passes
			 
	acc = mx.metric.Accuracy()
	fcnn_model.score(test_iter, acc)
	print(acc)

def main():
	if not os.path.isfile('X_train.npy'):
		X,y = io.load_pictures('../../data/characters/', True)
		X_train, X_test, y_train, y_test = io.split_data(X, y, 85)
		io.write_data_to_file(X_train, X_test, y_train, y_test)
	else:
		X_train, X_test, y_train, y_test = io.load_data_from_file()
	
	logging.getLogger().setLevel(logging.DEBUG) 
	batch_size = 10
	
	print "\n start train autoencoder 1..."
	autoencoder1 = autoencoder_one(X_train, batch_size)
	input_for_autoencoder2 = forward_autoencoder_one(autoencoder1, X_train)
	print "\n start train autoencoder 2..."
	autoencoder2 = autoencoder_two(input_for_autoencoder2, batch_size)
	print " \n start train fcnn ..."
	run_train_and_test_fcnn(X_train, X_test, y_train, y_test, batch_size, autoencoder1, autoencoder2)

if __name__ == "__main__":
	main()	