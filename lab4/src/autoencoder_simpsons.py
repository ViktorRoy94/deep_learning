
import mxnet as mx
import logging
import numpy as np
import time				  

print("Load data from files ...")
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")
print("Done")

def custom_metric(label, pred):
	label = label.reshape(-1, 28*28*3)
	return np.mean((label - pred) ** 2)
eval_metric = mx.metric.create(custom_metric)

t1 = time.time()

logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

hidden_neurons = 3*28*28

batch_size = 10
train_iter = mx.io.NDArrayIter(X_train,
                               X_train,
                               batch_size, shuffle = True)						
val_iter = mx.io.NDArrayIter(X_test,
                             X_test,
							 batch_size)

data = mx.sym.var('data')
data = mx.sym.flatten(data = data)
fc1 = mx.symbol.FullyConnected(data = data, num_hidden = 400, name = 'full')
encoder = mx.sym.Activation(data = fc1, act_type = 'relu')

decoder = mx.symbol.FullyConnected(data = encoder, num_hidden = hidden_neurons, name = 'decoder')
out_autoencoder = mx.sym.LinearRegressionOutput(data = decoder, name = 'softmax')

print "\n start train autoencoder ..."

# Train
autoencoder_model = mx.mod.Module(symbol = out_autoencoder, context = mx.cpu())
autoencoder_model.fit(train_iter,  # train data
                 eval_data = val_iter,  # validation data
                 optimizer = 'sgd',  # use SGD to train
                 optimizer_params = {'learning_rate':0.01},  # use fixed learning rate
                 eval_metric = eval_metric,  # report mse during training
                 #batch_end_callback = mx.callback.Speedometer(1000, 1000), # output progress
                 num_epoch = 100) 

				 
t2 = time.time()
print 'time train autoencoder = ', t2-t1, 'c'
				 
arg_params, aux_params = autoencoder_model.get_params()

new_args = dict({k:arg_params[k] for k in arg_params if 'full' in k})

print " \n start train fcnn ..."

t1 = time.time()

train_iter2 = mx.io.NDArrayIter(X_train,
                                y_train,
                                batch_size, shuffle = True)						
test_iter2 = mx.io.NDArrayIter(X_test,
                               y_test,
							   batch_size)
							 
data = mx.sym.var('data')
data = mx.sym.flatten(data = data)
fc1 = mx.symbol.FullyConnected(data = data, num_hidden = 400, name = 'full')
relu = mx.sym.Activation(data = fc1, act_type = 'relu')
fc2 = mx.symbol.FullyConnected(data = relu, num_hidden = 18)
out = mx.sym.SoftmaxOutput(data = fc2, name = 'softmax')

fcnn_model = mx.mod.Module(symbol = out, context = mx.cpu())
fcnn_model.fit(train_iter2,  # train data
               eval_data = test_iter2,  # validation data
               optimizer = 'sgd',  # use SGD to train
			   arg_params = new_args,
			   allow_missing = True,
			   initializer = mx.init.Xavier(rnd_type = 'gaussian', factor_type = "in", magnitude = 2),
               optimizer_params = {'learning_rate':0.01},  # use fixed learning rate
               eval_metric = 'acc',  # report cross-entropy during training
               #batch_end_callback = mx.callback.Speedometer(1000, 1000), # output progress
               num_epoch = 50)  # train for at most 10 dataset passes
			 
acc = mx.metric.Accuracy()
fcnn_model.score(test_iter2, acc)
print(acc)

t2 = time.time()
print 'time train fcnn = ', t2-t1, 'c'
