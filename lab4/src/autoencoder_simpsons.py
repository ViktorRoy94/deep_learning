import mxnet as mx
import logging
import os.path
import time
import numpy as np
import read_write_data as io  

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


def custom_metric(label, pred):
	label = label.reshape(-1, 28*28*3)
	return np.mean((label - pred) ** 2)

@timer("time train autoencoder = ")
def autoencoder(X_train, X_test):
    logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

    hidden_neurons = 3*28*28

    batch_size = 10
    train_iter = mx.io.NDArrayIter(X_train,
                                   X_train,
                                   batch_size, shuffle = True)            
    val_iter = mx.io.NDArrayIter(X_test,
                                 X_test,
                                 batch_size)

    eval_metric = mx.metric.create(custom_metric)
    data = mx.sym.var('data')
    data = mx.sym.flatten(data = data)
    fc1 = mx.symbol.FullyConnected(data = data, num_hidden = 400, name = 'full')
    encoder = mx.sym.Activation(data = fc1, act_type = 'relu')

    decoder = mx.symbol.FullyConnected(data = encoder, num_hidden = hidden_neurons, name = 'decoder')
    out_autoencoder = mx.sym.LinearRegressionOutput(data = decoder, name = 'softmax')

    # Train
    autoencoder_model = mx.mod.Module(symbol = out_autoencoder, context = mx.cpu())
    autoencoder_model.fit(train_iter,  # train data
                     eval_data = val_iter,  # validation data
                     optimizer = 'sgd',  # use SGD to train
                     optimizer_params = {'learning_rate':0.01},  # use fixed learning rate
                     eval_metric = eval_metric,  # report mse during training
                     batch_end_callback = mx.callback.Speedometer(1000, 1000), # output progress
                     num_epoch = 10) 

    arg_params, aux_params = autoencoder_model.get_params()

    new_args = dict({k:arg_params[k] for k in arg_params if 'full' in k})
    return new_args
    
@timer("time train fcnn = ")
def run_train_and_test(X_train, X_test, y_train, y_test, args):
    batch_size = 10
    train_iter = mx.io.NDArrayIter(X_train,
                                    y_train,
                                    batch_size, shuffle = True)						
    test_iter = mx.io.NDArrayIter(X_test,
                                   y_test,
    							   batch_size)
    							 
    data = mx.sym.var('data')
    data = mx.sym.flatten(data = data)
    fc1 = mx.symbol.FullyConnected(data = data, num_hidden = 400, name = 'full')
    relu = mx.sym.Activation(data = fc1, act_type = 'relu')
    fc2 = mx.symbol.FullyConnected(data = relu, num_hidden = 18)
    out = mx.sym.SoftmaxOutput(data = fc2, name = 'softmax')

    fcnn_model = mx.mod.Module(symbol = out, context = mx.cpu())
    fcnn_model.fit(train_iter,  # train data
                   eval_data = test_iter,  # validation data
                   optimizer = 'sgd',  # use SGD to train
    			   arg_params = args,
    			   allow_missing = True,
    			   initializer = mx.init.Xavier(rnd_type = 'gaussian', factor_type = "in", magnitude = 2),
                   optimizer_params = {'learning_rate':0.01},  # use fixed learning rate
                   eval_metric = 'acc',  # report cross-entropy during training
                   batch_end_callback = mx.callback.Speedometer(1000, 1000), # output progress
                   num_epoch = 50)  # train for at most 10 dataset passes
    			 
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

    args = autoencoder(X_train, X_test)
    run_train_and_test(X_train, X_test, y_train, y_test, args)


if __name__ == "__main__":
  main()