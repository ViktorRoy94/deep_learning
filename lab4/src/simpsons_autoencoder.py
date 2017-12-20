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

def run_train_and_test(X_train, X_test, y_train, y_test):
    print("X_train.shape =", X_train.shape)
    print("y_train.shape =", y_train.shape)
    print("X_test.shape =", X_test.shape)
    print("y_test.shape =", y_test.shape)

    logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

    # Initialize deep model
    size = X_train.shape[0]

    batch_size = 100
    train_iter = mx.io.NDArrayIter(X_train,
                                   y_train,
                                   batch_size, shuffle = True)
    val_iter = mx.io.NDArrayIter(X_test,
                                 y_test,
                                 batch_size)
    							 
    data = mx.sym.var('data')
    encoder = mx.symbol.FullyConnected(data=data, num_hidden=5, name='encoder')
    en_act = mx.sym.Activation(data=encoder, act_type='sigmoid', name='encoder_activation')

    decoder = mx.symbol.FullyConnected(data=en_act, num_hidden=size, name='decoder')
    out = mx.sym.SoftmaxOutput(data=decoder, name='softmax')

    # Train
    autoencoder_model = mx.mod.Module(symbol = out, context = mx.cpu())
    autoencoder_model.fit(train_iter,  # train data
                     eval_data = val_iter,  # validation data
                     optimizer = 'sgd',  # use SGD to train
                     optimizer_params = {'learning_rate':0.01},  # use fixed learning rate
                     eval_metric = 'ce',  # report cross-entropy during training
                     batch_end_callback = mx.callback.Speedometer(batch_size, 1000), # output progress
                     num_epoch = 100)  # train for at most 10 dataset passes

    # Test
    test_iter = mx.io.NDArrayIter(X_test,
                                  y_test,
                                  batch_size)
    							  
    ce = mx.metric.CrossEntropy()
    autoencoder_model.score(test_iter, ce)
    print(ce)




def main():
    if not os.path.isfile('X_train.npy'):
        X,y = io.load_pictures('../../characters/', True)
        X_train, X_test, y_train, y_test = io.split_data(X, y, 85)
        io.write_data_to_file(X_train, X_test, y_train, y_test)
    else:
        X_train, X_test, y_train, y_test = io.load_data_from_file()

    #Train and test
    run_train_and_test(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
  main()