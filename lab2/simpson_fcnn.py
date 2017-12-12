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
    data = mx.sym.flatten(data = data)
    # 1st full layer
    fc = mx.sym.FullyConnected(data = data, num_hidden = 300)
    tanh = mx.sym.Activation(data = fc, act_type = "tanh") #relu, sigmoid, tanh, softrelu
    # 2d full layer
    #fc2 = mx.sym.FullyConnected(data = tanh, num_hidden = 100)
    #tanh2 = mx.sym.Activation(data = fc2, act_type = "tanh")
    # 3 full layer
    #fc3 = mx.sym.FullyConnected(data = tanh2, num_hidden = 18)
    fc3 = mx.sym.FullyConnected(data = tanh, num_hidden = 18)
    fcnn = mx.sym.SoftmaxOutput(data = fc3, name = 'softmax')

    # Train
    fcnn_model = mx.mod.Module(symbol = fcnn, context = mx.cpu())
    fcnn_model.fit(train_iter,  # train data
                   eval_data = val_iter,  # validation data
                   optimizer = 'sgd',  # use SGD to train
                   optimizer_params = {'learning_rate':0.01},  # use fixed learning rate
                   eval_metric = 'acc',  # report accuracy during training
                   batch_end_callback = mx.callback.Speedometer(batch_size, 100000), # output progress for each 100 data batches
                   num_epoch = 100)  # train for at most 100 dataset passes

    # Test
    test_iter = mx.io.NDArrayIter(X_test,
                                  y_test,
                                  batch_size)
    acc = mx.metric.Accuracy()
    fcnn_model.score(test_iter, acc)
    print(acc)

def main():
    if not os.path.isfile('X_train.npy'):
        X,y = io.load_pictures('characters\\', True)
        X_train, X_test, y_train, y_test = io.split_data(X, y, 85)
        io.write_data_to_file(X_train, X_test, y_train, y_test)
    else:
        X_train, X_test, y_train, y_test = io.load_data_from_file()

    #Train and test
    run_train_and_test(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()