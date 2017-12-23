import mxnet as mx
import os, urllib
import read_write_data as io
import logging
import time

def timer(f):
    def tmp(*args, **kwargs):
        t1 = time.time()
        res = f(*args, **kwargs)
        t2 = time.time()
        print("Function time = %f" % (t2-t1))
        return res
    return tmp

def get_model(prefix, epoch):
    mx.test_utils.download(prefix+'-symbol.json')
    mx.test_utils.download(prefix+'-%04d.params' % (epoch,))

def get_iterators(X_train, y_train, X_test, y_test, batch_size):
    train_iter = mx.io.NDArrayIter(X_train,
                                   y_train,
                                   batch_size, shuffle = True)
    val_iter = mx.io.NDArrayIter(X_test,
                                 y_test,
                                 batch_size)
    return train_iter, val_iter

@timer
def fit(symbol, train, val, batch_size):
    mod = mx.mod.Module(symbol=symbol, context=mx.cpu())
    logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
    mod.fit(train, val,
            num_epoch=8,
            allow_missing=True,
            batch_end_callback = mx.callback.Speedometer(batch_size, 1000),
            kvstore='device',
            optimizer='sgd',
            optimizer_params={'learning_rate':0.01},
            initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
            eval_metric='acc')
    metric = mx.metric.Accuracy()
    return mod.score(val, metric)

def main():
    get_model('http://data.mxnet.io/models/imagenet/resnet/18-layers/resnet-18', 0)
    sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-18', 0)
    #print (sym.get_internals())
    
    batch_size = 10
    num_classes = 18
    
    if not os.path.isfile('X_train.npy'):
        X,y = io.load_pictures('characters/', True)
        X_train, X_test, y_train, y_test = io.split_data(X, y, 85)
        io.write_data_to_file(X_train, X_test, y_train, y_test)
    else:
        X_train, X_test, y_train, y_test = io.load_data_from_file()
    
    train_iter, val_iter = get_iterators(X_train, y_train, X_test, y_test, batch_size)

    mod_score = fit(sym, train_iter, val_iter, batch_size)
    print (mod_score)

if __name__ == "__main__":
    main()
