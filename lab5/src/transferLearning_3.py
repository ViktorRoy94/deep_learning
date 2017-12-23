import mxnet as mx
import numpy as np
import logging
import time
import glob
import cv2
import read_write_data as io
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

map_characters = {0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon',
        2: 'bart_simpson', 3: 'charles_montgomery_burns', 4: 'chief_wiggum',
        5: 'comic_book_guy', 6: 'edna_krabappel', 7: 'homer_simpson',
        8: 'kent_brockman', 9: 'krusty_the_clown', 10: 'lisa_simpson',
        11: 'marge_simpson', 12: 'milhouse_van_houten', 13: 'moe_szyslak',
        14: 'ned_flanders', 15: 'nelson_muntz', 16: 'principal_skinner',
        17: 'sideshow_bob'}


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

def get_image(url, show=False):
    img = cv2.cvtColor(cv2.imread(url), cv2.COLOR_BGR2RGB)
    if img is None:
         return None
    # convert into format (batch, RGB, width, height)
    img = cv2.resize(img, (24, 24))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return img

@timer
def train(mod, train, val, batch_size):
    logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
    mod.fit(train, val,
            num_epoch=100,
            allow_missing=True,
            batch_end_callback = mx.callback.Speedometer(batch_size, 1000),
            kvstore='device',
            optimizer='sgd',
            optimizer_params={'learning_rate':0.01},
            initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
            eval_metric='acc')
    metric = mx.metric.Accuracy()
    return mod.score(val, metric)

def predict(url, mod):
    img = get_image(url, show=True)
    mod.forward(Batch([mx.nd.array(img)]))
    features = mod.get_outputs()[0].asnumpy()
    return features

def main():
    get_model('http://data.mxnet.io/models/imagenet/resnet/18-layers/resnet-18', 0)
    sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-18', 0)
    
    all_layers = sym.get_internals()
    fe_sym = all_layers['flatten0_output']
    fe_mod = mx.mod.Module(symbol=fe_sym, context=mx.cpu(), label_names=None)
    fe_mod.bind(for_training=False, data_shapes=[('data', (1,3,24,24))])
    fe_mod.set_params(arg_params, aux_params)

    X = []
    y = []
    numCorrect = 0
    path = 'characters/'
    for class_simpson, char in map_characters.items():
        pictures = [k for k in glob.glob(path + '%s/*' % char)]
        for pic in pictures:
            features = predict(pic, fe_mod)
            X.append(features)
            y.append(class_simpson)

    X_train, X_test, y_train, y_test = io.split_data(np.array(X), np.array(y), 85)

    batch_size = 10
    train_iter = mx.io.NDArrayIter(X_train,
                                   y_train,
                                   batch_size, shuffle = True)
    val_iter = mx.io.NDArrayIter(X_test,
                                 y_test,
                                 batch_size)

    data = mx.sym.var('data')
    data = mx.sym.flatten(data = data)
    fc = mx.sym.FullyConnected(data = data, num_hidden = 300)
    tanh = mx.sym.Activation(data = fc, act_type = "tanh")
    fc2 = mx.sym.FullyConnected(data = tanh, num_hidden = 18)
    fcnn = mx.sym.SoftmaxOutput(data = fc2, name = 'softmax')
    fcnn_model = mx.mod.Module(symbol = fcnn, context = mx.cpu())

    train(fcnn_model, train_iter, val_iter, batch_size)


if __name__ == "__main__":
    main()
