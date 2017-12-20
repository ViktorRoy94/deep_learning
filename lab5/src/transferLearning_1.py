import mxnet as mx
import numpy as np
import cv2
import glob

from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

map_characters = {0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon',
        2: 'bart_simpson', 3: 'charles_montgomery_burns', 4: 'chief_wiggum',
        5: 'comic_book_guy', 6: 'edna_krabappel', 7: 'homer_simpson',
        8: 'kent_brockman', 9: 'krusty_the_clown', 10: 'lisa_simpson',
        11: 'marge_simpson', 12: 'milhouse_van_houten', 13: 'moe_szyslak',
        14: 'ned_flanders', 15: 'nelson_muntz', 16: 'principal_skinner',
        17: 'sideshow_bob'}

def get_model(prefix, epoch):
    mx.test_utils.download(prefix+'-symbol.json')
    mx.test_utils.download(prefix+'-%04d.params' % (epoch,))

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

def predict(url, mod):
    img = get_image(url, show=True)
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    y = np.argsort(np.squeeze(prob))[::-1]
    return y
    
def main():
    get_model('http://data.mxnet.io/models/imagenet/resnet/18-layers/resnet-18', 0)
    sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-18', 0)

    mod = mx.mod.Module(symbol=sym, label_names=None, context=mx.cpu())
    mod.bind(for_training = False, data_shapes=[('data', (1, 3, 24, 24))])
    mod.set_params(arg_params, aux_params, allow_missing=True)

    size = 0
    numCorrect = 0
    path = 'characters/'
    for class_simpson, char in map_characters.items():
        pictures = [k for k in glob.glob(path + '%s/*' % char)]
        for pic in pictures:
            y = predict(pic, mod)
            if y[0] == class_simpson:
                numCorrect = numCorrect + 1
            size = size + 1

    accuracy = numCorrect / size
    print(accuracy)

if __name__ == "__main__":
    main()
