import glob
import cv2
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
        print(char + "  " + str(nb_pic))
        for pic in np.random.choice(pictures, nb_pic):
            a = cv2.imread(pic)
            if BGR:
                a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
            a = cv2.resize(a, (pic_size,pic_size)).astype('float32') / 255
            a = np.rollaxis(a, 2, 0)
            pics.append(a)
            labels.append(class_simpson)
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