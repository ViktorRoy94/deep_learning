import cv2
import glob
import numpy as np

map_characters = {0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon',
        2: 'bart_simpson', 3: 'charles_montgomery_burns', 4: 'chief_wiggum',
        5: 'comic_book_guy', 6: 'edna_krabappel', 7: 'homer_simpson',
        8: 'kent_brockman', 9: 'krusty_the_clown', 10: 'lisa_simpson',
        11: 'marge_simpson', 12: 'milhouse_van_houten', 13: 'moe_szyslak',
        14: 'ned_flanders', 15: 'nelson_muntz', 16: 'principal_skinner',
        17: 'sideshow_bob'}

pic_size = 64

def load_pictures(path, BGR):
    pics = []
    labels = []
    for k, char in map_characters.items():
    	print(char)
        pictures = [k for k in glob.glob(path + '%s/*' % char)]
        nb_pic = len(pictures)
        for pic in np.random.choice(pictures, nb_pic):
            a = cv2.imread(pic)
            if BGR:
                a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
            a = cv2.resize(a, (pic_size,pic_size))
            pics.append(a)
            labels.append(k)
    return np.array(pics), np.array(labels)

def main():
    X,y = load_pictures('/home/viktor/Downloads/simpsons/', True)
    print(X.shape)
    print(y.shape)

if __name__ == '__main__':
    main()
