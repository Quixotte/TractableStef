
from sklearn import cross_validation
from numpy import genfromtxt, savetxt
from sklearn.cross_validation import train_test_split

import caffe
import sys
import os
import trainsplit
import h5py
import shutil
import csv
import math
import numpy
from scipy import misc

base = '/data/ad6813/Nus-wide/TractableStef'
numpy_file_name = os.path.join(base, "mean_image.npy")

def load_one_image(filename, do_print = False):
    img = misc.imread(filename).astype(numpy.float32)
    if do_print:
        print type(data)
        print type(img)
        print img.shape, img.dtype
    return img

def main():


    file_labels = os.path.join(base, 'labels.txt')

    dataset = trainsplit.importCSV(file_labels, delimiterChar=' ')

    data_all   = [x[0] for x in dataset]


    data_train = data_all

    # TODO: doesn't work with vector labels
    # data_train, data_test, labels_train, labels_test = train_test_split(data_all, labels_all, test_size=0.20, random_state=42)

    # HDF5DataLayer source should be a file containing a list of HDF5 filenames.
    # To show this off, we'll list the same data file twice.
    # writeHD5(hd5_train_filename, data_train, labels_train)

    image_dir = os.path.join(base, "res_imgs/")

    #load images
    #train_images = [load_one_image(os.path.join(image_dir, d)) for d in data_train if os.path.exists(image_dir + d)]

    correct_shape = (256, 256, 3)   #apperantly some images are 256x256, not RGB, filtering them out, could convert but just want to quickfix
                                    #Todo: Can be done waaay better with caffe.io, but that can wait
    n_correct_images = 8295         #found in earlier run, don't want to recalc every time: 8295, nice for pre-allocation of memory though

    images = numpy.zeros((n_correct_images, correct_shape[0], correct_shape[1], correct_shape[2]), dtype=numpy.float32)

    index = 0
    for (i, d) in enumerate(data_train):
        try:
            img = load_one_image(os.path.join(image_dir, d))
            if img.shape == correct_shape:
                images[index] = img
                index += 1
        except IOError:
            print "Couldn't find file", d

    mean_image = numpy.mean(images, axis=0)
    numpy_file_name = os.path.join(base, "mean_image.npy")
    numpy.save(numpy_file_name, mean_image)



if __name__ == "__main__":
    main()
    loaded_img = numpy.load(numpy_file_name)
    print loaded_img
