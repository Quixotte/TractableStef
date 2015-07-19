
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

def load_one_image(filename, do_print = False):
    img = misc.imread(filename).astype(numpy.float32)
    if do_print:
        print type(data)
        print type(img)
        print img.shape, img.dtype
    return img

def main():
    base = '/data/ad6813/Nus-wide/TractableStef'

    file_labels = os.path.join(base, 'labels.txt')

    label2num = {'lake':1, 'plants':2, 'window':3, 'buildings':4, 'grass':5, 'animal':6, 'water':7, 'person':8, 'clouds':9, 'sky':10, 'NA':0}

    dataset = trainsplit.importCSV(file_labels, delimiterChar=' ')

    data_all   = [x[0] for x in dataset]
    labels_all = [x[1:] for x in dataset]
    labels_all = [map(lambda label: label2num[label], x) for x in labels_all]

    labels_binary = numpy.zeros((len(labels_all), len(label2num)), dtype=numpy.float32)
    for (i, v_label) in enumerate(labels_all):
        # label_coded = [0] * 11
        for num in v_label:
            labels_binary[i,num] = 1.0

        # labels_binary[i] = label_coded
        # labels_binary[i] = numpy.array(label_coded, dtype=numpy.uint8)

    print 'First few labels:', labels_binary[0:10]
    print '==='

    data_train = data_all
    labels_train = labels_binary

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

    mean_image = numpy.mean(images)
    numpy_file_name = os.path.join(base, "mean_image")
    numpy.save(numpy_file_name, mean_image)

    loaded_img = numpy.load(numpy_file_name)

    if mean_image == loaded_img:
        print "they're the same"
if __name__ == "__main__":
    main()