
from sklearn import cross_validation
from numpy import genfromtxt, savetxt
from sklearn.cross_validation import train_test_split

import caffe
import sys
import os
import h5py
import shutil
import csv
import math
import numpy
from scipy import misc
import calc_mean_image

def importCSV( csvfilePath, delimiterChar=",", ignoreFirstRow=False):
    """
        Import from CSV, yield each row as a tuple
    """
    data = []
    with open(csvfilePath, 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=delimiterChar)
        for index, row in enumerate(csvreader):
            if ignoreFirstRow:
                ignoreFirstRow = False
                continue
            data.append( row )
        csvfile.close()
    return data


def writeHD5(path, data, label):
    with h5py.File(path, 'w') as f:
        f['data'] = data
        f['label'] = label.astype(numpy.float32)



def learn_and_test(solver_file):
    caffe.set_mode_cpu()
    solver = caffe.get_solver(solver_file)
    solver.solve()

    accuracy = 0
    test_iters = int(len(Xt) / solver.test_nets[0].blobs['data'].num)
    for i in range(test_iters):
        solver.test_nets[0].forward()
        accuracy += solver.test_nets[0].blobs['accuracy'].data
    accuracy /= test_iters
    return accuracy

def load_one_image(filename, image_mean, do_print = False):
    #Have to be transposed for the 3 x 256 x 256 instead of 256x256x3 (which it was originally)
    #scale from 0-256 to 0-1, since the hdf5 layer can't do this
    img = (caffe.io.load_image(filename).astype(numpy.float32) - image_mean).T * 0.00390625
    if do_print:
        print type(data)
        print type(img)
        print img.shape, img.dtype
    return img

def main():
    base = '/data/ad6813/Nus-wide/TractableStef'

    file_labels = os.path.join(base, 'labels.txt')

    dirname = os.path.join(base, 'hd5')

    if not os.path.exists(os.path.abspath(dirname)):
        os.makedirs(dirname)

    label2num = {'lake':1, 'plants':2, 'window':3, 'buildings':4, 'grass':5, 'animal':6, 'water':7, 'person':8, 'clouds':9, 'sky':10, 'NA':0}

    dataset = importCSV(file_labels, delimiterChar=' ')

    data_all   = [x[0] for x in dataset]
    labels_all = [x[1:] for x in dataset]
    labels_all = [map(lambda label: label2num[label], x) for x in labels_all]

    # labels_binary = [0] * len(labels_all)
    # for (i, v_label) in enumerate(labels_all):
    #     label_coded = [0] * 11
    #     for num in v_label:
    #         label_coded[num] = 1

    #     # labels_binary[i] = label_coded
    #     labels_binary[i] = numpy.array(label_coded, dtype=numpy.uint8)
    labels_binary = numpy.zeros((len(labels_all), len(label2num)), dtype=numpy.float32)
    for (i, v_label) in enumerate(labels_all):
        # label_coded = [0] * 11
        for num in v_label:
            labels_binary[i,num] = 1.0

        # labels_binary[i] = label_coded
        # labels_binary[i] = numpy.array(label_coded, dtype=numpy.uint8)

    print labels_binary
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

    n_correct_images = 9532         #found in earlier run, don't want to recalc every time: 8295, nice for pre-allocation of memory though

    correct_shape = (3, 256, 256)

    N_train = int(n_correct_images*0.9)
    print 'Number of training images:'
    print N_train

    train_images = numpy.zeros((N_train, correct_shape[0], correct_shape[1], correct_shape[2]), dtype=numpy.float32)
    train_labels = numpy.zeros((N_train, labels_train.shape[1]), dtype=numpy.float32)

    test_images = numpy.zeros((n_correct_images - N_train, correct_shape[0], correct_shape[1], correct_shape[2]), dtype=numpy.float32)
    test_labels = numpy.zeros((n_correct_images - N_train, labels_train.shape[1]), dtype=numpy.float32)

    mean_image = calc_mean_image.get_image_mean()

    index = 0
    for (i, d) in enumerate(data_train):
        try:
            img = load_one_image(os.path.join(image_dir, d), mean_image)
            if index < N_train:
                train_images[index] = img
                train_labels[index] = labels_train[i, :]
            else:
                test_images[index - N_train] = img
                test_labels[index - N_train] = labels_train[i, :]
            index += 1

        if i > 8000:
            break #gettin out of memory errors, gotta use less images i guess
        except IOError:
            print "Couldn't find file", d

    print "n_correct_images =" , str(index)

    #test_images = [load_one_image(image_dir + d) for d in data_test]

    hd5_train_images_filename = os.path.join(base, "hd5_images_stef/hd5_images_train")
    hd5_test_images_filename = os.path.join(base, "hd5_images_stef/hd5_images_test")
    print 'Shape of images:'
    print train_images.shape

    print 'Shape of train labels:'
    print train_labels.shape

    print train_labels

    print 'Shape of test labels:'
    print test_labels.shape

    print test_labels

    do_write = True
    chunk_size = 64
    if do_write:
        #Write away training data
        n__train_images = train_images.shape[0]
        chunks = numpy.arange(int(math.ceil(n__train_images/chunk_size)))
        print chunks
        hd5_meta_train_stef = os.path.join(base, "hd5_images_stef/stef_train.txt")

        with open(hd5_meta_train_stef, 'w') as meta:
            for chunk in chunks:
                print 'chunk:'
                print chunk
                tmp_train_images = train_images[chunk*chunk_size : (chunk+1)*chunk_size]
                tmp_train_labels = train_labels[chunk*chunk_size : (chunk+1)*chunk_size]
                try:
                    with h5py.File(hd5_train_images_filename + str(chunk) + ".hdf5", 'w') as f:
                        f.create_dataset("data", tmp_train_images.shape , compression='gzip', compression_opts=1, dtype=numpy.float32, data=tmp_train_images)
                        f.create_dataset("label", tmp_train_labels.shape , compression='gzip', compression_opts=1, dtype=numpy.float32, data=tmp_train_labels)
                except IOError:
                    print 'IOerror, carrying on'


                meta.write(hd5_train_images_filename + str(chunk) + ".hdf5" + '\n')

        #Write away test data
        n_test_images = test_images.shape[0]
        chunks = numpy.arange(int(math.ceil(n_test_images/chunk_size)))
        print chunks
        hd5_meta_test_stef = os.path.join(base, "hd5_images_stef/stef_test.txt")

        with open(hd5_meta_test_stef, 'w') as meta:
            for chunk in chunks:
                print 'chunk:'
                print chunk
                tmp_test_images = test_images[chunk*chunk_size : (chunk+1)*chunk_size]
                tmp_test_labels = test_labels[chunk*chunk_size : (chunk+1)*chunk_size]
                try:
                    with h5py.File(hd5_test_images_filename + str(chunk) + ".hdf5", 'w') as f:
                        f.create_dataset("data", tmp_test_images.shape , compression='gzip', compression_opts=1, dtype=numpy.float32, data=tmp_test_images)
                        f.create_dataset("label", tmp_test_labels.shape , compression='gzip', compression_opts=1, dtype=numpy.float32, data=tmp_test_labels)
                except IOError:
                    print 'IOerror, carrying on'


                meta.write(hd5_test_images_filename + str(chunk) + ".hdf5" + '\n')


if __name__ == "__main__":
    main()
    #load_one_image("2.jpg")