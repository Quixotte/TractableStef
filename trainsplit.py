
from sklearn import cross_validation
from numpy import genfromtxt, savetxt
from sklearn.cross_validation import train_test_split

import caffe

import os
import h5py
import shutil
import csv

import numpy



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

def main():
    base = '/data/ad6813/Nus-wide/TractableStef'

    file_labels = os.path.join(base, 'labels.txt')

    dirname = os.path.join(base, 'hd5')

    if not os.path.exists(os.path.abspath(dirname)):
        os.makedirs(dirname)

    hd5_train_filename = os.path.join(dirname, 'train.h5')
    hd5_test_filename  = os.path.join(dirname, 'test.h5')
    hd5_meta_train     = os.path.join(base, 'train.txt')
    hd5_meta_test      = os.path.join(base, 'test.txt')

    label2num = {'lake':1, 'plants':2, 'window':3, 'buildings':4, 'grass':5, 'animal':6, 'water':7, 'person':8, 'clouds':9, 'sky':10, 'NA':11}

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

    labels_binary = numpy.zeros((len(labels_all), 12), dtype=numpy.float32)
    for (i, v_label) in enumerate(labels_all):
        # label_coded = [0] * 11
        for num in v_label:
            labels_binary[i,num] = 1.0

        # labels_binary[i] = label_coded
        # labels_binary[i] = numpy.array(label_coded, dtype=numpy.uint8)

    print labels_binary
    print 'First few labels:', labels_binary[0:6]
    print '==='

    N = len(labels_binary)
    N_train = int(N*0.25)

    data_train = data_all[0:N_train]
    data_test  = data_all[N_train:N]
    labels_train = labels_binary[0:N_train]
    labels_test  = labels_binary[N_train:N]

    # TODO: doesn't work with vector labels
    # data_train, data_test, labels_train, labels_test = train_test_split(data_all, labels_all, test_size=0.20, random_state=42)

    print '==='
    print labels_train

    # HDF5DataLayer source should be a file containing a list of HDF5 filenames.
    # To show this off, we'll list the same data file twice.
    # writeHD5(hd5_train_filename, data_train, labels_train)
    with h5py.File(hd5_train_filename, 'w') as f:
        f['data'] = data_train
        f['label'] = labels_train.astype(numpy.float32)
    with open(hd5_meta_train, 'w') as f:
        f.write(hd5_train_filename + '\n')
        f.write(hd5_train_filename + '\n')

    # HDF5 is pretty efficient, but can be further compressed.
    comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
    with h5py.File(hd5_test_filename, 'w') as f:
        f.create_dataset('data', data=data_test, **comp_kwargs)
        f.create_dataset('label', data=labels_test.astype(numpy.float32), **comp_kwargs)
    with open(hd5_meta_test, 'w') as f:
        f.write(hd5_test_filename + '\n')

    # %timeit learn_and_test('hdf5_classification/solver.prototxt')
    acc = learn_and_test('examples/hdf5_classification/solver.prototxt')
    print("Accuracy: {:.3f}".format(acc))



if __name__ == "__main__":
    main()
