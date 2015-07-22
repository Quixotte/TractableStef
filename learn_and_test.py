
from sklearn import cross_validation
from numpy import genfromtxt, savetxt
from sklearn.cross_validation import train_test_split

import caffe

import os
import h5py
import shutil
import csv

import numpy

def learn(solver_file, label_num):

    #net = caffe.Net(solver_file)
    #net.forward() # this will load the next mini-batch as defined in the net
    #label1 = net.blobs['label1'].data # or whatever you want

    print '========'
    print 'currently solving for label number: ' + str(label_num)
    caffe.set_mode_gpu()
    solver = caffe.get_solver(solver_file)
    solver.step(101)

def get_advanced_accuracy(net_file, caffe_model, label_num):

    print net_file
    print caffe_model
    net = caffe.Net(net_file, caffe_model, 1)

    caffe.set_mode_gpu()
    caffe.set_phase_test()

    n_test_files = 954  #run trainsplit to find out this number again

    pos_acc = []
    neg_acc = []

    for i in numpy.arange(n_test_files):
        res = net.forward()

        accuracy = net.blobs['accuracy'].data
        label = net.blobs['label' + str(label_num)]

        if numpy.sum(label) == 1:
            pos_acc.append(numpy.sum(label))
        else:
            neg_acc.append(numpy.sum(label))

    pos_acc = numpy.mean(numpy.asarray(pos_acc))
    neg_acc = numpy.mean(numpy.asarray(neg_acc))
    results_dir = 'results/'
    results_file = 'label_' + str(label_num) + '_results'
    results = os.path.join(results_dir, results_file)
    with open(results, 'w') as f:
        f.write("Positive accuracy rate: " + str(pos_acc))
        f.write("Negative accuracy rate: " + str(neg_acc))


if __name__ == "__main__2":
    base = 'binary_solvers/binary_stef_solver_'
    for i in numpy.arange(1,2):
        file_name = base + str(i) + '.prototxt'
        learn(file_name, i)

if __name__ == "__main__":

    ##In order for this to work the batch size of the test has to be 1.

    net_base = 'binary_solvers/binary_stef_net_'
    model_base = '../../../../tmp/stef_net/snapshot_file_binary_'
    for i in numpy.arange(1,2):
        net_file = net_base + str(i) + '.prototxt'
        snapshot_file = model_base + str(i) + '.caffemodel'
        get_advanced_accuracy(net_file, snapshot_file, i)



    #learn_and_test('binary_solvers/binary_stef_solver_1.prototxt', 1)
