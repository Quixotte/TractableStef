
from sklearn import cross_validation
from numpy import genfromtxt, savetxt
from sklearn.cross_validation import train_test_split

import caffe

import os
import h5py
import shutil
import csv

import numpy

def get_advanced_accuracy(solver_file, caffe_model, label_num):
    net = caffe.Net(solver_file, caffe_model)

    caffe.set_mode_gpu()
    caffe.set_phase_test()


    res = net.forward()

    accuracy = net.blobs['accuracy'].data
    label = net.blobs['label' + str(label_num)]

    pos_acc = []
    neg_acc = []

    if numpy.sum(label) == 1:
        pos_acc.append(numpy.sum(label))
    else:
        neg_acc.append(numpy.sum(label))



def learn_and_test(solver_file, label_num):

    #net = caffe.Net(solver_file)
    #net.forward() # this will load the next mini-batch as defined in the net
    #label1 = net.blobs['label1'].data # or whatever you want

    print '========'
    print 'currently solving for label number: ' + str(label_num)
    caffe.set_mode_gpu()
    solver = caffe.get_solver(solver_file)
    solver.step(10001)

    #accuracy = 0
    #test_iters = int(len(Xt) / solver.test_nets[0].blobs['data'].num)
    #for i in range(test_iters):
    #    solver.test_nets[0].forward()
    #    accuracy += solver.test_nets[0].blobs['accuracy'].data
    #accuracy /= test_iters
    #return accuracy

if __name__ == "__main__":
    base = 'binary_solvers/binary_stef_solver_'
    for i in numpy.arange(1,12):
        file_name = base + str(i) + '.prototxt'
        learn_and_test(file_name, i)


    #learn_and_test('binary_solvers/binary_stef_solver_1.prototxt', 1)
