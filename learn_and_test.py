
from sklearn import cross_validation
from numpy import genfromtxt, savetxt
from sklearn.cross_validation import train_test_split

import caffe

import os
import h5py
import shutil
import csv

import numpy

def learn_and_test(solver_file, label_num):

    #net = caffe.Net(solver_file)
    #net.forward() # this will load the next mini-batch as defined in the net
    #label1 = net.blobs['label1'].data # or whatever you want

    print '========'
    print 'currently solving for label number: ' + str(label_num)
    caffe.set_mode_gpu()
    solver = caffe.get_solver(solver_file)
    solver.step(1)

    #accuracy = 0
    #test_iters = int(len(Xt) / solver.test_nets[0].blobs['data'].num)
    #for i in range(test_iters):
    #    solver.test_nets[0].forward()
    #    accuracy += solver.test_nets[0].blobs['accuracy'].data
    #accuracy /= test_iters
    #return accuracy

if __name__ == "__main__":
    base = 'binary_solvers/binary_stef_solver_'
    for i in numpy.arange(9,12):
        file_name = base + str(i) + '.prototxt'
        learn_and_test(file_name, i)


    #learn_and_test('binary_solvers/binary_stef_solver_1.prototxt', 1)
