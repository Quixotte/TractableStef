
from sklearn import cross_validation
from numpy import genfromtxt, savetxt
from sklearn.cross_validation import train_test_split

import caffe

import os
import h5py
import shutil
import csv

import numpy

def learn_and_test(solver_file):

    #net = caffe.Net(solver_file)
    #net.forward() # this will load the next mini-batch as defined in the net
    #label1 = net.blobs['label1'].data # or whatever you want

    #print label1

    caffe.set_mode_gpu()
    solver = caffe.get_solver(solver_file)
    solver.solve()

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
        print file_name
    #learn_and_test('stef_solver.prototxt')
