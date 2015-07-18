
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
    caffe.set_mode_gpu()
    solver = caffe.get_solver(solver_file)
    solver.step(1000)

    #accuracy = 0
    #test_iters = int(len(Xt) / solver.test_nets[0].blobs['data'].num)
    #for i in range(test_iters):
    #    solver.test_nets[0].forward()
    #    accuracy += solver.test_nets[0].blobs['accuracy'].data
    #accuracy /= test_iters
    #return accuracy

if __name__ == "__main__":
    learn_and_test('stef_binary_solver.prototxt')
