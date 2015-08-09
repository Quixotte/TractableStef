
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


    accuracy = 0
    i = 0
    while accuracy == 0 and i < 10:        #hacky solution to prevent 0 acc, not pretty :/
        solver = caffe.get_solver(solver_file)
        solver.step(100)
        if "accuracy" in solver.net.blobs.keys():
            print "================"
            print "accuracy key is there"
        accuracy = solver.net.blobs['accuracy'].data
        i+=1
        if accuracy == 0:
            print 'accuracy is 0 ,trying to re-initialize'

    print 'accuracy not 0 anymore, continueing training'
    #solver.step(5000)


results_dir = 'results/'

def get_advanced_accuracy_MTL(net_file, caffe_model):
    print net_file
    print caffe_model

    net = caffe.Net(net_file, caffe_model, 1) #the 1 stands for testing only

    caffe.set_mode_gpu()

    pos_acc = [[]]
    neg_acc = [[]]

    n_test_files = 954

    for i in numpy.arange(n_test_files):
        net.forward()
        for label in numpy.arange(1,12):
            label_acc = net.blobs['accuracy' + str(label)].data
            label_value = net.blobs['label' + str(label)].data

            if numpy.sum(label) == 1.:
                pos_acc[label - 1, :].append(numpy.sum(label_acc))
            else:
                neg_acc[label - 1, :].append(numpy.sum(label_acc))

    pos_acc = numpy.mean(numpy.asarray(pos_acc), axis=1)
    neg_acc = numpy.mean(numpy.asarray(neg_acc), axis=1)

    results_file = 'MTL_results'
    results = os.path.join(results_dir, results_file)

    with open(results, 'w') as f:
        for i in numpy.arange(1,12):
            f.write("Label " + str(i) + "\n")
            f.write("Positive accuracy rate: " + str(pos_acc[i-1]) + "\n")
            f.write("Negative accuracy rate: " + str(neg_acc[i-1]) + "\n\n")



def get_advanced_accuracy(net_file, caffe_model, label_num):

    print net_file
    print caffe_model
    net = caffe.Net(net_file, caffe_model, 1) #the 1 stands for testing only

    caffe.set_mode_gpu()
    #caffe.set_phase_test()

    n_test_files = 954  #run trainsplit to find out this number again

    pos_acc = []
    neg_acc = []

    for i in numpy.arange(n_test_files):
        net.forward()

        accuracy = net.blobs['accuracy'].data
        label = net.blobs['label' + str(label_num)].data
        #print "accuracy: " + str(accuracy) + " with label: " + str(label)
        if numpy.sum(label) == 1.:
            pos_acc.append(numpy.sum(accuracy)) #change label to 1 dimension from 1x1
        else:
            neg_acc.append(numpy.sum(accuracy))

    #print pos_acc
    #print neg_acc

    pos_acc = numpy.mean(numpy.asarray(pos_acc))
    neg_acc = numpy.mean(numpy.asarray(neg_acc))

    results_file = 'label_' + str(label_num) + '_results'
    results = os.path.join(results_dir, results_file)

    with open(results, 'w') as f:
        f.write("Positive accuracy rate: " + str(pos_acc) + "\n")
        f.write("Negative accuracy rate: " + str(neg_acc))


if __name__ == "__main__":
    #learn("stef_solver.prototxt", 42)

    base = 'binary_solvers/binary_stef_solver_'
    for i in numpy.arange(1, 12):
        file_name = base + str(i) + '.prototxt'
        learn(file_name, i)

    #get_advanced_accuracy_MTL("stef_net.prototxt", "nets/nets/snapshot_MTL_iter_10000.caffemodel")

if __name__ == "__main__1":

    ##In order for this to work the batch size of the test has to be 1.

    net_base = 'binary_solvers/binary_stef_net_'
    model_base = 'nets/snapshot_file_binary_'
    for i in numpy.arange(1,12):
        net_file = net_base + str(i) + '.prototxt'
        snapshot_file = model_base + str(i) + '_iter_10000.caffemodel'
        get_advanced_accuracy(net_file, snapshot_file, i)



    #learn_and_test('binary_solvers/binary_stef_solver_1.prototxt', 1)
