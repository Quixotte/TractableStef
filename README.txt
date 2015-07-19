Building MTL deep learning network with caffe

Done so far:
reformatted raw images to hdf5 format so I can use the HDF5 data layer which can handle
multiple labels.

Got the core multi-task learning network up and running, slicing labels into seperate LOSS
and ACCURACY layers.

Todo:
Save the net to a file with python

Build the binary nets

Save and plot the loss and accuracy of both binary and mtl nets.

figure out the best training params

figure out the best layers

Fancy preprocessing or data augmentation: take 5 240x240 patches instead of raw image.

See if "dropout" technique can be added
