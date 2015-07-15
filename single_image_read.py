__author__ = 'stefjanssen'

from scipy import misc

img = misc.imread('1.jpg')
print type(img)
print img.shape, img.dtype
