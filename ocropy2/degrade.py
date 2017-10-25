################################################################
### text image generation with Cairo
################################################################

from __future__ import print_function

import ctypes
from numpy import *
from scipy import *
from scipy.misc import imsave
from pylab import *
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import distance_transform_edt,binary_erosion,binary_dilation
from scipy.ndimage.interpolation import map_coordinates,zoom,rotate
from scipy.stats.mstats import mquantiles


def gauss_degrade(image,margin=1.0,change=None,noise=0.02,minmargin=0.5,inner=1.0):
    if image.ndim==3: image = mean(image,axis=2)
    m = mean([amin(image),amax(image)])
    image = 1*(image>m)
    if margin<minmargin: return 1.0*image
    pixels = sum(image)
    if change is not None:
        npixels = int((1.0+change)*pixels)
    else:
        edt = distance_transform_edt(image==0)
        npixels = sum(edt<=(margin+1e-4))
    r = int(max(1,2*margin+0.5))
    ri = int(margin+0.5-inner)
    if ri<=0: mask = binary_dilation(image,iterations=r)-image
    else: mask = binary_dilation(image,iterations=r)-binary_erosion(image,iterations=ri)
    image = image+mask*randn(*image.shape)*noise*min(1.0,margin**2)
    smoothed = gaussian_filter(1.0*image,margin)
    frac = max(0.0,min(1.0,npixels*1.0/prod(image.shape)))
    threshold = mquantiles(smoothed,prob=[1.0-frac])[0]
    result = (smoothed>threshold)
    return 1.0*result

def gauss_distort(images,maxdelta=2.0,sigma=10.0):
    n,m = images[0].shape
    deltas = randn(2,n,m)
    deltas = gaussian_filter(deltas,(0,sigma,sigma))
    deltas /= max(amax(deltas),-amin(deltas))
    deltas *= maxdelta
    xy = transpose(array(meshgrid(range(n),range(m))),axes=[0,2,1])
    # print(xy.shape, deltas.shape)
    deltas +=  xy
    return [map_coordinates(image,deltas,order=1) for image in images]
