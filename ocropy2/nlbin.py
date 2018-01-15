#!/usr/bin/env python

from __future__ import print_function

from pylab import *
from numpy.ctypeslib import ndpointer
import argparse,os,os.path
from scipy.ndimage import filters,interpolation,morphology,measurements
from scipy import stats
import multiprocessing

class Struct(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)
    def __call__(self, **kw):
        result = Struct(**self.__dict__)
        result.__dict__.update(kw)
        return result

default_args = Struct(threshold=0.5,
                      zoom=0.5,
                      escale=1.0,
                      bignore=0.1,
                      perc=80,
                      range=20,
                      maxskew=2.0,
                      gray=False,
                      lo=5,
                      hi=90,
                      skewsteps=8,
                      debug=0,
                      nocheck=False,
                      show=False)

def print_info(*objs):
    print("INFO: ", *objs, file=sys.stdout)

def print_error(*objs):
    print("ERROR: ", *objs, file=sys.stderr)

def check_page(image):
    if len(image.shape)==3: return "input image is color image %s"%(image.shape,)
    if mean(image)<median(image): return "image may be inverted"
    h,w = image.shape
    if h<600: return "image not tall enough for a page image %s"%(image.shape,)
    if h>10000: return "image too tall for a page image %s"%(image.shape,)
    if w<600: return "image too narrow for a page image %s"%(image.shape,)
    if w>10000: return "line too wide for a page image %s"%(image.shape,)
    return None

def estimate_skew_angle(image,angles):
    estimates = []
    for a in angles:
        v = mean(interpolation.rotate(image,a,order=0,mode='constant'),axis=1)
        v = var(v)
        estimates.append((v,a))
    if args.debug>0:
        plot([y for x,y in estimates],[x for x,y in estimates])
        ginput(1,args.debug)
    _,a = max(estimates)
    return a

def H(s): return s[0].stop-s[0].start
def W(s): return s[1].stop-s[1].start
def A(s): return W(s)*H(s)

def dshow(image,info):
    if args.debug<=0: return
    ion(); gray(); imshow(image); title(info); ginput(1,args.debug)

def process_image(raw, args=default_args):
    dshow(raw,"input")
    # perform image normalization
    image = raw-amin(raw)
    assert amin(image) < amax(image)
    image /= amax(image)

    if not args.nocheck:
        page_error = check_page(amax(image)-image)
        assert not page_error, page_error

    # check whether the image is already effectively binarized
    if args.gray:
        extreme = 0
    else:
        extreme = (sum(image<0.05)+sum(image>0.95))*1.0/prod(image.shape)
    if extreme>0.95:
        comment = "no-normalization"
        flat = image
    else:
        comment = ""
        # if not, we need to flatten it by estimating the local whitelevel
        print_info("flattening")
        m = interpolation.zoom(image,args.zoom)
        m = filters.percentile_filter(m,args.perc,size=(args.range,2))
        m = filters.percentile_filter(m,args.perc,size=(2,args.range))
        m = interpolation.zoom(m,1.0/args.zoom)
        if args.debug>0: clf(); imshow(m,vmin=0,vmax=1); ginput(1,args.debug)
        w,h = minimum(array(image.shape),array(m.shape))
        flat = clip(image[:w,:h]-m[:w,:h]+1,0,1)
        if args.debug>0: clf(); imshow(flat,vmin=0,vmax=1); ginput(1,args.debug)

    # estimate skew angle and rotate
    if args.maxskew>0:
        print_info("estimating skew angle")
        d0,d1 = flat.shape
        o0,o1 = int(args.bignore*d0),int(args.bignore*d1)
        flat = amax(flat)-flat
        flat -= amin(flat)
        est = flat[o0:d0-o0,o1:d1-o1]
        ma = args.maxskew
        ms = int(2*args.maxskew*args.skewsteps)
        angle = estimate_skew_angle(est,linspace(-ma,ma,ms+1))
        flat = interpolation.rotate(flat,angle,mode='constant',reshape=0)
        flat = amax(flat)-flat
    else:
        angle = 0

    # estimate low and high thresholds
    print_info("estimating thresholds")
    d0,d1 = flat.shape
    o0,o1 = int(args.bignore*d0),int(args.bignore*d1)
    est = flat[o0:d0-o0,o1:d1-o1]
    if args.escale>0:
        # by default, we use only regions that contain
        # significant variance; this makes the percentile
        # based low and high estimates more reliable
        e = args.escale
        v = est-filters.gaussian_filter(est,e*20.0)
        v = filters.gaussian_filter(v**2,e*20.0)**0.5
        v = (v>0.3*amax(v))
        v = morphology.binary_dilation(v,structure=ones((int(e*50),1)))
        v = morphology.binary_dilation(v,structure=ones((1,int(e*50))))
        if args.debug>0: imshow(v); ginput(1,args.debug)
        est = est[v]
    lo = stats.scoreatpercentile(est.ravel(),args.lo)
    hi = stats.scoreatpercentile(est.ravel(),args.hi)
    # rescale the image to get the gray scale image
    print_info("rescaling")
    flat -= lo
    flat /= (hi-lo)
    flat = clip(flat,0,1)
    if args.debug>0: imshow(flat,vmin=0,vmax=1); ginput(1,args.debug)
    bin = 1*(flat>args.threshold)

    # output the normalized grayscale and the thresholded images
    print_info("lo-hi (%.2f %.2f) angle %4.1f %s" % (lo, hi, angle, comment))
    if args.debug>0 or args.show: clf(); gray();imshow(bin); ginput(1,max(0.1,args.debug))

    return bin, flat
