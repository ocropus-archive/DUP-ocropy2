#!/usr/bin/python

import matplotlib

import numpy as np
import pylab
import os
import glob
import time
import ocropy2
import dlinputs as dli

path = """
/work/DATABASES
""".split()

dbpath = dli.find_file(path, "uw3-dew.db")

def fix(image):
    h, w = image.shape
    assert h==48
    assert w>0
    assert np.amax(image) < 1.5
    image = np.expand_dims(image, 2)
    return image

class Inputs(object):
    def __init__(self, **kw):
        self.normalize = True
        self.dbfix = True
        self.__dict__.update(kw)

    def training_data(self, table="train", **kw):
        self.__dict__.update(kw)
        data = dli.itsqlite(dbpath, table=table) | \
               dli.itmap(image=dli.pilreads) | \
               dli.itmap(image=fix)
        #data = dli.itshuffle(data, 5000)
        # data = dli.itmap(data, image=fix)
        #data = ocropus2.itfix(data)
        #normalizer = ocropus2.CenterNormalizer()
        #data = ocropus2.itmapper(data, input=normalizer.measure_and_normalize)
        return data

    def testing(self):
        return self.training_data(table="test")
