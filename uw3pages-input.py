import numpy as np
import dlinputs as dli

import random as pyr
from math import pi, cos, sin

import numpy as np
import scipy.ndimage as ndi

path = """
./pages
/work/OCR/uw3-pages
""".split()

pages = dli.find_directory(path)


params = dict(translation=0.03, rotation=1.0, scale=0.03, aniso=0.03)

def random_trs(translation=0.05, rotation=2.0, scale=0.1, aniso=0.1):
    if not isinstance(translation, (tuple, list)):
        translation = (-translation, translation)
    if not isinstance(rotation, (tuple, list)):
        rotation = (-rotation, rotation)
    if not isinstance(scale, (tuple, list)):
        scale = (-scale, scale)
    if not isinstance(aniso, (tuple, list)):
        aniso = (-aniso, aniso)
    dx = pyr.uniform(*translation)
    dy = pyr.uniform(*translation)
    alpha = pyr.uniform(*rotation)
    alpha = alpha * pi / 180.0
    scale = 10**pyr.uniform(*scale)
    aniso = 10**pyr.uniform(*aniso)
    c = cos(alpha)
    s = sin(alpha)
    #print "\t", (dx, dy), alpha, scale, aniso
    sm = np.array([[scale / aniso, 0], [0, scale * aniso]], 'f')
    m = np.array([[c, -s], [s, c]], 'f')
    m = np.dot(sm, m)

    def f(image, order=1):
        w, h = image.shape
        c = np.array([w, h]) / 2.0
        d = c - np.dot(m, c) + np.array([dx * w, dy * h])
        return ndi.affine_transform(image, m, offset=d, order=order)

    return f

def degrade(sample):
    f = random_trs(**params)
    result = dict(sample)
    if params is not None:
        for k in "input output mask".split():
            if k in sample:
                result[k] = f(sample[k])
    return sample

def make_mask(sample):
    result = dict(sample)
    result["mask"] = ndi.maximum_filter(result["output"], 20)
    return result

class Inputs(object):
    def training_data(self):
        return (dli.itdirtree(pages, ["png", "lines.png"])
                | dli.itren(input="png", output="lines.png")
                | dli.itmap(input=dli.pilgray, output=dli.pilgray)
                | dli.ittransform(make_mask)
                | dli.ittransform(degrade)
                )
