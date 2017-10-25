from __future__ import print_function

import numpy as np
from collections import defaultdict
import unicodedata
from scipy.ndimage import measurements

def make_target(cs,nc):
    """Given a list of target classes `cs` and a total
    maximum number of classes, compute an array that has
    a `1` in each column and time step corresponding to the
    target class."""
    result = np.zeros((2*len(cs)+1,nc))
    for i,j in enumerate(cs):
        result[2*i,0] = 1.0
        result[2*i+1,j] = 1.0
    result[-1,0] = 1.0
    return result

def translate_back0(outputs,threshold=0.25):
    """Simple code for translating output from a classifier
    back into a list of classes. TODO/ATTENTION: this can
    probably be improved."""
    ms = amax(outputs,axis=1)
    cs = argmax(outputs,axis=1)
    cs[ms<threshold*amax(outputs)] = 0
    result = []
    for i in range(1,len(cs)):
        if cs[i]!=cs[i-1]:
            if cs[i]!=0:
                result.append(cs[i])
    return result

def translate_back(outputs,threshold=0.7,pos=0):
    """Translate back. Thresholds on class 0, then assigns the maximum class to
    each region. ``pos`` determines the depth of character information returned:
        * `pos=0`: Return list of recognized characters
        * `pos=1`: Return list of position-character tuples
        * `pos=2`: Return list of character-probability tuples
     """
    labels,n = measurements.label(outputs[:,0]<threshold)
    mask = np.tile(labels.reshape(-1,1),(1,outputs.shape[1]))
    maxima = measurements.maximum_position(outputs,mask,np.arange(1,np.amax(mask)+1))
    if pos==1: return maxima # include character position
    if pos==2: return [(c, outputs[r,c]) for (r,c) in maxima] # include character probabilities
    return [c for (r,c) in maxima] # only recognized characters

class Codec:
    """Translate between integer codes and characters."""
    def init(self,charset):
        charset = sorted(list(set(charset)))
        self.code2char = {}
        self.char2code = {}
        for code,char in enumerate(charset):
            self.code2char[code] = char
            self.char2code[char] = code
        return self
    def size(self):
        """The total number of codes (use this for the number of output
        classes when training a classifier."""
        return len(list(self.code2char.keys()))
    def encode(self,s):
        "Encode the string `s` into a code sequence."
        # tab = self.char2code
        dflt = self.char2code["~"]
        return [self.char2code.get(c,dflt) for c in s]
    def decode(self,l):
        "Decode a code sequence into a string."
        s = [self.code2char.get(c,"~") for c in l]
        return s

ascii_labels = [""," ","~"] + [unichr(x) for x in range(33,126)]

def ascii_codec():
    "Create a codec containing just ASCII characters."
    return Codec().init(ascii_labels)

