import os
import sqlite3
import math
import numpy as np
import ocrnet
from PIL import Image
from StringIO import StringIO
import glob
import os.path
import codecs
import random as pyr
import re
import pylab
import ocrcodecs
import scipy.ndimage as ndi
import lineest

verbose = True

def image(x, normalize=True, gray=False):
    """Convert a string to an image.

    Commonly used as a decoder. This will ensure that
    the output is rank 3 (or rank 1 if gray=True)."""
    image = np.array(Image.open(StringIO(x)))
    assert isinstance(image, np.ndarray)
    if normalize: image = image / 255.0
    if gray:
        if image.ndim == 3:
            image = np.mean(image, 2)
        image = image.reshape(image.shape + (1,))
        assert image.ndim == 3 and image.shape[2] == 1
        return image
    else:
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=2)
        if image.shape[2] > 3:
            image = image[...,:3]
        assert image.ndim == 3 and image.shape[2] == 3
        return image

def isimage(v):
    if v[:4]=="\x89PNG": return True
    if v[:2]=="\377\330": return True
    if v[:6]=="IHDR\r\n": return True
    return False

def auto_decode(sample, normalize=True, gray=False):
    for k, v in sample.items():
        if k == "transcript":
            sample[k] = str(v)
        elif isinstance(v, buffer):
            v = str(v)
        elif isinstance(v, str) and isimage(v):
            try:
                sample[k] = image(v, normalize=normalize, gray=gray)
            except ValueError:
                pass
    return sample

def fiximage(image, transcript, minlen=4, maxheight=48):
    """The UW3 dataset contains some vertical text lines; rotate these
    to horizontal. Also, fix lines that are too tall"""
    if image.ndim==3:
        image = np.mean(image, 2)
    image = lineest.autocrop(image)
    h, w = image.shape
    if h > w and len(transcript)>minlen:
        image = image[::-1].T
    h, w = image.shape
    if h > maxheight:
        z = maxheight * 1.0 / h
        image = ndi.zoom(image, [z, z], order=1)
    return image

def itfix(data):
    for sample in data:
        sample["input"] = fiximage(sample["input"], sample["transcript"])
        yield sample

default_renames = {
    "rowid": "id",
    "index": "id",
    "inx": "id",
    "image": "input",
    "images": "input",
    "inputs": "input",
    "output": "target",
    "outputs": "target",
    "cls" : "target",
    "targets": "target",
}

def itsqlite(dbfile, table="train", nepochs=1, cols="*",
             decoder=auto_decode, fields=None, renames=default_renames, extra=""):
    assert "," not in table
    if "::" in dbfile:
        dbfile, table = dbfile.rsplit("::", 1)
    assert os.path.exists(dbfile)
    db = sqlite3.connect(dbfile)
    if fields is not None:
        fields = fields.split(",")
    for epoch in xrange(nepochs):
        if verbose:
            print "# epoch", epoch, "of", nepochs, "from", dbfile, table
        count = 0
        c = db.cursor()
        sql = "select %s from %s %s" % (cols, table, extra)
        for row in c.execute(sql):
            sample = {}
            if fields is not None:
                assert len(fields) == len(row)
                for i, r in enumerate(row):
                    sample[fields[i]] = r
            else:
                cs = [x[0] for x in c.description]
                for i, col in enumerate(cs):
                    value = row[i]
                    if isinstance(value, buffer):
                        value = str(value)
                    ocol = renames.get(col, col)
                    assert ocol not in sample
                    sample[ocol] = value
            if decoder: sample = decoder(sample)
            if "transcript" in sample:
                assert isinstance(sample["transcript"], (str, unicode)), decoder
            count = count+1
            yield sample
        c.close()
        del c

def itbook(dname, epochs=1000):
    fnames = glob.glob(dname+"/????/??????.gt.txt")
    for epoch in range(epochs):
        # pyr.shuffle(fnames)
        fnames.sort()
        for fname in fnames:
            base = re.sub(".gt.txt$", "", fname)
            if not os.path.exists(base+".dew.png"): continue
            image = pylab.imread(base+".dew.png")
            if image.ndim==3: image = np.mean(image, 2)
            image -= np.amin(image)
            image /= np.amax(image)
            image = 1.0 - image
            with codecs.open(fname, "r", "utf-8") as stream:
                transcript = stream.read().strip()
            yield dict(input=image, transcript=transcript)

def itinfinite(sample):
    """Repeat the same sample over and over again (for testing)."""
    while True:
        yield sample

def itmaxsize(sample, h, w):
    for sample in data:
        shape = sample["input"].shape
        if shape[0] >= h: continue
        if shape[1] >= w: continue
        yield sample

def image2seq(image):
    """Turn a WH or WH3 image into an LD sequence."""
    if image.ndim==3 and image.shape[2]==3:
        image = np.mean(image, 2)
    assert image.ndim==2
    return ocrnet.astorch(image.T)

def itmapper(data, **keys):
    """Map the fields in each sample using name=f arguments."""
    for sample in data:
        sample = sample.copy()
        for k, f in keys.items():
            sample[k] = f(sample[k])
        yield sample

def itimbatched(data, batchsize=5, scale=1.8, seqkey="input"):
    """List-batch input samples into similar sized batches."""
    buckets = {}
    for sample in data:
        seq = sample[seqkey]
        d, l = seq.shape[:2]
        r = int(math.floor(math.log(l) / math.log(scale)))
        batched = buckets.get(r, {})
        for k, v in sample.items():
            if k in batched:
                batched[k].append(v)
            else:
                batched[k] = [v]
        if len(batched[seqkey]) >= batchsize:
            batched["_bucket"] = r
            yield batched
            batched = {}
        buckets[r] = batched
    for r, batched in buckets.items():
        if batched == {}: continue
        batched["_bucket"] = r
        yield batched

def makeseq(image):
    """Turn an image into an LD sequence."""
    assert isinstance(image, np.ndarray), type(image)
    if image.ndim==3 and image.shape[2]==3:
        image = np.mean(image, 2)
    assert image.ndim==2
    return image.T

def makebatch(images, for_target=False, l_pad=0, d_pad=0):
    """Given a list of LD sequences, make a BLD batch tensor."""
    assert isinstance(images, list), type(images)
    assert isinstance(images[0], np.ndarray), images
    assert images[0].ndim==2, images[0].ndim
    l, d = np.amax(np.array([img.shape for img in images], 'i'), axis=0)
    l += l_pad
    d += d_pad
    ibatch = np.zeros([len(images), int(l), int(d)])
    if for_target:
        ibatch[:, :, 0] = 1.0
    for i, image in enumerate(images):
        l, d = image.shape
        ibatch[i, :l, :d] = image
    return ibatch

def images2batch(images):
    images = [makeseq(x) for x in images]
    return makebatch(images)

def maketarget(s, codec="ascii"):
    """Turn a string into an LD target."""
    assert isinstance(s, (str, unicode)), (type(s), s)
    codec = ocrnet.codecs.get(codec, codec)
    codes = codec.encode(s)
    n = codec.size()
    return ocrcodecs.make_target(codes, n)

def transcripts2batch(transcripts, codec="ascii"):
    targets = [maketarget(s) for s in transcripts]
    return makebatch(targets, for_target=True)

def itlinebatcher(data):
    for sample in data:
        sample = sample.copy()
        sample["input"] = images2batch(sample["input"])
        sample["target"] = transcripts2batch(sample["transcript"])
        yield sample

def itshuffle(data, bufsize=1000):
    """Shuffle the data in the stream.

    This uses a buffer of size `bufsize`. Shuffling at
    startup is less random; this is traded off against
    yielding samples quickly."""
    buf = []
    for sample in data:
        if len(buf) < bufsize:
            buf.append(data.next())
        k = pyr.randint(0, len(buf)-1)
        sample, buf[k] = buf[k], sample
        yield sample

def itlines(images):
    for i, image in enumerate(images):
        if image.ndim==3:
            image = np.mean(image[:,:,:3], 2)
        sample = dict(input=image, key=i, id=i, transcript="")
        yield sample
