#def _patched_view_4d(*tensors):
#    output = []
#    for t in tensors:
#        assert t.dim() == 3
#        size = list(t.size())
#        size.insert(2, 1)
#        output += [t.contiguous().view(*size)]
#    return output
#
#import torch.nn._functions.conv
#
#torch.nn._functions.conv._view4d = _patched_view_4d

from pylab import *
import os
import glob
import random as pyr
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import ocrcodecs
import cctc
import lineest
import inputs

class Ignore(Exception):
    pass

# In[8]:
def asnd(x, torch_axes=None):
    """Convert torch/numpy to numpy."""
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, Variable):
        x = x.data
    if isinstance(x, (torch.cuda.FloatTensor, torch.cuda.DoubleTensor, torch.cuda.IntTensor)):
        x = x.cpu()
    assert isinstance(x, torch.Tensor)
    x = x.numpy()
    if torch_axes is not None:
        x = x.transpose(torch_axes)
    return x

# In[11]:
def novar(x):
    if isinstance(x, Variable):
        return x.data
    return x

def astorch(x, axes=None, single=True):
    """Convert torch/numpy to torch."""
    if isinstance(x, np.ndarray):
        if axes is not None:
            x = x.transpose(axes)
        if x.dtype == np.dtype("f"):
            return torch.FloatTensor(x)
        elif x.dtype == np.dtype("d"):
            if single:
                return torch.FloatTensor(x)
            else:
                return torch.DoubleTensor(x)
        elif x.dtype == np.dtype("i"):
            return torch.IntTensor(x)
        else:
            error("unknown dtype")
    return x

# In[13]:
def typeas(x, y):
    """Make x the same type as y, for numpy, torch, torch.cuda."""
    assert not isinstance(x, Variable)
    if isinstance(y, Variable):
        y = y.data
    if isinstance(y, np.ndarray):
        return asnd(x)
    if isinstance(x, np.ndarray):
        if isinstance(y, (torch.FloatTensor, torch.cuda.FloatTensor)):
            x = torch.FloatTensor(x)
        else:
            x = torch.DoubleTensor(x)
    return x.type_as(y)

# In[16]:
def one_sequence_softmax(x):
    """Compute softmax over a sequence; shape is (l, d)"""
    y = asnd(x)
    assert y.ndim==2, "%s: input should be (length, depth)" % y.shape
    l, d = y.shape
    y = amax(y, axis=1)[:, newaxis] -y
    y = clip(y, -80, 80)
    y = exp(y)
    y = y / sum(y, axis=1)[:, newaxis]
    return typeas(y, x)

def sequence_softmax(x):
    """Compute sotmax over a batch of sequences; shape is (b, l, d)."""
    y = asnd(x)
    assert y.ndim==3, "%s: input should be (batch, length, depth)" % y.shape
    for i in range(len(y)):
        y[i] = one_sequence_softmax(y[i])
    return typeas(y, x)

def is_normalized(x, eps=1e-3):
    """Check whether a batch of sequences (b, l, d) is normalized in d."""
    assert x.dim() == 3
    marginal = x.sum(2)
    return (marginal - 1.0).abs().lt(eps).all()

def ctc_align(prob, target):
    """Perform CTC alignment on torch sequence batches (using ocrolstm)"""
    prob_ = prob.cpu()
    target = target.cpu()
    b, l, d = prob.size()
    bt, lt, dt = target.size()
    assert bt==b, (bt, b)
    assert dt==d, (dt, d)
    assert is_normalized(prob), prob
    assert is_normalized(target), target
    result = torch.rand(1)
    cctc.ctc_align_targets_batch(result, prob_, target)
    return typeas(result, prob)

class Textline2Img(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
    def forward(self, seq):
        b, l, d = seq.size()
        return seq.view(b, 1, l, d)

class Img2Seq(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
    def forward(self, img):
        b, d, w, h = img.size()
        perm = img.permute(0, 2, 1, 3).contiguous()
        return perm.view(b, w, d*h)

class ImgMaxSeq(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
    def forward(self, img):
        # BDWH -> BDW -> BWD
        return img.max(3)[0].squeeze(3).permute(0, 2, 1).contiguous()

class ImgSumSeq(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
    def forward(self, img):
        # BDWH -> BDW -> BWD
        return img.sum(3)[0].squeeze(3).permute(0, 2, 1).contiguous()

class Lstm2Dto1D(nn.Module):
    """An LSTM that summarizes one dimension."""
    def __init__(self, ninput=None, noutput=None):
        nn.Module.__init__(self)
        self.ninput = ninput
        self.noutput = noutput
        self.lstm = nn.LSTM(ninput, noutput, 1, bidirectional=False)
    def forward(self, img, volatile=False):
        # BDWH -> HBWD -> HBsD
        b, d, w, h = img.size()
        seq = img.permute(3, 0, 2, 1).contiguous().view(h, b*w, d)
        bs = b*w
        h0 = Variable(typeas(torch.zeros(1, bs, self.noutput), img), volatile=volatile)
        c0 = Variable(typeas(torch.zeros(1, bs, self.noutput), img), volatile=volatile)
        # HBsD -> HBsD
        post_lstm, _ = self.lstm(seq, (h0, c0))
        # HBsD -> BsD -> BWD
        final = post_lstm.select(0, h-1).view(b, w, self.noutput)
        return final

class RowwiseLSTM(nn.Module):
    def __init__(self, ninput=None, noutput=None, ndir=2):
        nn.Module.__init__(self)
        self.ndir = ndir
        self.ninput = ninput
        self.noutput = noutput
        self.lstm = nn.LSTM(ninput, noutput, 1, bidirectional=self.ndir-1)
    def forward(self, img):
        volatile = not isinstance(img, Variable) or img.volatile
        b, d, h, w = img.size()
        # BDHW -> WHBD -> WLD
        seq = img.permute(3, 2, 0, 1).contiguous().view(w, h*b, d)
        # WLD
        h0 = typeas(torch.zeros(self.ndir, h*b, self.noutput), img)
        c0 = typeas(torch.zeros(self.ndir, h*b, self.noutput), img)
        h0 = Variable(h0, volatile=volatile)
        c0 = Variable(c0, volatile=volatile)
        seqresult, _ = self.lstm(seq, (h0, c0))
        # WLD' -> BD'HW
        result = seqresult.view(w, h, b, self.noutput*self.ndir).permute(2, 3, 1, 0)
        return result

class Lstm2D(nn.Module):
    """A 2D LSTM module."""
    def __init__(self, ninput=None, noutput=None, npre=None, nhidden=None, ndir=2, ksize=3):
        nn.Module.__init__(self)
        self.ndir = ndir
        npre = npre or noutput
        nhidden = nhidden or noutput
        self.sizes = (ninput, npre, nhidden, noutput)
        assert ksize%2==1
        padding = (ksize-1)//2
        self.conv = nn.Conv2d(ninput, npre, kernel_size=ksize, padding=padding)
        self.hlstm = RowwiseLSTM(npre, nhidden, ndir=ndir)
        self.vlstm = RowwiseLSTM(self.ndir*nhidden, noutput, ndir)
    def forward(self, img, volatile=False):
        ninput, npre, nhidden, noutput = self.sizes
        # BDHW
        filtered = self.conv(img)
        horiz = self.hlstm(filtered)
        horizT = horiz.permute(0, 1, 3, 2).contiguous()
        vert = self.vlstm(horizT)
        vertT = vert.permute(0, 1, 3, 2).contiguous()
        return vertT

class LstmLin(nn.Module):
    """A simple bidirectional LSTM with linear output mapping."""
    def __init__(self, ninput=None, nhidden=None, noutput=None, ndir=2):
        nn.Module.__init__(self)
        assert ninput is not None
        assert nhidden is not None
        assert noutput is not None
        self.ndir = 2
        self.ninput = ninput
        self.nhidden = nhidden
        self.nooutput = noutput
        self.layers = []
        self.lstm = nn.LSTM(ninput, nhidden, 1, bidirectional=self.ndir-1)
        self.conv = nn.Conv1d(ndir*nhidden, noutput, 1)
    def forward(self, seq, volatile=False):
        # BLD -> LBD
        bs, l, d = seq.size()
        assert d==self.ninput, seq.size()
        seq = seq.permute(1, 0, 2).contiguous()
        h0 = Variable(torch.zeros(self.ndir, bs, self.nhidden).type_as(novar(seq)),
                      volatile=volatile)
        c0 = Variable(torch.zeros(self.ndir, bs, self.nhidden).type_as(novar(seq)),
                      volatile=volatile)
        post_lstm, _ = self.lstm(seq, (h0, c0))
        assert post_lstm is not None
        # LBD -> BDL
        post_conv = self.conv(post_lstm.permute(1, 2, 0).contiguous())
        # BDL -> BLD
        return post_conv.permute(0, 2, 1).contiguous()

def conv_layers(sizes=[64, 64], k=3, mp=(1,2), mpk=2, relu=1e-3, bn=None, fmp=False):
    """Construct a set of conv layers programmatically based on some parameters."""
    if isinstance(mp, (int, float)):
        mp = (1, mp)
    layers = []
    nin = 1
    for nout in sizes:
        if nout < 0:
            nout = -nout
            layers += [Lstm2D(nin, nout)]
        else:
            layers += [nn.Conv2d(nin, nout, k, padding=(k-1)//2)]
            if relu>0:
                layers += [nn.LeakyReLU(relu)]
            else:
                layers += [nn.ReLU()]
            if bn is not None:
                assert isinstance(bn, float)
                layers += [nn.BatchNorm2d(nout, momentum=bn)]
        if not fmp:
            layers += [nn.MaxPool2d(kernel_size=mpk, stride=mp)]
        else:
            layers += [nn.FractionalMaxPool2d(kernel_size=mpk, output_ratio=(1.0/mp[0], 1.0/mp[1]))]
        nin = nout
    return layers

def make_projector(project):
    """Given the name of a projection method for the H dimension, turns a BDWH image
    into a BLD sequence, with DH combined into the new depth. Projection methods
    include concat, max, sum, and lstm:n, with n being the new depth."""
    if project is None or project=="cocnat":
        return Img2Seq()
    elif project == "max":
        return ImgMaxSeq()
    elif project == "sum":
        return ImgSumSeq()
    elif project.startswith("lstm:"):
        _, n = project.split(":")
        return Lstm2Dto1D(int(n))
    else:
        raise Exception("unknown projection: "+project)

def wrapup(layers):
    """Wraps up a list of layers into an nn.Sequential if necessary."""
    assert isinstance(layers, list)
    if len(layers)==1:
        return layers[0]
    else:
        return nn.Sequential(*layers)

def size_tester(model, shape):
    """Given a layer or a list of layers, runs a random input of the given shape
    through it and returns the output shape."""
    if isinstance(model, list):
        model = nn.Sequential(*model)
    test = torch.rand(*shape)
    test_output = model(Variable(test, volatile=True))
    print "# preproc", test.size(), "->", test_output.size()
    return test_output.size()

def make_lstm(ninput=48, nhidden=100, noutput=None, project=None, **kw):
    """Builds an LSTM-based recognizer, possibly with convolutional preprocessing."""
    layers = []
    d = ninput
    if "sizes" in kw:
        layers += [Textline2Img()]
        layers += conv_layers(**kw)
        layers += [make_projector(project)]
        b, l, d = size_tester(layers, (1, 400, d))
        assert b==1, (b, l, d)
    layers += [LstmLin(ninput=d, nhidden=nhidden, noutput=noutput)]
    return wrapup(layers)


# In[26]:
def ctc_loss(logits, target):
    """A CTC loss function for BLD sequence training."""
    assert logits.is_contiguous()
    assert target.is_contiguous()
    probs = sequence_softmax(logits)
    aligned = ctc_align(probs, target)
    assert aligned.size()==probs.size(), (aligned.size(), probs.size())
    deltas = aligned - probs
    logits.backward(deltas.contiguous())
    return deltas, aligned

def mock_ctc_loss(logits, target):
    deltas = torch.randn(*logits.size()) * 0.001
    logits.backward(deltas.cuda())

# In[30]:

codecs = dict(
    ascii=ocrcodecs.ascii_codec()
)

def maketarget(s, codec="ascii"):
    """Turn a string into an LD target."""
    assert isinstance(s, (str, unicode))
    codec = codecs.get(codec, codec)
    codes = codec.encode(s)
    n = codec.size()
    target = astorch(ocrcodecs.make_target(codes, n))
    return target

def transcribe(probs, codec="ascii"):
    codes = ocrcodecs.translate_back(asnd(probs))
    codec = codecs.get(codec, codec)
    s = codec.decode(codes)
    return "".join(s)

def makeseq(image):
    """Turn an image into an LD sequence."""
    assert isinstance(image, np.ndarray), type(image)
    if image.ndim==3 and image.shape[2]==3:
        image = np.mean(image, 2)
    assert image.ndim==2
    return astorch(image.T)

def makebatch(images, for_target=False):
    """Given a list of LD sequences, make a BLD batch tensor."""
    assert isinstance(images, list), type(images)
    assert isinstance(images[0], torch.FloatTensor), images
    assert images[0].dim() == 2, images[0].dim()
    l, d = amax(array([img.size() for img in images], 'i'), axis=0)
    ibatch = torch.zeros(len(images), int(l), int(d))
    if for_target:
        ibatch[:, :, 0] = 1.0
    for i, image in enumerate(images):
        l, d = image.size()
        ibatch[i, :l, :d] = image
    return ibatch

#import memory_profiler
class SimpleOCR(object):
    """Perform simple OCRopus-like 1D LSTM OCR."""
    def __init__(self, model, lr=1e-4, momentum=0.9, ninput=48, builder=None,
                 cuda=True, codec=None, mname=None):
        if codec is None: codec = ocrcodecs.ascii_codec()
        self.codec = codec
        self.noutput = self.codec.size()
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.cuda = cuda
        self.normalizer = lineest.CenterNormalizer()
        self.setup_model()
    def setup_model(self):
        if self.model is None: return
        if self.cuda: self.model.cuda()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        if not hasattr(self.model, "META"):
            self.model.META = dict(ntrain=0)
        self.ntrain = self.model.META["ntrain"]
    def gpu(self):
        self.cuda = True
        self.setup_model()
    def cpu(self):
        self.cuda = False
        self.setup_model()
    def set_lr(self, lr, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
    def save(self, fname):
        """Save this model."""
        if "%0" in fname:
            fname = fname % self.ntrain
        print "# saving", fname
        if not hasattr(self.model, "META"):
            self.model.META = {}
        self.model.META["ntrain"] = self.ntrain
        torch.save(self.model, fname)
    def load(self, fname):
        """Load this model."""
        self.model = torch.load(fname)
        if not hasattr(self.model, "META"):
            self.model.META = {}
        self.ntrain = self.model.META.get("ntrain", 0)
        self.setup_model()
    def C(self, x):
        """Convert to cuda if required."""
        if self.cuda: return x.cuda()
        return x
    def pad_length(self, input, r=20):
        b, f, l, d = input.shape
        assert f==1, "must have #features == 1"
        result = np.zeros((b, f, l+r, d))
        result[:, :, :l, :] = input
        return result
    def train_batch(self, input, target):
        """Train a BLD input batch against a BLD target batch."""
        assert input.shape[0] == target.shape[0]
        input = self.pad_length(input)
        input = astorch(input)
        target = astorch(target)
        self.input = torch.FloatTensor()
        self.logits = torch.FloatTensor()
        self.aligned = torch.FloatTensor()
        self.target = torch.FloatTensor()
        try:
            self.ntrain += input.size(0)
            self.input = Variable(self.C(input))
            self.target = self.C(target)
            # print input.size(), input.min(), input.max()
            # print target.size(), target.min(), target.max()
            output = self.model.forward(self.input)
            output = output.permute(0, 2, 1).contiguous()
            self.logits = output
            self.probs = sequence_softmax(self.logits)
            self.optimizer.zero_grad()
            _, self.aligned = ctc_loss(self.logits, self.target)
            self.optimizer.step()
        except Ignore:
            print "input", input.size()
            print "logits", self.logits.size()
            print "aligned", self.aligned.size()
            print "target", self.target.size()
    def train(self, image, transcript):
        """Train a single image and its transcript using LSTM+CTC."""
        input = makeseq(image).unsqueeze(0)
        target = maketarget(transcript).unsqueeze(0)
        self.train_batch(input, target)
    def train_multi(self, images, transcripts):
        """Train a list of images and their transcripts using LSTM+CTC."""
        images = makebatch([makeseq(x) for x in images])
        targets = makebatch([maketarget(x) for x in transcripts], for_target=True)
        self.train_batch(images, targets)
    def predict_batch(self, input):
        """Train a BLD input batch against a BLD target batch."""
        input = self.pad_length(input)
        input = astorch(input)
        input = Variable(self.C(input), volatile=True)
        output = self.model.forward(input)
        output = output.permute(0, 2, 1).contiguous()
        probs = novar(sequence_softmax(output))
        result = [transcribe(probs[i]) for i in range(len(probs))]
        return result
    def recognize(self, lines):
        data = inputs.itlines(lines)
        data = inputs.itmapper(data, input=self.normalizer.measure_and_normalize)
        data = inputs.itimbatched(data, 20)
        data = inputs.itlinebatcher(data)
        result = []
        for batch in data:
            input = batch["input"]
            if input.ndim==3:
                input = np.expand_dims(input, 1)
            try:
                predicted = self.predict_batch(input)
            except RuntimeError:
                predicted = ["_"] * len(input)
            for i in range(len(predicted)):
                result.append((batch["id"][i], predicted[i]))
        return [transcript for i, transcript in sorted(result)]
