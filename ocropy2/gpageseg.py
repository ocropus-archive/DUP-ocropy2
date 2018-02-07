import numpy
import pylab
from numpy import *
from pylab import *
from pylab import mean
from scipy.ndimage import filters, interpolation, measurements, morphology
from scipy.ndimage.morphology import *
import logging
logger = logging.getLogger()

################################################################
# utilities for lists of slices, treating them like rectangles
################################################################

class Struct(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __call__(self, **kw):
        result = Struct(**self.__dict__)
        result.__dict__.update(kw)
        return result

default_args = Struct(nocheck=False,
                      minscale=12,
                      maxlines=300,
                      scale=0.0,
                      hscale=1.0,
                      vscale=1.0,
                      threshold=0.2,
                      noise=8,
                      usegauss=False,
                      maxseps=0,
                      sepwiden=10,
                      blackseps=False,
                      maxcolseps=3,
                      csminheight=10,
                      csminaspect=1.1,
                      pad=3,
                      expand=3,
                      quiet=False,
                      debug=False)


def is_slices(u):
    for s in u:
        if type(s) != slice:
            return False
    return True


def dims(s):
    """List of dimensions of the slice list."""
    return tuple([x.stop - x.start for x in s])


def dim0(s):
    """Dimension of the slice list for dimension 0."""
    return s[0].stop - s[0].start


def dim1(s):
    """Dimension of the slice list for dimension 1."""
    return s[1].stop - s[1].start


def box(r0, r1, c0, c1):
    return (slice(r0, r1), slice(c0, c1))


def start(u):
    return tuple([x.start for x in u])


def stop(u):
    return tuple([x.stop for x in u])


def bounds(a):
    """Return a list of slices corresponding to the array bounds."""
    return tuple([slice(0, a.shape[i]) for i in range(a.ndim)])


def volume(a):
    """Return the area of the slice list."""
    return numpy.prod([max(x.stop - x.start, 0) for x in a])


def empty(a):
    """Test whether the slice is empty."""
    return a is None or volume(a) == 0


def shift(u, offsets, scale=1):
    u = list(u)
    for i in range(len(offsets)):
        u[i] = slice(u[i].start + scale * offsets[i],
                     u[i].stop + scale * offsets[i])
    return tuple(u)

# These are special because they only operate on the first two
# dimensions.  That's useful for RGB images.


def area(a):
    """Return the area of the slice list (ignores anything past a[:2]."""
    return numpy.prod([max(x.stop - x.start, 0) for x in a[:2]])


def aspect(a):
    return height(a) * 1.0 / width(a)

# Geometric operations.


def pad(u, d):
    """Pad the slice list by the given amount."""
    return tuple([slice(u[i].start - d, u[i].stop + d) for i in range(len(u))])


def intersect(u, v):
    """Compute the intersection of the two slice lists."""
    if u is None:
        return v
    if v is None:
        return u
    return tuple([slice(max(u[i].start, v[i].start), min(u[i].stop, v[i].stop)) for i in range(len(u))])


def xoverlap(u, v):
    return max(0, min(u[1].stop, v[1].stop) - max(u[1].start, v[1].start))


def yoverlap(u, v):
    return max(0, min(u[0].stop, v[0].stop) - max(u[0].start, v[0].start))


def xcenter(s):
    return mean([s[1].stop, s[1].start])


def ycenter(s):
    return mean([s[0].stop, s[0].start])


def center(s):
    return (ycenter(s), xcenter(s))


def width(s):
    return s[1].stop - s[1].start


def height(s):
    return s[0].stop - s[0].start


def cut(image, box, margin=0, bg=0, dtype=None):
    """Cut out a region given by a box (row0,col0,row1,col1),
    with an optional margin."""
    assert len(box) == 2 and is_slices(box)
    if dtype is None:
        dtype = image.dtype
    if image.ndim == 3:
        assert image.shape[2] == 3
        result = [cut(image[:, :, i], box, margin, bg, dtype)
                  for i in range(image.shape[2])]
        result = numpy.transpose(result, [1, 2, 0])
        return result
    elif image.ndim == 2:
        box = pad(box, margin)
        cbox = intersect(box, bounds(image))
        if empty(cbox):
            result = numpy.empty(dims(box), dtype=dtype)
            result.ravel()[:] = bg
            return result
        cimage = image[cbox]
        if cbox == box:
            return cimage
        else:
            if dtype is None:
                dtype = image.dtype
            result = numpy.empty(dims(box), dtype=dtype)
            result.ravel()[:] = bg
            moved = shift(cbox, start(box), -1)
            result[moved] = cimage
            return result
    else:
        raise Exception("not implemented for ndim!=2 or ndim!=3")


def label(image, **kw):
    """Redefine the scipy.ndimage.measurements.label function to
    work with a wider range of data types.  The default function
    is inconsistent about the data types it accepts on different
    platforms."""
    try:
        return measurements.label(image, **kw)
    except:
        pass
    types = ["int32", "uint32", "int64", "unit64", "int16", "uint16"]
    for t in types:
        try:
            return measurements.label(array(image, dtype=t), **kw)
        except:
            pass
    # let it raise the same exception as before
    return measurements.label(image, **kw)


def find_objects(image, **kw):
    """Redefine the scipy.ndimage.measurements.find_objects function to
    work with a wider range of data types.  The default function
    is inconsistent about the data types it accepts on different
    platforms."""
    try:
        return measurements.find_objects(image, **kw)
    except:
        pass
    types = ["int32", "uint32", "int64", "unit64", "int16", "uint16"]
    for t in types:
        try:
            return measurements.find_objects(array(image, dtype=t), **kw)
        except:
            pass
    # let it raise the same exception as before
    return measurements.find_objects(image, **kw)


def check_binary(image):
    assert image.dtype == 'B' or image.dtype == 'i' or image.dtype == dtype('bool'),\
        "array should be binary, is %s %s" % (image.dtype, image.shape)
    assert amin(image) >= 0 and amax(image) <= 1,\
        "array should be binary, has values %g to %g" % (
            amin(image), amax(image))


def norm_max(a):
    return a/amax(a)


def remove_noise(line, minsize=8):
    """Remove small pixels from an image."""
    if minsize==0: return line
    bin = (line>0.5*amax(line))
    labels,n = label(bin)
    sums = measurements.sum(bin,labels,range(n+1))
    sums = sums[labels]
    good = minimum(bin,1-(sums>0)*(sums<minsize))
    return good


def r_dilation(image, size, origin=0):
    """Dilation with rectangular structuring element using maximum_filter"""
    return filters.maximum_filter(image, size, origin=origin)


def r_erosion(image, size, origin=0):
    """Erosion with rectangular structuring element using maximum_filter"""
    return filters.minimum_filter(image, size, origin=origin)


def rb_dilation(image, size, origin=0):
    """Binary dilation using linear filters."""
    output = zeros(image.shape, 'f')
    filters.uniform_filter(image, size, output=output,
                           origin=origin, mode='constant', cval=0)
    return array(output > 0, 'i')


def rb_erosion(image, size, origin=0):
    """Binary erosion using linear filters."""
    output = zeros(image.shape, 'f')
    filters.uniform_filter(image, size, output=output,
                           origin=origin, mode='constant', cval=1)
    return array(output == 1, 'i')


def rb_opening(image, size, origin=0):
    """Binary opening using linear filters."""
    image = rb_erosion(image, size, origin=origin)
    return rb_dilation(image, size, origin=origin)


def rb_closing(image, size, origin=0):
    """Binary closing using linear filters."""
    image = rb_dilation(image, size, origin=origin)
    return rb_erosion(image, size, origin=origin)


def showlabels(x, n=7):
    pylab.imshow(where(x > 0, x % n + 1, 0), cmap=pylab.cm.gist_stern)


def spread_labels(labels, maxdist=9999999):
    """Spread the given labels to the background"""
    distances, features = morphology.distance_transform_edt(
        labels == 0, return_distances=1, return_indices=1)
    indexes = features[0] * labels.shape[1] + features[1]
    spread = labels.ravel()[indexes.ravel()].reshape(*labels.shape)
    spread *= (distances < maxdist)
    return spread


def keep_marked(image, markers):
    """Given a marker image, keep only the connected components
    that overlap the markers."""
    labels, _ = label(image)
    marked = unique(labels * (markers != 0))
    kept = in1d(labels.ravel(), marked)
    return (image != 0) * kept.reshape(*labels.shape)


def correspondences(labels1, labels2):
    """Given two labeled images, compute an array giving the correspondences
    between labels in the two images."""
    q = 100000
    assert amin(labels1) >= 0 and amin(labels2) >= 0
    assert amax(labels2) < q
    combo = labels1 * q + labels2
    result = unique(combo)
    result = array([result // q, result % q])
    return result


def propagate_labels_simple(regions, labels):
    """Given an image and a set of labels, apply the labels
    to all the regions in the image that overlap a label."""
    rlabels, _ = label(regions)
    cors = correspondences(rlabels, labels)
    outputs = zeros(amax(rlabels) + 1, 'i')
    for o, i in cors.T:
        outputs[o] = i
    outputs[0] = 0
    return outputs[rlabels]


def propagate_labels(image, labels, conflict=0):
    """Given an image and a set of labels, apply the labels
    to all the regions in the image that overlap a label.
    Assign the value `conflict` to any labels that have a conflict."""
    rlabels, _ = label(image)
    cors = correspondences(rlabels, labels)
    outputs = zeros(amax(rlabels) + 1, 'i')
    oops = -(1 << 30)
    for o, i in cors.T:
        if outputs[o] != 0:
            outputs[o] = oops
        else:
            outputs[o] = i
    outputs[outputs == oops] = conflict
    outputs[0] = 0
    return outputs[rlabels]


def select_regions(binary, f, min=0, nbest=100000):
    """Given a scoring function f over slice tuples (as returned by
    find_objects), keeps at most nbest regions whose scores is higher
    than min."""
    labels, n = label(binary)
    objects = find_objects(labels)
    scores = [f(o) for o in objects]
    best = argsort(scores)
    keep = zeros(len(objects) + 1, 'i')
    if nbest > 0:
        for i in best[-nbest:]:
            if scores[i] <= min:
                continue
            keep[i + 1] = 1
    # print scores,best[-nbest:],keep
    # print sorted(list(set(labels.ravel())))
    # print sorted(list(set(keep[labels].ravel())))
    return keep[labels]


def all_neighbors(image):
    """Given an image with labels, find all pairs of labels
    that are directly neighboring each other."""
    q = 100000
    assert amax(image) < q
    assert amin(image) >= 0
    u = unique(q * image + roll(image, 1, 0))
    d = unique(q * image + roll(image, -1, 0))
    l = unique(q * image + roll(image, 1, 1))
    r = unique(q * image + roll(image, -1, 1))
    all = unique(r_[u, d, l, r])
    all = c_[all // q, all % q]
    all = unique(array([sorted(x) for x in all]))
    return all

################################################################
# Iterate through the regions of a color image.
################################################################


def renumber_labels_ordered(a, correspondence=0):
    """Renumber the labels of the input array in numerical order so
    that they are arranged from 1...N"""
    assert amin(a) >= 0
    assert amax(a) <= 2**25
    labels = sorted(unique(ravel(a)))
    renum = zeros(amax(labels) + 1, dtype='i')
    renum[labels] = arange(len(labels), dtype='i')
    if correspondence:
        return renum[a], labels
    else:
        return renum[a]


def renumber_labels(a):
    """Alias for renumber_labels_ordered"""
    return renumber_labels_ordered(a)


def pyargsort(seq, cmp=cmp, key=lambda x: x):
    """Like numpy's argsort, but using the builtin Python sorting
    function.  Takes an optional cmp."""
    return sorted(range(len(seq)), key=lambda x: key(seq.__getitem__(x)), cmp=cmp)


def renumber_by_xcenter(seg):
    """Given a segmentation (as a color image), change the labels
    assigned to each region such that when the labels are considered
    in ascending sequence, the x-centers of their bounding boxes
    are non-decreasing.  This is used for sorting the components
    of a segmented text line into left-to-right reading order."""
    objects = [(slice(0, 0), slice(0, 0))] + find_objects(seg)

    def xc(o):
        # if some labels of the segmentation are missing, we
        # return a very large xcenter, which will move them all
        # the way to the right (they don't show up in the final
        # segmentation anyway)
        if o is None:
            return 999999
        return mean((o[1].start, o[1].stop))
    xs = array([xc(o) for o in objects])
    order = argsort(xs)
    segmap = zeros(amax(seg) + 1, 'i')
    for i, j in enumerate(order):
        segmap[j] = i
    return segmap[seg]


def ordered_by_xcenter(seg):
    """Verify that the labels of a segmentation are ordered
    spatially (as determined by the x-center of their bounding
    boxes) in left-to-right reading order."""
    objects = [(slice(0, 0), slice(0, 0))] + find_objects(seg)

    def xc(o): return mean((o[1].start, o[1].stop))
    xs = array([xc(o) for o in objects])
    for i in range(1, len(xs)):
        if xs[i - 1] > xs[i]:
            return 0
    return 1


def B(a):
    if a.dtype == dtype('B'):
        return a
    return array(a, 'B')


class record:
    def __init__(self, **kw): self.__dict__.update(kw)


def blackout_images(image, ticlass):
    """Takes a page image and a ticlass text/image classification image and replaces
    all regions tagged as 'image' with rectangles in the page image.  The page image
    is modified in place.  All images are iulib arrays."""
    rgb = ocropy.intarray()
    ticlass.textImageProbabilities(rgb, image)
    r = ocropy.bytearray()
    g = ocropy.bytearray()
    b = ocropy.bytearray()
    ocropy.unpack_rgb(r, g, b, rgb)
    components = ocropy.intarray()
    components.copy(g)
    n = ocropy.label_components(components)
    print("[note] number of image regions", n)
    tirects = ocropy.rectarray()
    ocropy.bounding_boxes(tirects, components)
    for i in range(1, tirects.length()):
        r = tirects.at(i)
        ocropy.fill_rect(image, r, 0)
        r.pad_by(-5, -5)
        ocropy.fill_rect(image, r, 255)


def binary_objects(binary):
    labels, n = label(binary)
    objects = find_objects(labels)
    return objects


def estimate_scale(binary):
    objects = binary_objects(binary)
    bysize = sorted(objects, key=area)
    scalemap = zeros(binary.shape)
    for o in bysize:
        if amax(scalemap[o]) > 0:
            continue
        scalemap[o] = area(o)**0.5
    scale = median(scalemap[(scalemap > 3) & (scalemap < 100)])
    return scale


def compute_boxmap(binary, scale, threshold=(.5, 4), dtype='i'):
    objects = binary_objects(binary)
    bysize = sorted(objects, key=area)
    boxmap = zeros(binary.shape, dtype)
    for o in bysize:
        if area(o)**.5 < threshold[0] * scale:
            continue
        if area(o)**.5 > threshold[1] * scale:
            continue
        boxmap[o] = 1
    return boxmap


def compute_lines(segmentation, scale):
    """Given a line segmentation map, computes a list
    of tuples consisting of 2D slices and masked images."""
    lobjects = find_objects(segmentation)
    lines = []
    for i, o in enumerate(lobjects):
        if o is None:
            continue
        if dim1(o) < 2 * scale or dim0(o) < scale:
            continue
        mask = (segmentation[o] == i + 1)
        if amax(mask) == 0:
            continue
        result = record()
        result.label = i + 1
        result.bounds = o
        result.mask = mask
        lines.append(result)
    return lines


def pad_image(image, d, cval=inf):
    result = ones(array(image.shape) + 2 * d)
    result[:, :] = amax(image) if cval == inf else cval
    result[d:-d, d:-d] = image
    return result


def extract(image, y0, x0, y1, x1, mode='nearest', cval=0):
    h, w = image.shape
    ch, cw = y1 - y0, x1 - x0
    y, x = clip(y0, 0, max(h - ch, 0)), clip(x0, 0, max(w - cw, 0))
    sub = image[y:y + ch, x:x + cw]
    # print("extract", image.dtype, image.shape)
    try:
        r = interpolation.shift(sub, (y - y0, x - x0),
                                mode=mode, cval=cval, order=0)
        if cw > w or ch > h:
            pady0, padx0 = max(-y0, 0), max(-x0, 0)
            r = interpolation.affine_transform(r, eye(2), offset=(
                pady0, padx0), cval=1, output_shape=(ch, cw))
        return r

    except RuntimeError:
        # workaround for platform differences between 32bit and 64bit
        # scipy.ndimage
        dtype = sub.dtype
        sub = array(sub, dtype='float64')
        sub = interpolation.shift(
            sub, (y - y0, x - x0), mode=mode, cval=cval, order=0)
        sub = array(sub, dtype=dtype)
        return sub


def extract_masked(image, linedesc, pad=5, expand=0):
    """Extract a subimage from the image using the line descriptor.
    A line descriptor consists of bounds and a mask."""
    y0, x0, y1, x1 = [int(x) for x in [linedesc.bounds[0].start, linedesc.bounds[1].start,
                                       linedesc.bounds[0].stop, linedesc.bounds[1].stop]]
    if pad > 0:
        mask = pad_image(linedesc.mask, pad, cval=0)
    else:
        mask = linedesc.mask
    line = extract(image, y0 - pad, x0 - pad, y1 + pad, x1 + pad)
    if expand > 0:
        mask = filters.maximum_filter(mask, (expand, expand))
    line = where(mask, line, amax(line))
    return line


def reading_order(lines, highlight=None, debug=0):
    """Given the list of lines (a list of 2D slices), computes
    the partial reading order.  The output is a binary 2D array
    such that order[i,j] is true if line i comes before line j
    in reading order."""
    order = zeros((len(lines), len(lines)), 'B')

    def x_overlaps(u, v):
        return u[1].start < v[1].stop and u[1].stop > v[1].start

    def above(u, v):
        return u[0].start < v[0].start

    def left_of(u, v):
        return u[1].stop < v[1].start

    def separates(w, u, v):
        if w[0].stop < min(u[0].start, v[0].start):
            return 0
        if w[0].start > max(u[0].stop, v[0].stop):
            return 0
        if w[1].start < u[1].stop and w[1].stop > v[1].start:
            return 1
    if highlight is not None:
        clf()
        title("highlight")
        imshow(binary)
        ginput(1, debug)
    for i, u in enumerate(lines):
        for j, v in enumerate(lines):
            if x_overlaps(u, v):
                if above(u, v):
                    order[i, j] = 1
            else:
                if [w for w in lines if separates(w, u, v)] == []:
                    if left_of(u, v):
                        order[i, j] = 1
            if j == highlight and order[i, j]:
                print((i, j))
                y0, x0 = center(lines[i])
                y1, x1 = center(lines[j])
                plot([x0, x1 + 200], [y0, y1])
    if highlight is not None:
        print()
        ginput(1, debug)
    return order


def topsort(order):
    """Given a binary array defining a partial order (o[i,j]==True means i<j),
    compute a topological sort.  This is a quick and dirty implementation
    that works for up to a few thousand elements."""
    n = len(order)
    visited = zeros(n)
    L = []

    def visit(k):
        if visited[k]:
            return
        visited[k] = 1
        for l in find(order[:, k]):
            visit(l)
        L.append(k)
    for k in range(n):
        visit(k)
    return L  # [::-1]


def show_lines(image, lines, lsort):
    """Overlays the computed lines on top of the image, for debugging
    purposes."""
    ys, xs = [], []
    clf()
    cla()
    imshow(image)
    for i in range(len(lines)):
        l = lines[lsort[i]]
        y, x = center(l.bounds)
        xs.append(x)
        ys.append(y)
        o = l.bounds
        r = matplotlib.patches.Rectangle(
            (o[1].start, o[0].start), edgecolor='r', fill=0, width=dim1(o), height=dim0(o))
        gca().add_patch(r)
    h, w = image.shape
    ylim(h, 0)
    xlim(0, w)
    plot(xs, ys)



def norm_max(v):
    return v / amax(v)


def check_page(image):
    if len(image.shape) == 3:
        return "input image is color image %s" % (image.shape,)
    if mean(image) < median(image):
        return "image may be inverted"
    h, w = image.shape
    if h < 600:
        return "image not tall enough for a page image %s" % (image.shape,)
    if h > 10000:
        return "image too tall for a page image %s" % (image.shape,)
    if w < 600:
        return "image too narrow for a page image %s" % (image.shape,)
    if w > 10000:
        return "line too wide for a page image %s" % (image.shape,)
    slots = int(w * h * 1.0 / (30 * 30))
    _, ncomps = measurements.label(image > mean(image))
    if ncomps < 10:
        return "too few connected components for a page image (got %d)" % (ncomps,)
    if ncomps > slots:
        return "too many connnected components for a page image (%d > %d)" % (ncomps, slots)
    return None



def B(a):
    if a.dtype == dtype('B'):
        return a
    return array(a, 'B')


def DSAVE(title, image, args=None):
    if not args.debug:
        return
    if type(image) == list:
        assert len(image) == 3
        image = transpose(array(image), [1, 2, 0])
    fname = "_" + title + ".png"
    logger.info("debug " + fname)
    imsave(fname, image)


################################################################
# Column finding.
###
# This attempts to find column separators, either as extended
# vertical black lines or extended vertical whitespace.
# It will work fairly well in simple cases, but for unusual
# documents, you need to tune the parameters.
################################################################

def compute_separators_morph(binary, scale, args=None):
    """Finds vertical black lines corresponding to column separators."""
    d0 = int(max(5, scale / 4))
    d1 = int(max(5, scale)) + args.sepwiden
    thick = r_dilation(binary, (d0, d1))
    vert = rb_opening(thick, (10 * scale, 1))
    vert = r_erosion(vert, (d0 // 2, args.sepwiden))
    vert = select_regions(vert, dim1, min=3, nbest=2 * args.maxseps)
    vert = select_regions(vert, dim0, min=20 * scale, nbest=args.maxseps)
    return vert


def compute_colseps_morph(binary, scale, maxseps=3, minheight=20, maxwidth=5, args=None):
    """Finds extended vertical whitespace corresponding to column separators
    using morphological operations."""
    boxmap = compute_boxmap(binary, scale, dtype='B')
    bounds = rb_closing(B(boxmap), (int(5 * scale), int(5 * scale)))
    bounds = maximum(B(1 - bounds), B(boxmap))
    cols = 1 - rb_closing(boxmap, (int(20 * scale), int(scale)))
    cols = select_regions(cols, aspect, min=args.csminaspect)
    cols = select_regions(
        cols, dim0, min=args.csminheight * scale, nbest=args.maxcolseps)
    cols = r_erosion(cols, (int(0.5 + scale), 0))
    cols = r_dilation(cols, (int(0.5 + scale), 0),
                      origin=(int(scale / 2) - 1, 0))
    return cols


def compute_colseps_mconv(binary, scale=1.0, args=None):
    """Find column separators using a combination of morphological
    operations and convolution."""
    h, w = binary.shape
    smoothed = filters.gaussian_filter(1.0 * binary, (scale, scale * 0.5))
    smoothed = filters.uniform_filter(smoothed, (5.0 * scale, 1))
    thresh = (smoothed < amax(smoothed) * 0.1)
    DSAVE("1thresh", thresh, args=args)
    blocks = rb_closing(binary, (int(4 * scale), int(4 * scale)))
    DSAVE("2blocks", blocks, args=args)
    seps = minimum(blocks, thresh)
    seps = select_regions(
        seps, dim0, min=args.csminheight * scale, nbest=args.maxcolseps)
    DSAVE("3seps", seps, args=args)
    blocks = r_dilation(blocks, (5, 5))
    DSAVE("4blocks", blocks, args=args)
    seps = maximum(seps, 1 - blocks)
    DSAVE("5combo", seps, args=args)
    return seps


def compute_colseps_conv(binary, scale=1.0, args=None):
    """Find column separators by convoluation and
    thresholding."""
    h, w = binary.shape
    # find vertical whitespace by thresholding
    smoothed = filters.gaussian_filter(1.0 * binary, (scale, scale * 0.5))
    smoothed = filters.uniform_filter(smoothed, (5.0 * scale, 1))
    thresh = (smoothed < amax(smoothed) * 0.1)
    DSAVE("1thresh", thresh, args=args)
    # find column edges by filtering
    grad = filters.gaussian_filter(
        1.0 * binary, (scale, scale * 0.5), order=(0, 1))
    grad = filters.uniform_filter(grad, (10.0 * scale, 1))
    # grad = abs(grad) # use this for finding both edges
    grad = (grad > 0.5 * amax(grad))
    DSAVE("2grad", grad, args=args)
    # combine edges and whitespace
    seps = minimum(thresh, filters.maximum_filter(
        grad, (int(scale), int(5 * scale))))
    seps = filters.maximum_filter(seps, (int(2 * scale), 1))
    DSAVE("3seps", seps, args=args)
    # select only the biggest column separators
    seps = select_regions(
        seps, dim0, min=args.csminheight * scale, nbest=args.maxcolseps)
    DSAVE("4seps", seps, args=args)
    return seps


def compute_colseps(binary, scale, args):
    """Computes column separators either from vertical black lines or whitespace."""
    logger.info("considering at most %g whitespace column separators" %
               args.maxcolseps)
    colseps = compute_colseps_conv(binary, scale, args=args)
    DSAVE("colwsseps", 0.7 * colseps + 0.3 * binary, args=args)
    if args.blackseps and args.maxseps == 0:
        # simulate old behaviour of blackseps when the default value
        # for maxseps was 2, but only when the maxseps-value is still zero
        # and not set manually to a non-zero value
        args.maxseps = 2
    if args.maxseps > 0:
        logger.info("considering at most %g black column separators" %
                   args.maxseps)
        seps = compute_separators_morph(binary, scale, args=args)
        DSAVE("colseps", 0.7 * seps + 0.3 * binary, args=args)
        #colseps = compute_colseps_morph(binary,scale, args=args)
        colseps = maximum(colseps, seps)
        binary = minimum(binary, 1 - seps)
    return colseps, binary


################################################################
# Text Line Finding.
###
# This identifies the tops and bottoms of text lines by
# computing gradients and performing some adaptive thresholding.
# Those components are then used as seeds for the text lines.
################################################################

def compute_gradmaps(binary, scale, args=None):
    # use gradient filtering to find baselines
    boxmap = compute_boxmap(binary, scale)
    cleaned = boxmap * binary
    DSAVE("cleaned", cleaned, args=args)
    if args.usegauss:
        # this uses Gaussians
        grad = filters.gaussian_filter(1.0 * cleaned, (args.vscale * 0.3 * scale,
                                                       args.hscale * 6 * scale), order=(1, 0))
    else:
        # this uses non-Gaussian oriented filters
        grad = filters.gaussian_filter(1.0 * cleaned, (max(4, args.vscale * 0.3 * scale),
                                                       args.hscale * scale), order=(1, 0))
        grad = filters.uniform_filter(
            grad, (args.vscale, args.hscale * 6 * scale))
    bottom = norm_max((grad < 0) * (-grad))
    top = norm_max((grad > 0) * grad)
    return bottom, top, boxmap


def compute_line_seeds(binary, bottom, top, colseps, scale, args=None):
    """Base on gradient maps, computes candidates for baselines
    and xheights.  Then, it marks the regions between the two
    as a line seed."""
    t = args.threshold
    vrange = int(args.vscale * scale)
    bmarked = filters.maximum_filter(
        bottom == filters.maximum_filter(bottom, (vrange, 0)), (2, 2))
    bmarked = bmarked * (bottom > t * amax(bottom) * t) * (1 - colseps)
    tmarked = filters.maximum_filter(
        top == filters.maximum_filter(top, (vrange, 0)), (2, 2))
    tmarked = tmarked * (top > t * amax(top) * t / 2) * (1 - colseps)
    tmarked = filters.maximum_filter(tmarked, (1, 20))
    seeds = zeros(binary.shape, 'i')
    delta = max(3, int(scale / 2))
    for x in range(bmarked.shape[1]):
        transitions = sorted([(y, 1) for y in find(
            bmarked[:, x])] + [(y, 0) for y in find(tmarked[:, x])])[::-1]
        transitions += [(0, 0)]
        for l in range(len(transitions) - 1):
            y0, s0 = transitions[l]
            if s0 == 0:
                continue
            seeds[y0 - delta:y0, x] = 1
            y1, s1 = transitions[l + 1]
            if s1 == 0 and (y0 - y1) < 5 * scale:
                seeds[y1:y0, x] = 1
    seeds = filters.maximum_filter(seeds, (1, int(1 + scale)))
    seeds = seeds * (1 - colseps)
    DSAVE("lineseeds", [seeds, 0.3 * tmarked +
                        0.7 * bmarked, binary], args=args)
    seeds, _ = label(seeds)
    return seeds


################################################################
# The complete line segmentation process.
################################################################

def remove_hlines(binary, scale, maxsize=10):
    labels, _ = label(binary)
    objects = find_objects(labels)
    for i, b in enumerate(objects):
        if width(b) > maxsize * scale:
            labels[b][labels[b] == i + 1] = 0
    return array(labels != 0, 'B')


def compute_segmentation(binary, scale, args=None):
    """Given a binary image, compute a complete segmentation into
    lines, computing both columns and text lines."""
    binary = array(binary, 'B')

    # start by removing horizontal black lines, which only
    # interfere with the rest of the page segmentation
    binary = remove_hlines(binary, scale)

    # do the column finding
    if not args.quiet:
        logger.info("computing column separators")
    colseps, binary = compute_colseps(binary, scale, args=args)

    # now compute the text line seeds
    if not args.quiet:
        logger.info("computing lines")
    bottom, top, boxmap = compute_gradmaps(binary, scale, args=args)
    seeds = compute_line_seeds(binary, bottom, top, colseps, scale, args=args)
    DSAVE("seeds", [bottom, top, boxmap], args=args)

    # spread the text line seeds to all the remaining
    # components
    if not args.quiet:
        logger.info("propagating labels")
    llabels = propagate_labels(boxmap, seeds, conflict=0)
    if not args.quiet:
        logger.info("spreading labels")
    spread = spread_labels(seeds, maxdist=scale)
    llabels = where(llabels > 0, llabels, spread * binary)
    segmentation = llabels * binary
    return segmentation


################################################################
# Processing each file.
################################################################

def find_lines(binary, gray=None, args=default_args):
    assert binary.ndim == 2
    assert sum((binary != 0) * (binary != amax(binary))) == 0
    binary = 1 * (binary != 0)

    if not args.nocheck:
        error = check_page(amax(binary) - binary)
        assert not error, error

    if gray is not None:
        assert gray.ndim == 2
        assert amin(gray) >= 0
        assert amax(gray) <= 1

    binary = 1 - binary

    # find document scale

    if args.scale == 0:
        scale = estimate_scale(binary)
    else:
        scale = args.scale
    logger.info("scale %f" % (scale))
    if isnan(scale) or scale > 1000.0:
        logger.info("%s: bad scale (%g); skipping\n" % (fname, scale))
        return
    if scale < args.minscale:
        logger.info("%s: scale (%g) less than --minscale; skipping\n" %
                    (fname, scale))
        return

    # find columns and text lines

    if not args.quiet:
        logger.info("computing segmentation")
    segmentation = compute_segmentation(binary, scale, args=args)
    if amax(segmentation) > args.maxlines:
        logger.info("%s: too many lines %g" % (fname, amax(segmentation)))
        return
    if not args.quiet:
        logger.info("number of lines %g" % amax(segmentation))

    # compute the reading order

    if not args.quiet:
        logger.info("finding reading order")
    lines = compute_lines(segmentation, scale)
    order = reading_order([l.bounds for l in lines])
    lsort = topsort(order)

    # renumber the labels so that they conform to the specs

    nlabels = amax(segmentation) + 1
    renumber = zeros(nlabels, 'i')
    for i, v in enumerate(lsort):
        renumber[lines[v].label] = 0x010000 + (i + 1)
    segmentation = renumber[segmentation]

    # finally, output everything

    lines = [lines[i] for i in lsort]

    cleaned = remove_noise(binary, args.noise)
    extracted = []
    for i, l in enumerate(lines):
        binline = extract_masked(
            1 - cleaned, l, pad=args.pad, expand=args.expand)
        if gray is not None:
            grayline = extract_masked(
                gray, l, pad=args.pad, expand=args.expand)
        else:
            grayline = None
        extracted.append(Struct(bbox=l, binary=binline, gray=grayline))

    return Struct(segmentation=segmentation, extracted=extracted)
