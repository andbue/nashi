# -*- coding: utf-8 -*-

import numpy as np
import skimage.io as skimage_io
from skimage.draw import polygon
from io import BytesIO
from PIL import Image


def coordstringtoarray(coordstring, scale=1):
    coords = [p.split(",") for p in coordstring.split()]
    return np.array([(int(scale*int(c[1])), int(scale*int(c[0])))
                     for c in coords])


def expandcoords(coords, imgshape, context, sides=1, rcoords=None):
    if type(coords) != np.ndarray:
        coords = coordstringtoarray(coords)
    if rcoords and type(rcoords) != np.ndarray:
        rcoords = coordstringtoarray(rcoords)
    print(coords)
    print(type(coords))
    lineh = max([p[0] for p in coords]) - min([p[0] for p in coords])
    xmin = max(0, int(min(p[0] for p in coords) - (context * lineh)))
    xmax = min(imgshape[1], int(max(p[0] for p in coords) + (context * lineh)))
    ymin = min(p[1] for p in rcoords)
    ymax = max(p[1] for p in rcoords)
    return np.array([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])


def cutout(pageimg, coords, scale=1):
    """Cuts out coords from pageimg."""
    if len(pageimg.shape) > 2:
        pageimg = pageimg[:, :, 0]
    if type(coords) != np.ndarray:
        coords = coordstringtoarray(coords, scale)
    rr, cc = polygon(coords[:, 0], coords[:, 1], pageimg.shape)
    offset = (min([x[0] for x in coords]), min([x[1] for x in coords]))
    box = np.ones(
        (max([x[0] for x in coords]) - offset[0],
         max([x[1] for x in coords]) - offset[1]),
        dtype=pageimg.dtype) * 255
    box[rr-offset[0], cc-offset[1]] = pageimg[rr, cc]
    return box, offset


def getsnippet(filename, coords, imgshape=None, context=0, rcoords=None):
    pageimg = skimage_io.imread(filename)
    scale = 1
    if imgshape:
        scale = pageimg.shape[1] / imgshape[0]
    if context:
        coords = expandcoords(coords, imgshape, context, rcoords=rcoords)
    cut = cutout(pageimg, coords, scale)[0]
    file = BytesIO()
    image = Image.fromarray(cut)
    image.save(file, 'png')
    file.name = "line.png"
    file.seek(0)
    return file.read()
