# -*- coding: utf-8 -*-

import numpy as np
import skimage.io as skimage_io
from skimage.draw import polygon
from io import BytesIO
from PIL import Image


def cutout(pageimg, coordstring, scale=1):
    """Cuts out coords from pageimg."""
    if len(pageimg.shape) > 2:
        pageimg = pageimg[:, :, 0]
    coords = [p.split(",") for p in coordstring.split()]
    coords = np.array([(int(scale*int(c[1])), int(scale*int(c[0])))
                      for c in coords])
    rr, cc = polygon(coords[:, 0], coords[:, 1], pageimg.shape)
    offset = (min([x[0] for x in coords]), min([x[1] for x in coords]))
    box = np.ones(
        (max([x[0] for x in coords]) - offset[0],
         max([x[1] for x in coords]) - offset[1]),
        dtype=pageimg.dtype) * 255
    box[rr-offset[0], cc-offset[1]] = pageimg[rr, cc]
    return box, offset


def getsnippet(filename, coordstring, imgshape=None):
    pageimg = skimage_io.imread(filename)
    scale = 1
    if imgshape:
        scale = pageimg.shape[1] / imgshape[0]
    cut = cutout(pageimg, coordstring, scale)[0]
    file = BytesIO()
    image = Image.fromarray(cut)
    image.save(file, 'png')
    file.name = "line.png"
    file.seek(0)
    return file.read()
