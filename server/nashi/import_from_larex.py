# -*- coding: utf-8 -*-

"""
Scans the LAREX_DIR for books and PageXML,
if book or page not in db or page.data empty:
create, load, pagesegment, update no_lines_segm
"""

# from nashi import app
from nashi.models import Book, Page
from nashi.database import db_session

import numpy as np
from skimage.measure import find_contours, approximate_polygon

from glob import glob
from os import path
from lxml import etree
from PIL import Image, ImageDraw

from kraken import pageseg, binarization
from kraken.lib import morph, sl
from kraken.lib.util import pil2array
from kraken.binarization import is_bitonal
from kraken.lib.exceptions import KrakenInputException


def cutout(im, coords):
    """
        Cut out coords from image, crop and return new image.
    """
    coords = [tuple(t) for t in coords]
    if not coords:
        return None
    maskim = Image.new('1', im.size, 0)
    ImageDraw.Draw(maskim).polygon(coords, outline=1, fill=1)
    new = Image.new(im.mode, im.size, "white")
    masked = Image.composite(im, new, maskim)
    cropped = masked.crop([
            min([x[0] for x in coords]), min([x[1] for x in coords]),
            max([x[0] for x in coords]), max([x[1] for x in coords])])
    return cropped


class record(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)


def compute_lines(segmentation, spread, scale):
    """Given a line segmentation map, computes a list
    of tuples consisting of 2D slices and masked images."""
    lobjects = morph.find_objects(segmentation)
    lines = []
    for i, o in enumerate(lobjects):
        if o is None:
            continue
        if sl.dim1(o) < 2*scale or sl.dim0(o) < scale:
            continue
        mask = (segmentation[o] == i+1)
        if np.amax(mask) == 0:
            continue
        result = record()
        result.label = i+1
        result.bounds = o
        polygon = []
        if ((segmentation[o] != 0) == (segmentation[o] != i+1)).any():
            ppoints = draw_polygon(spread[o], i+1)
            ppoints = ppoints[1:] if ppoints else []
            polygon = [(o[0].start+p[0], o[1].start+p[1]) for p in ppoints]
        if not polygon:
            polygon = [(o[0].start, o[1].start), (o[0].stop,  o[1].start),
                       (o[0].stop,  o[1].stop),  (o[0].start, o[1].stop)]
        result.polygon = polygon
        result.mask = mask
        lines.append(result)
    return lines


def draw_polygon(lspread, lineno):
    """Draws a polygon around area of value lineno in array lspread."""
    lspread = np.pad(lspread, 1, "constant", constant_values=0)
    cont = find_contours(np.where(lspread == lineno, lineno, 2*lineno), lineno)
    if len(cont) == 1 and all(cont[0][0] == cont[0][-1]):
        polyg = approximate_polygon(cont[0], tolerance=1).astype(int)
        return [(p[0]-1, p[1]-1) for p in polyg]
    else:
        return []


def segment(im, text_direction='horizontal-lr', scale=None, maxcolseps=2,
            black_colseps=False):
    """
    Segments a page into text lines.
    Segments a page into text lines and returns the absolute coordinates of
    each line in reading order.
    Args:
        im (PIL.Image): A bi-level page of mode '1' or 'L'
        text_direction (str): Principal direction of the text
                              (horizontal-lr/rl/vertical-lr/rl)
        scale (float): Scale of the image
        maxcolseps (int): Maximum number of whitespace column separators
        black_colseps (bool): Whether column separators are assumed to be
                              vertical black lines or not
    Returns:
        {'text_direction': '$dir', 'boxes': [(x1, y1, x2, y2),...]}: A
        dictionary containing the text direction and a list of reading order
        sorted bounding boxes under the key 'boxes'.
    Raises:
        KrakenInputException if the input image is not binarized or the text
        direction is invalid.
    """

    if im.mode != '1' and not is_bitonal(im):
        raise KrakenInputException('Image is not bi-level')

    # rotate input image for vertical lines
    if text_direction.startswith('horizontal'):
        angle = 0
        offset = (0, 0)
    elif text_direction == 'vertical-lr':
        angle = 270
        offset = (0, im.size[1])
    elif text_direction == 'vertical-rl':
        angle = 90
        offset = (im.size[0], 0)
    else:
        raise KrakenInputException('Invalid text direction')

    im = im.rotate(angle, expand=True)

    # honestly I've got no idea what's going on here. In theory a simple
    # np.array(im, 'i') should suffice here but for some reason the
    # tostring/fromstring magic in pil2array alters the array in a way that is
    # needed for the algorithm to work correctly.
    a = pil2array(im)
    binary = np.array(a > 0.5*(np.amin(a) + np.amax(a)), 'i')
    binary = 1 - binary

    if not scale:
        scale = pageseg.estimate_scale(binary)

    binary = pageseg.remove_hlines(binary, scale)
    # emptyish images will cause exceptions here.
    try:
        if black_colseps:
            colseps, binary = pageseg.compute_black_colseps(binary, scale,
                                                            maxcolseps)
        else:
            colseps = pageseg.compute_white_colseps(binary, scale, maxcolseps)
    except ValueError:
        return {'text_direction': text_direction, 'boxes':  []}

    bottom, top, boxmap = pageseg.compute_gradmaps(binary, scale)
    seeds = pageseg.compute_line_seeds(binary, bottom, top, colseps, scale)
    llabels1 = morph.propagate_labels(boxmap, seeds, conflict=0)
    spread = morph.spread_labels(seeds, maxdist=scale)
    llabels = np.where(llabels1 > 0, llabels1, spread*binary)
    segmentation = llabels*binary

    lines_and_polygons = compute_lines(segmentation, spread, scale)
    # TODO: rotate_lines for polygons
    order = pageseg.reading_order([l.bounds for l in lines_and_polygons])
    lsort = pageseg.topsort(order)
    lines = [lines_and_polygons[i].bounds for i in lsort]
    lines = [(s2.start, s1.start, s2.stop, s1.stop) for s1, s2 in lines]
    return {'text_direction': text_direction,
            'boxes': pageseg.rotate_lines(lines, 360-angle, offset).tolist(),
            'lines': lines_and_polygons,
            'script_detection': False}


def pagexmllineseg(xmlfile, imgpath, text_direction='horizontal-lr'):
    root = etree.parse(xmlfile).getroot()
    ns = {"ns": root.nsmap[None]}

    # convert point notation from older pagexml versions
    for c in root.xpath("//ns:Coords[not(@points)]", namespaces=ns):
        cc = []
        for point in c.xpath("./ns:Point", namespaces=ns):
            # coordstrings = [x.split(",") for x in c.attrib["points"].split()]
            cx = point.attrib["x"]
            cy = point.attrib["y"]
            c.remove(point)
            cc.append(cx+","+cy)
        c.attrib["points"] = " ".join(cc)

    coordmap = {}
    for r in root.xpath('//ns:TextRegion', namespaces=ns):
        rid = r.attrib["id"]
        coordmap[rid] = {"type": r.attrib["type"]}
        coordmap[rid]["coords"] = []
        for c in r.xpath("./ns:Coords", namespaces=ns) + r.xpath("./Coords"):
            coordmap[rid]["coordstring"] = c.attrib["points"]
            coordstrings = [x.split(",") for x in c.attrib["points"].split()]
            coordmap[rid]["coords"] += [[int(x[0]), int(x[1])]
                                        for x in coordstrings]

    filename = root.xpath('//ns:Page', namespaces=ns)[0]\
        .attrib["imageFilename"]
    filename = imgpath + "/" + filename

    im = Image.open(filename)

    for n, c in enumerate(sorted(coordmap)):
        coords = coordmap[c]['coords']
        if len(coords) < 3:
            continue
        cropped = cutout(im, coords)
        offset = (min([x[0] for x in coords]), min([x[1] for x in coords]))
        if cropped is not None:
            if not binarization.is_bitonal(cropped):
                try:
                    cropped = binarization.nlbin(cropped)
                except SystemError:
                    continue
            if coordmap[c]["type"] == "drop-capital":
                lines = [1]
            else:
                # if line in
                lines = segment(cropped, text_direction=text_direction,
                                maxcolseps=-1)

                lines = lines["lines"] if "lines" in lines else []
        else:
            lines = []

        for n, l in enumerate(lines):
            if coordmap[c]["type"] == "drop-capital":
                coordstrg = coordmap[c]["coordstring"]
            else:
                coords = ((x[1]+offset[0], x[0]+offset[1]) for x in l.polygon)
                coordstrg = " ".join([str(x[0])+","+str(x[1]) for x in coords])
            textregion = root.xpath('//ns:TextRegion[@id="'+c+'"]',
                                    namespaces=ns)[0]
            linexml = etree.SubElement(textregion, "TextLine",
                                       attrib={"id": "{}_l{:03d}".format(
                                        c, n+1)})
            coordsxml = etree.SubElement(linexml, "Coords",
                                         attrib={"points": coordstrg})
    xmlstring = etree.tounicode(root.getroottree()).replace(
        "http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19",
        "http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15")
    no_lines_segm = int(root.xpath("count(//TextLine)"))
    return xmlstring, no_lines_segm


def create_book(bookpath):
    """ Return book database object for path in larex dir. """
    bookname = path.split(bookpath)[1]
    no_pages_total = len(glob(bookpath+"/*.png"))
    book = Book.query.filter_by(name=bookname).one()
    if not book:
        book = Book(name=bookname, no_pages_total=no_pages_total)
    else:
        book.no_pages_total = no_pages_total
    return book


def add_page(book, xmlfile, commit=True):
    """ Add page from PageXML file to book (segment lines, update version...).
        TODO: add reading order
    """
    text_direction = 'horizontal-rl'\
        if book.name.endswith("_ar") else 'horizontal-lr'
    bookpath = path.split(xmlfile)[0]
    pagename = path.split(xmlfile)[1].split(".")[0]
    page = Page.query.filter_by(book_id=book.id, name=pagename).one()
    if not page:
        page = Page(book=book, name=pagename)

    page.data, page.no_lines_segm = \
        pagexmllineseg(xmlfile, bookpath, text_direction=text_direction)
    if commit:
        db_session.add(book)
        db_session.commit()
    return page.no_lines_segm


if __name__ == '__main__':
    for bookpath in [x for x in glob(app.config["LAREX_DIR"]+"/*")
                     if path.isdir(x)]:

        book = create_book(bookpath)

        for xmlfile in sorted(glob(bookpath+"/*.xml")):
            add_page(book, xmlfile, commit=False)

        db_session.add(book)
        db_session.commit()
