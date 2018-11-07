# -*- coding: utf-8 -*-

import zipfile

from nashi.models import Book, Page
from nashi.database import db_session

from glob import glob
from os import path, mkdir, symlink, chmod
from shutil import chown, copy
from lxml import etree
from sqlalchemy.orm.exc import NoResultFound


def scan_bookfolder(bookfolder, img_subdir):
    """ Scan bookfolder and write book info to database. """
    books = glob(bookfolder + "//*/")
    for bookpath in books:
        bookname = path.split(bookpath[:-1])[1]
        files = set([f.split(sep=".")[0] for f in glob(bookpath + img_subdir
                                                       + "*.png")])
        no_pages_total = len(files)
        book = Book.query.filter_by(name=bookname).first()
        if not book:
            book = Book(name=bookname, no_pages_total=no_pages_total)
            db_session.add(book)
        else:
            book.no_pages_total = no_pages_total
    db_session.commit()


def upload_pagexml(file):
    try:
        zf = zipfile.ZipFile(file._file)
    except zipfile.BadZipFile:
        return "Upload failed. Please upload a valid zip file."
    result = {}
    for fn in zf.namelist():
        try:
            bookname, filename = fn.split("/")
        except ValueError:
            return "Upload failed. The files inside the zip file have to be" +\
                   " named <BOOKNAME>/<PAGENAME>.xml."
        if not filename:
            continue
        try:
            book = Book.query.filter_by(name=bookname).one()
        except NoResultFound:
            return "Import aborted. Book {} is not in your library."\
                    .format(bookname)
        if bookname not in result:
            result[bookname] = 0
        pagename = path.splitext(path.split(filename)[1])[0]
        try:
            page = Page.query.filter_by(book_id=book.id, name=pagename).one()
        except NoResultFound:
            page = Page(book=book, name=pagename)
        # return "Import aborted. Book {}, page {} is not in your library."\
        #       .format(bookname, pagename)
        pagexml = zf.read(fn).decode("utf-8")
        root = etree.fromstring(pagexml)
        ns = {"ns": root.nsmap[None]}
        page.no_lines_segm = int(root.xpath("count(//ns:TextLine)",
                                            namespaces=ns))
        page.no_lines_gt = int(root.xpath(
            'count(//ns:TextLine/ns:TextEquiv[@index="0"])', namespaces=ns))
        page.no_lines_ocr = int(root.xpath('count(//ns:TextLine'
                                           '[count(./ns:TextEquiv'
                                           '[@index>0])>0])', namespaces=ns))
        page.data = etree.tounicode(root.getroottree())
        result[bookname] += 1
    db_session.commit()
    res = "Import successfull: {}.".format(", ".join(
        [": ".join([i[0], str(i[1])]) for i in result.items()]))
    return res


def getlayers(book):
    layers = set([])
    for p in Page.query.filter_by(book_id=book.id):
        root = etree.fromstring(p.data)
        ns = {"ns": root.nsmap[None]}
        nl = set(root.xpath('//ns:TextEquiv/@index', namespaces=ns))
        layers = layers | nl
    return sorted(list(layers), key=int)


def copy_to_larex(bookname, booksdir, img_subdir, larexdir, larexgrp):
    pngfiles = glob("{}/{}/{}/*.png".format(booksdir, bookname, img_subdir))
    pngfiles = [path.abspath(f) for f in pngfiles]
    pages = set([path.split(f)[1].split(".")[0] for f in pngfiles])
    cpimgs = []
    for p in sorted(pages):
        fullbinp = path.abspath("{}/{}/{}/{}.bin.png"
                                .format(booksdir, bookname, img_subdir, p))
        fullnrmp = path.abspath("{}/{}/{}/{}.nrm.png"
                                .format(booksdir, bookname, img_subdir, p))
        if fullbinp in pngfiles:
            cpimgs.append(fullbinp)
        elif fullnrmp in pngfiles:
            cpimgs.append(fullnrmp)
        else:
            cpimgs.append([x for x in pngfiles if path.split(x)[1].startswith(
                p+".")][0])
    if cpimgs:
        if not path.isdir(larexdir + "/" + bookname):
            mkdir(larexdir + "/" + bookname)
            chmod(larexdir + "/" + bookname, 0o770)
            try:
                chown(larexdir + "/" + bookname, group=larexgrp)
            except PermissionError:
                pass  # Needs error handling, maybe flash?
        for f in cpimgs:
            fname = path.split(f)[1]
            if not path.isfile("{}/{}/{}".format(larexdir, bookname, fname)):
                symlink(f, "{}/{}/{}".format(larexdir, bookname, fname))
            # copy(f, "{}/{}/{}".format(larexdir, bookname, fname))
    return len(cpimgs)
