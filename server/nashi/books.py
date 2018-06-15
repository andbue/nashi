# -*- coding: utf-8 -*-

from nashi.models import Book
from nashi.database import db_session

from glob import glob
from os import path, mkdir, symlink, chmod
from shutil import chown, copy


def scan_bookfolder(bookfolder):
    """ Scan bookfolder and write book info to database. """
    books = glob(bookfolder + "//*/")
    for bookpath in books:
        bookname = path.split(bookpath[:-1])[1]
        files = set([f.split(sep=".")[0] for f in glob(bookpath+"*.png")])
        no_pages_total = len(files)
        book = Book.query.filter_by(name=bookname).first()
        if not book:
            book = Book(name=bookname, no_pages_total=no_pages_total)
            db_session.add(book)
        else:
            book.no_pages_total = no_pages_total
    db_session.commit()


def copy_to_larex(bookname, booksdir, larexdir, larexgrp):
    pngfiles = glob("{}/{}/*.png".format(booksdir, bookname))
    pages = set([path.split(f)[1].split(".")[0] for f in pngfiles])
    cpimgs = []
    for p in sorted(pages):
        fullbinp = "{}/{}/{}.bin.png".format(booksdir, bookname, p)
        fullnrmp = "{}/{}/{}.nrm.png".format(booksdir, bookname, p)
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
