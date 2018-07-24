# -*- coding: utf-8 -*-

"""
Load one particular book and import into database...
"""


from nashi.models import Book, Page
from nashi.database import db_session
from sqlalchemy.orm.exc import NoResultFound

from glob import glob
from os import path
from lxml import etree

import argparse


def import_folder(bookpath):
    bookname = path.split(bookpath)[1]
    no_pages_total = len(glob(bookpath+"/*.xml"))
    try:
        book = Book.query.filter_by(name=bookname).one()
    except NoResultFound:
        book = Book(name=bookname, no_pages_total=no_pages_total)
        book.no_pages_total = no_pages_total

    print('Importing book "{}"...'.format(bookname))
    cnt = 0
    for xmlfile in sorted(glob(bookpath+"/*.xml")):
        pagename = path.split(xmlfile)[1].split(".")[0]
        print("Importing page {}...".format(pagename))

        try:
            page = Page.query.filter_by(book_id=book.id, name=pagename).one()
        except NoResultFound:
            page = Page(book=book, name=pagename)

        root = etree.parse(xmlfile).getroot()
        ns = {"ns": root.nsmap[None]}

        textregions = root.xpath('//ns:TextRegion', namespaces=ns)

        page.no_lines_segm = int(root.xpath("count(//ns:TextLine)",
                                 namespaces=ns))
        page.no_lines_gt = int(root.xpath(
            'count(//ns:TextLine/ns:TextEquiv[@index="0"])', namespaces=ns))
        page.no_lines_ocr = int(root.xpath('count(//ns:TextLine[count'
                                           '(./ns:TextEquiv[@index>0])>0])',
                                           namespaces=ns))
        page.data = etree.tounicode(root.getroottree())
        cnt += 1

    db_session.add(book)
    db_session.commit()
    print('{} pages imported for book {}.'.format(cnt, bookname))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bookfolder", type=str,
                        help="Give the directory that contains the PageXML "
                        "files to import.")
    args = parser.parse_args()
    import_folder(args.bookfolder)


if __name__ == "__main__":
    main()
