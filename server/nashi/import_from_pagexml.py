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


def import_folder(bookpath, bookname="", pages="*.xml"):
    if not bookname:
        bookname = path.split(bookpath)[1]
    no_pages_total = len(glob(bookpath+"/*.xml"))
    try:
        book = Book.query.filter_by(name=bookname).one()
    except NoResultFound:
        book = Book(name=bookname, no_pages_total=no_pages_total)
        book.no_pages_total = no_pages_total

    print('Importing book "{}"...'.format(bookname))
    cnt = 0
    for xmlfile in sorted(glob(bookpath+"/"+pages)):
        pagename = path.split(xmlfile)[1].split(".")[0]
        print("Importing page {}...".format(pagename))

        try:
            page = Page.query.filter_by(book_id=book.id, name=pagename).one()
        except NoResultFound:
            page = Page(book=book, name=pagename)

        root = etree.parse(xmlfile).getroot()
        ns = {"ns": root.nsmap[None]}

        # convert point notation from pagexml version 2013
        for c in root.xpath("//ns:Coords[not(@points)]", namespaces=ns):
            cc = []
            for point in c.xpath("./ns:Point", namespaces=ns):
                cc.append(point.attrib["x"]+","+point.attrib["y"])
                c.remove(point)
            c.attrib["points"] = " ".join(cc)

        textregions = root.xpath('//ns:TextRegion', namespaces=ns)

        page.no_lines_segm = int(root.xpath("count(//ns:TextLine)",
                                 namespaces=ns))
        page.no_lines_gt = int(root.xpath(
            'count(//ns:TextLine/ns:TextEquiv[@index="0"])', namespaces=ns))
        page.no_lines_ocr = int(root.xpath('count(//ns:TextLine[count'
                                           '(./ns:TextEquiv[@index>0])>0])',
                                           namespaces=ns))
        page.data = etree.tounicode(root.getroottree()).replace(
            "http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19",
            "http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15")
        cnt += 1

    db_session.add(book)
    db_session.commit()
    print('{} pages imported for book {}.'.format(cnt, bookname))


def bookdelete():
    parser = argparse.ArgumentParser()
    parser.add_argument("bookname", type=str, help="The name of the book.")
    args = parser.parse_args()
    try:
        book = Book.query.filter_by(name=args.bookname).one()
    except NoResultFound:
        print("Book {} not in database!".format(args.bookname))
        return
    for p in book.pages:
        db_session.delete(p)
    db_session.delete(book)
    db_session.commit()
    print("Deleted book {} from database.".format(args.bookname))


def bookexport():
    parser = argparse.ArgumentParser()
    parser.add_argument("bookfolder", type=str,
                        help="Give the directory to write the PageXML files "
                        "to.")
    parser.add_argument("bookname", type=str, help="The name of "
                        "the book to export.")
    args = parser.parse_args()
    try:
        book = Book.query.filter_by(name=args.bookname).one()
    except NoResultFound:
        print("Book {} not in database!".format(args.bookname))
        return
    for page in book.pages:
        with open("{}/{}.xml".format(args.bookfolder, page.name), "w") as f:
            f.write(page.data)
    print("{} pages exported.".format(len(book.pages)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bookfolder", type=str,
                        help="Give the directory that contains the PageXML "
                        "files to import.")
    parser.add_argument("bookname", type=str, default="", help="The name of "
                        "the book. Defaults to the last folder in the path of "
                        "the given bookfolder.")
    parser.add_argument("pages", type=str, default="*.xml", help="A wildcard "
                        "for the pages to import. By default it imports all "
                        "xml files in the given directory.")
    args = parser.parse_args()
    import_folder(args.bookfolder, args.bookname, args.pages)


if __name__ == "__main__":
    main()
