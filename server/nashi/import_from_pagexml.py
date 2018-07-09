# -*- coding: utf-8 -*-

"""
Load one particular book and import into database...
"""


from nashi import app
from models import Book, Page
from database import db_session
from glob import glob
from os import path
from lxml import etree

TEXTDIRECTION = 'horizontal-lr'

bookpath = app.config["LAREX_DIR"]+"/Some_Book"

bookname = path.split(bookpath)[1]
no_pages_total = len(glob(bookpath+"/*.png"))

book = Book.query.filter_by(name=bookname).first()
if not book:
    book = Book(name=bookname, no_pages_total=no_pages_total)
else:
    book.no_pages_total = no_pages_total

print(bookname)

for xmlfile in sorted(glob(bookpath+"/*.xml")):
    pagename = path.splitext(path.split(xmlfile)[1])[0]
    print(pagename)

    page = Page.query.filter_by(book_id=book.id, name=pagename).first()
    if not page:
        page = Page(book=book, name=pagename)

    root = etree.parse(xmlfile).getroot()
    ns = {"ns": root.nsmap[None]}

    textregions = root.xpath('//ns:TextRegion', namespaces=ns)

    page.no_lines_segm = int(root.xpath("count(//ns:TextLine)", namespaces=ns))
    page.no_lines_gt = int(root.xpath(
        'count(//ns:TextLine/ns:TextEquiv[@index="0"])', namespaces=ns))
    page.no_lines_ocr = int(root.xpath("count(//ns:TextLine)", namespaces=ns))
    page.data = etree.tounicode(root.getroottree()).replace(
             "http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19",
             "http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15"
            )


db_session.add(book)

db_session.commit()
