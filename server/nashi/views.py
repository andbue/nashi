from flask import send_from_directory, render_template, make_response,\
    request, json, jsonify, flash, redirect, url_for, Response
from flask_security import login_required
from flask_security.core import current_user
import requests
import gzip

import zipfile
import unicodedata
from lxml import etree, html
from glob import glob
from os import path
from io import BytesIO
from time import gmtime
from copy import deepcopy

from sqlalchemy.orm.exc import NoResultFound

from nashi import app
from nashi.models import Book, Page, EditorSettings
from nashi.database import db_session
from nashi.books import scan_bookfolder, copy_to_larex, upload_pagexml,\
    getlayers
from nashi.tasks import lareximport
from nashi.image import getsnippet


@app.route('/')
@login_required
def index():
    books = {}
    for book in Book.query.all():
        ocrstatus = book.ocrpid if book.ocrpid else "N/A"
        books[book.name] = {
                "no_pages_total": book.no_pages_total,
                "no_pages_segm": len(book.pages),
                "no_lines_segm": sum([p.no_lines_segm for p in book.pages]),
                "no_lines_gt": sum([p.no_lines_gt for p in book.pages]),
                "no_lines_ocr": sum([p.no_lines_ocr for p in book.pages]),
                "ocrstatus": ocrstatus
                }

    return render_template('index.html', books=books)


@app.route('/_library/<action>', methods=['POST', 'GET'])
@login_required
def library(action=""):
    print("action: " + action)
    if action == "refresh_booklist":
        scan_bookfolder(app.config["BOOKS_DIR"], app.config["IMAGE_SUBDIR"])
        res = jsonify(success=1)
    if action == "upload_pagexml":
        res = upload_pagexml(request.files["importzip"])
    return res


@app.route('/_libedit/<bookname>/<action>', methods=['POST', 'GET'])
@login_required
def libedit(bookname, action=""):
    book = Book.query.filter_by(name=bookname).first()
    if not book:
        return jsonify(success=0)

    if action == "delete":
        db_session.delete(book)
        db_session.commit()
        return jsonify(success=1)

    if action == "copy_to_larex":
        count = copy_to_larex(bookname, app.config["BOOKS_DIR"],
                              app.config["IMAGE_SUBDIR"],
                              app.config["LAREX_DIR"], app.config["LAREX_GRP"])
        flash("Copied {} files to LAREX.".format(count))
        return jsonify(files_copied=count)

    if action == "select_from_larex":
        ps = glob("{}/{}/*.xml".format(app.config["LAREX_DIR"], bookname))
        pages = []
        for p in ps:
            pname = path.split(p)[1].split(".")[0]
            existing = Page.query.filter_by(
                book_id=book.id, name=pname).first()
            lines_ex = existing.no_lines_gt if existing else -1
            pages.append((pname, lines_ex))
        return jsonify(res=render_template('selectlarex.html', pages=pages))

    if action == "import_from_larex":
        data = request.json
        pages = data["pages"]
        pages = ["{}/{}/{}.xml".format(app.config["LAREX_DIR"], bookname, p)
                 for p in pages]
        for p in pages:
            if not path.isfile(p):
                pages[pages.index(p)] = p[:-3] + "bin.xml"
        task = lareximport.apply_async(
            args=[bookname], kwargs={"pages": pages})
        return jsonify({'Location': url_for('taskstatus', task_id=task.id)})


@app.route('/books/<bookname>/edit.html')
@login_required
def editor(bookname):
    book = Book.query.filter_by(name=bookname).first()
    if not book:
        return redirect(url_for("index"))
    pages = [(x.name, x.no_lines_gt, x.no_lines_segm) for x in book.pages]
    return render_template('editor.html', bookname=book.name, pages=pages)


@app.route('/_editorsettings', methods=['GET', 'POST'])
@login_required
def editorsettings():
    if current_user.is_anonymous:
        email = "user@nashi"
    else:
        email = current_user.email
    if request.method == "GET":
        try:
            s = EditorSettings.query.filter_by(email=email).one()
        except NoResultFound:
            return jsonify(status="fail")
        return jsonify(status="success", settings=json.loads(s.settings))

    if request.method == "POST":
        try:
            s = EditorSettings.query.filter_by(email=email).one()
        except NoResultFound:
            s = EditorSettings(email=email)
        s.settings = json.dumps(request.get_json())
        db_session.add(s)
        db_session.commit()
        return jsonify(status="success")


@app.route('/_ocrdata', methods=['POST'])
@login_required
def ocrdata():
    if "Content-Encoding" in request.headers and \
            request.headers["Content-Encoding"] == "gzip":
        data = json.loads(gzip.decompress(request.data).decode("utf-8"))
    else:
        data = request.get_json()
    cnt = 0
    for bname, bdict in data["ocrdata"].items():
        b = Book.query.filter_by(name=bname).one()
        for pname, pdict in bdict.items():
            p = Page.query.filter_by(book_id=b.id, name=pname).one()
            root = etree.fromstring(p.data)
            ns = {"ns": root.nsmap[None]}
            for lid, text in pdict.items():
                linexml = root.find('.//ns:TextLine[@id="'+lid+'"]',
                                    namespaces=ns)
                if linexml is None:
                    continue
                textequivxml = linexml.find('./ns:TextEquiv[@index="{}"]'
                                            .format(data["index"]),
                                            namespaces=ns)
                if textequivxml is None:
                    textequivxml = etree.SubElement(linexml,
                                                    "{{{}}}TextEquiv"
                                                    .format(ns["ns"]),
                                                    attrib={"index":
                                                            str(data["index"])
                                                            })
                unicodexml = textequivxml.find('./ns:Unicode',
                                               namespaces=ns)
                if unicodexml is None:
                    unicodexml = etree.SubElement(textequivxml,
                                                  "{{{}}}Unicode"
                                                  .format(ns["ns"]))
                unicodexml.text = text
                cnt += 1
            p.no_lines_ocr = int(root.xpath('count(//ns:TextLine'
                                            '[count(./ns:TextEquiv'
                                            '[@index>0])>0])',
                                            namespaces=ns))
            p.data = etree.tounicode(root.getroottree())
    db_session.commit()
    return "Imported {} lines.".format(cnt)


@app.route('/books/<bookname>/textedit.html')
@login_required
def textedit(bookname):
    return render_template("textedit.html", bookname=bookname)


@app.route('/books/<bookname>/textlayers.html', methods=['GET', 'POST'])
@login_required
def textlayers(bookname):
    data = request.get_json()
    b = Book.query.filter_by(name=bookname).one()

    if request.method == "GET":
        return jsonify(layers=getlayers(b))

    elif data["action"] == "copy":
        source = data["layer"]
        target = data["target"]
        ct = 0
        for p in Page.query.filter_by(book_id=b.id):
            root = etree.fromstring(p.data)
            ns = {"ns": root.nsmap[None]}
            for e in root.xpath('//ns:TextEquiv[@index="{}"]'.format(source),
                                namespaces=ns):
                tl = e.getparent()
                new = deepcopy(e)
                new.attrib["index"] = target
                old = e.xpath('../ns:TextEquiv[@index="{}"]'.format(target),
                              namespaces=ns)
                if old:
                    e.getparent().remove(old[0])
                e.getparent().append(new)
                ct += 1
            p.no_lines_gt = int(root.xpath('count(//ns:TextEquiv'
                                           '[@index="0"])', namespaces=ns))
            p.no_lines_ocr = int(root.xpath('count(//ns:TextLine'
                                            '[count(./ns:TextEquiv'
                                            '[@index>0])>0])',
                                            namespaces=ns))
            p.data = etree.tounicode(root.getroottree())
        db_session.commit()
        return jsonify(copied=ct)

    elif data["action"] == "delete":
        layer = int(data["layer"])
        ct = 0
        for p in Page.query.filter_by(book_id=b.id):
            root = etree.fromstring(p.data)
            ns = {"ns": root.nsmap[None]}
            for e in root.xpath('//ns:TextEquiv[@index="{}"]'.format(layer),
                                namespaces=ns):
                e.getparent().remove(e)
                ct += 1
            p.no_lines_gt = int(root.xpath('count(//ns:TextEquiv'
                                           '[@index="0"])', namespaces=ns))
            p.no_lines_ocr = int(root.xpath('count(//ns:TextLine'
                                            '[count(./ns:TextEquiv'
                                            '[@index>0])>0])',
                                            namespaces=ns))
            p.data = etree.tounicode(root.getroottree())
        db_session.commit()
        return jsonify(deleted=ct)


@app.route('/books/<bookname>/chartable.html', methods=['GET'])
@login_required
def chartable(bookname):
    layer = request.args.get("layer", "0")
    print("Layer: {}".format(layer))
    b = Book.query.filter_by(name=bookname).one()
    chars = ""
    for p in Page.query.filter_by(book_id=b.id):
        root = etree.fromstring(p.data)
        ns = {"ns": root.nsmap[None]}
        chars += "".join(root.xpath('//ns:TextLine/' +
                                    'ns:TextEquiv[@index="{}"]'.format(layer) +
                                    '/ns:Unicode/text()', namespaces=ns))
    chars = set(chars)
    table = []
    for c in chars:
        try:
            name = unicodedata.name(c)
        except ValueError:
            name = "unknown"
        table.append((c, 'U+{:04X}'.format(ord(c)), name))
    return render_template("chartable.html", bookname=bookname, results=table)


@app.route('/books/<bookname>/_textsearch.html', methods=['POST'])
@login_required
def textsearch(bookname):
    data = request.get_json()
    layer = data["layer"]

    if data.get("commented", False):
        book = Book.query.filter_by(name=bookname).one()
        results = []
        for p in book.pages:
            root = etree.fromstring(p.data)
            ns = {"ns": root.nsmap[None]}
            found = [t for t in root.xpath('//ns:TextLine[@comments]/' +
                                           'ns:TextEquiv[@index="{}"]'
                                           .format(layer), namespaces=ns)
                     if t.getparent().attrib["comments"]]
            for o in found:
                text = o.xpath('./ns:Unicode/text()', namespaces=ns)
                text = str(text[0]) if text else ""
                textregion = o.getparent()
                id = textregion.attrib["id"]
                comment = textregion.attrib["comments"]
                results.append((p.name, id, text, comment))
            if len(results) > 10000:
                return render_template("_searchresults.html", results=results,
                                       cnt="{} (there could be more,"
                                       .format(len(results))+"max. exceeded)")
        return render_template("_searchresults.html", results=results,
                               cnt=len(results))

    else:
        searchterm = data["searchterm"]
        book = Book.query.filter_by(name=bookname).one()
        results = []

        for p in book.pages:
            root = etree.fromstring(p.data)
            ns = {"ns": root.nsmap[None]}
            found = [t for t in root.xpath('//ns:TextEquiv[@index="{}"]'
                                           .format(layer) +
                                           '/ns:Unicode/text()',
                                           namespaces=ns) if searchterm in t]
            for o in found:
                text = str(o)
                textregion = o.getparent().getparent().getparent()
                id = textregion.attrib["id"]
                if "comments" in textregion.attrib:
                    comment = textregion.attrib["comments"]
                else:
                    comment = ""
                results.append((p.name, id, text, comment))
            if len(results) > 10000:
                return render_template("_searchresults.html", results=results,
                                       cnt="{} (there could be more,"
                                       .format(len(results))+"max. exceeded)")
        return render_template("_searchresults.html", results=results,
                               cnt=len(results))


@app.route('/books/<bookname>/_textreplace.html', methods=['POST'])
@login_required
def textreplace(bookname):
    data = request.get_json()
    replacements = {}
    for r in data["replacements"]:
        if r["page"] not in replacements:
            replacements[r["page"]] = []
        replacements[r["page"]].append((r["line"], r["text"].strip(),
                                        r["comment"].strip()))
    layer = data["layer"]
    book = Book.query.filter_by(name=bookname).one()
    cnt = 0
    for pname, rs in replacements.items():
        p = Page.query.filter_by(book_id=book.id, name=pname).one()
        xml = p.data
        root = etree.fromstring(xml)
        ns = {"ns": root.nsmap[None]}
        for r in rs:
            uc = root.xpath('//ns:TextLine[@id="{}"]'.format(r[0]) +
                            '/ns:TextEquiv[@index="{}"]'.format(layer) +
                            '/ns:Unicode', namespaces=ns)[0]
            uc.text = r[1]
            uc.getparent().getparent().attrib["comments"] = r[2]
            cnt += 1
        p.data = etree.tounicode(root.getroottree())
    db_session.commit()
    return "Wrote {} lines to layer {}.".format(cnt, layer)


@app.route('/books/<bookname>/<pageno>/<lineid>.png', methods=['GET'])
@login_required
def getlineimage(bookname, pageno, lineid):
    context = float(request.args.get("context", "0"))
    book = Book.query.filter_by(name=bookname).one()
    xml = Page.query.filter_by(book_id=book.id, name=pageno).one().data
    root = etree.fromstring(xml)
    ns = {"ns": root.nsmap[None]}
    coords = root.xpath('//ns:TextLine[@id="{}"]/ns:Coords/@points'
                        .format(lineid), namespaces=ns)[0]

    coords_region = root.xpath('//ns:TextLine[@id="{}"]/../ns:Coords/@points'
                               .format(lineid), namespaces=ns)[0]
    pxml = root.find(".//ns:Page", namespaces=ns)
    fn = pxml.attrib["imageFilename"]
    imgshape = (int(pxml.attrib["imageWidth"]),
                int(pxml.attrib["imageHeight"]))
    if fn.endswith(".bin.png"):
        altfile = "{}/{}/{}/{}.raw.png".format(app.config['BOOKS_DIR'],
                                               bookname,
                                               app.config['IMAGE_SUBDIR'],
                                               fn[:-7])
        if path.isfile(altfile):
            fn = fn[:-7] + "raw.png"
    im = getsnippet("{}/{}/{}/{}".format(app.config['BOOKS_DIR'], bookname,
                                         app.config['IMAGE_SUBDIR'], fn),
                    coords, imgshape, context=context, rcoords=coords_region)
    return Response(im, mimetype="image/png")


@app.route('/books/<bookname>_PageXML.zip')
@login_required
def getzip(bookname):
    book = Book.query.filter_by(name=bookname).one()
    zf = BytesIO()
    z = zipfile.ZipFile(zf, mode='w', compression=zipfile.ZIP_DEFLATED)
    for p in book.pages:
        z.writestr(bookname + "/" + p.name + ".xml", p.data)
        # with z.open(p.name + ".xml", "w") as cont:
        #    cont.write(p.data.encode("utf-8"))
    for contfile in z.filelist:
        contfile.date_time = gmtime()[:6]
    z.close()
    zf.seek(0)
    zipped = zf.read()
    zf.close()
    return Response(zipped, mimetype='application/zip')


@app.route('/books/<bookname>/<file>.xml')
@login_required
def getxml(bookname, file):
    book = Book.query.filter_by(name=bookname).first()
    xml = Page.query.filter_by(book_id=book.id, name=file).first().data
    return Response(xml, mimetype='text/xml')


@app.route('/books/<bookname>/<file>.png')
@login_required
def getpng(bookname, file):
    upgrade = request.args.get("upgrade", "")
    if upgrade\
        and path.isfile(
            "{}/{}/{}/{}.{}.png".format(app.config['BOOKS_DIR'],
                                        bookname, app.config['IMAGE_SUBDIR'],
                                        file.split(".")[0], upgrade)):
        file = "{}.{}".format(file.split(".")[0], upgrade)
        print("upgraded")
    if not upgrade and file.endswith(".bin"):
        altfile = "{}/{}/{}/{}.raw.png".format(app.config['BOOKS_DIR'],
                                               bookname,
                                               app.config['IMAGE_SUBDIR'],
                                               file[:-4])
        if path.isfile(altfile):
            file = file[:-3] + "raw"
    # return app.send_static_file("0003.png")
    return send_from_directory(app.config['BOOKS_DIR'] + bookname
                               + app.config['IMAGE_SUBDIR'],
                               file+".png")


@app.route('/books/<bookname>/<pagename>/comments_jump', methods=['POST'])
@login_required
def comments_jump(bookname, pagename):
    book = Book.query.filter_by(name=bookname).one()
    pnames = sorted([p.name for p in book.pages])
    data = request.json
    reverse = data["dir"] < 0
    if reverse:
        pnames.reverse()
    pnames = pnames[pnames.index(pagename) + 1:]
    result = {"page": "", "line": ""}
    for p in pnames:
        page = Page.query.filter_by(book_id=book.id, name=p).one()
        root = etree.fromstring(page.data)
        ns = {"ns": root.nsmap[None]}
        found = root.xpath('//ns:TextLine', namespaces=ns)
        if reverse:
            found.reverse()
        for textline in found:
            if "comments" in textline.attrib and textline.attrib["comments"]:
                result["page"] = p
                result["line"] = textline.attrib["id"]
                break
        if result["page"]:
            break
    return jsonify(result=result)


@app.route('/books/<bookname>/<pagename>/search_continue', methods=['POST'])
@login_required
def search_continue(bookname, pagename):
    book = Book.query.filter_by(name=bookname).one()
    pnames = sorted([p.name for p in book.pages])
    data = request.json
    reverse = data["dir"] < 0
    searchterm = data["searchterm"]
    if reverse:
        pnames.reverse()
    pnames = pnames[pnames.index(pagename) + 1:]
    result = {"page": "", "line": ""}
    for p in pnames:
        page = Page.query.filter_by(book_id=book.id, name=p).one()
        root = etree.fromstring(page.data)
        ns = {"ns": root.nsmap[None]}
        found = root.xpath('//ns:TextLine', namespaces=ns)
        if reverse:
            found.reverse()
        for textline in found:
            indz = textline.xpath('./ns:TextEquiv/@index', namespaces=ns)
            lowestindex = min(indz) if indz else ""
            textcontent = textline.find(
                './ns:TextEquiv[@index="{}"]/ns:Unicode'.format(lowestindex),
                namespaces=ns).text if lowestindex else ""
            if not textcontent:
                textcontent = ""
            comm = ""
            if data["comments"] and "comments" in textline.attrib:
                comm = textline.attrib["comments"]
            if searchterm in textcontent or searchterm in comm:
                result["page"] = p
                result["line"] = textline.attrib["id"]
                break
        if result["page"]:
            break
    return jsonify(result=result)


@app.route('/books/<bookname>/<pagename>/data', methods=['POST', 'GET'])
@login_required
def pagedata(bookname, pagename):
    pnamesplits = pagename.split("::")
    command = ""
    if len(pnamesplits) == 2:
        pagename, command = pnamesplits
    book = Book.query.filter_by(name=bookname).first()
    if command:
        plist = sorted([x.name for x in
                        Book.query.filter_by(name=bookname).first().pages])
        if command == "first":
            pagename = min(plist)
        elif command == "next":
            pagename = plist[(plist.index(pagename) + 1) % len(plist)]
        elif command == "prev":
            pagename = plist[(plist.index(pagename) - 1) % len(plist)]

    page = Page.query.filter_by(book_id=book.id, name=pagename).first()
    root = etree.fromstring(page.data)
    ns = {"ns": root.nsmap[None]}

    if request.method == "GET":
        if (request.args.get('download', '', type=str) == 'xml'):
            response = make_response(page.data)
            response.headers['Cache-Control'] = 'no-cache'
            response.headers['Content-Type'] = 'text/xml'
            response.headers['Content-Disposition'] =\
                "attachment; filename={}_{}.xml".format(bookname, pagename)
            return response

        pageattr = root.find(".//ns:Page", namespaces=ns).attrib
        image = {"file": pageattr["imageFilename"],
                 "image_x": pageattr["imageWidth"],
                 "image_y": pageattr["imageHeight"]}
        direction = "rtl" if bookname.endswith("_ar") else "ltr"
        regionmap = {}
        for r in root.findall(".//ns:TextRegion/ns:Coords[@points]",
                              namespaces=ns):
            textregion = r.getparent()
            r_id = textregion.attrib["id"]
            regionmap[r_id] = {}
            regionmap[r_id]["points"] = r.attrib["points"]
        pagemap = {}
        for l in root.findall(".//ns:TextLine/ns:Coords[@points]",
                              namespaces=ns):
            textline = l.getparent()
            l_id = textline.attrib["id"]
            pagemap[l_id] = {}
            pagemap[l_id]["points"] = l.attrib["points"]
            pagemap[l_id]["region"] = textline.getparent().attrib["id"]
            pagemap[l_id]["comments"] = textline.attrib["comments"] if \
                "comments" in textline.attrib else ""
            indz = textline.xpath('./ns:TextEquiv/@index', namespaces=ns)
            lowestindex = min(indz) if indz else ""
            textcontent = textline.find(
                './ns:TextEquiv[@index="{}"]/ns:Unicode'.format(lowestindex),
                namespaces=ns).text if lowestindex else ""
            status = "empty"
            if textcontent:
                status = "ocr" if int(lowestindex) else "gt"
            if textcontent is None:
                textcontent = ""
            pagemap[l_id]["text"] = {"status": status, "content": textcontent}
        return jsonify(page=pagename, image=image, lines=pagemap,
                       regions=regionmap, direction=direction)

    if request.method == "POST":
        data = request.json
        if current_user.is_anonymous:
            user = "user@nashi"
        else:
            user = current_user.email

        for l in [l for l in data["edits"] if l["action"] == "delete"]:
            cur = l["id"]
            ldata = l
            line = root.find('.//ns:TextLine[@id="'+cur+'"]', namespaces=ns)
            line.getparent().remove(line)

        for l in [l for l in data["edits"] if l["action"] == "create"]:
            cur = l["id"]
            ldata = l
            region = root.find('.//ns:TextRegion[@id="{}"]'.format(
                ldata["input"]["region"]), namespaces=ns)
            line = etree.SubElement(region, "{{{}}}TextLine".format(
                ns["ns"]), attrib={"id": cur})
            coords = etree.SubElement(line, "{{{}}}Coords".format(ns["ns"]),
                                      attrib={"points":
                                              ldata["input"]["points"]})

        for l in [l for l in data["edits"] if l["action"] == "change"]:
            cur = l["id"]
            ldata = l
            text = ldata["input"]["text"]["content"].strip()
            textstatus = ldata["input"]["text"]["status"]
            comments = ldata["input"]["comments"]
            points = ldata["input"]["points"]
            rid = ldata["input"]["region"]
            line = root.find('.//ns:TextLine[@id="'+cur+'"]',
                             namespaces=ns)
            line.attrib["comments"] = comments
            if textstatus == "edit":
                tequiv = line.find('.//ns:TextEquiv[@index="0"]',
                                   namespaces=ns)
                if tequiv is None:
                    tequiv = etree.SubElement(line, "{{{}}}TextEquiv"
                                              .format(ns["ns"]),
                                              attrib={"index": "0"})
                    unicodexml = etree.SubElement(tequiv, "Unicode")
                else:
                    unicodexml = tequiv.find('./ns:Unicode', namespaces=ns)
                tequiv.attrib["comments"] = "User: " + user
                unicodexml.text = text
            else:  # points changed
                coords = line.find('.//ns:Coords', namespaces=ns)
                coords.attrib["points"] = points

        page.no_lines_gt = int(root.xpath('count(//TextEquiv[@index="0"])')) +\
            int(root.xpath('count(//ns:TextEquiv[@index="0"])', namespaces=ns))
        page.no_lines_segm = int(root.xpath('count(//TextLine)')) + \
            int(root.xpath('count(//ns:TextLine)', namespaces=ns))
        page.data = etree.tounicode(root.getroottree())
        db_session.commit()
        return jsonify(lineinfo=str(page.no_lines_gt)
                       + "/" + str(page.no_lines_segm))


@app.route('/bootstrapexample')
@login_required
def bootstrapexample():
    return render_template('bootstrapexample.html')


@app.route('/larexredir/<bookname>')
@login_required
def larexredir(bookname):
    if not(len(glob(app.config["LAREX_DIR"]+"/"+bookname))):
        flash("Book not in LAREX folder. Copy images first.")
        return redirect(url_for("index"))
    try:
        page = requests.get(app.config['LAREX_URL_SERVER'])
    except requests.exceptions.ConnectionError:
        flash("No connection to LAREX server!")
        return redirect(url_for("index"))
    html_content = html.fromstring(page.content)
    ids = html_content.xpath('.//div[@id="lib"]//tr[td = "'+bookname+'"]/@id')
    if len(ids) == 1:
        return redirect(app.config['LAREX_URL_CLIENT']+"viewer?book="+ids[0])
    else:
        flash("Book not found in LAREX. Check your server configuration.")
        return redirect(url_for("index"))


@app.route('/status/<task_id>')
@login_required
def taskstatus(task_id):
    task = lareximport.AsyncResult(task_id)
    if task.state == 'PENDING':
        # job did not start yet
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', '')
        }
        if 'result' in task.info:
            response['result'] = task.info['result']
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }
    return jsonify(response)


@app.route('/loggedin')
@login_required
def loggedin():
    # authentification for nginx
    return ""
