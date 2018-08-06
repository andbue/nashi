"""
This is some development towards a command line client for OCR.
How to use it at the moment:

import argparse
from nashi.client import NashiClient, setup_train_args, params_from_args

ncl = NashiClient("https://my.nashi.server.de.vu/nashi")
ncl.login("bot@my.nashi.server.de.vu", "Secret123")
ncl.add_gt("Some_Book")

parser = argparse.ArgumentParser()
setup_train_args(parser, omit=["files", "validation"])
args = parser.parse_known_args()[0]
params = params_from_args(args)

ncl.train(params=params, training_to_validation=0.8)

ncl.add_lines("Some_Book")
ncl.predict("model.ckpt")
ncl.results_write_server(index=1)
"""

import argparse
import requests
import zipfile
import json
import gzip
from lxml import etree, html
from io import BytesIO
from os import path, mkdir
from PIL import Image
import numpy as np
from skimage.draw import polygon
from calamari_ocr.scripts.train import setup_train_args
from calamari_ocr.ocr.dataset import DataSet
from calamari_ocr.utils import parallel_map, split_all_ext
from calamari_ocr.utils.glob import glob_all
from calamari_ocr.utils.path import split_all_ext
from calamari_ocr.ocr.dataset import FileDataSet
from calamari_ocr.ocr.trainer import Trainer, Predictor
from calamari_ocr.ocr.data_processing.default_data_preprocessor\
    import DefaultDataPreprocessor
from calamari_ocr.ocr.text_processing import DefaultTextPreprocessor,\
    text_processor_from_proto, BidiTextProcessor#
from calamari_ocr.ocr.data_processing.default_data_preprocessor import DefaultDataPreprocessor
from calamari_ocr.ocr.text_processing import DefaultTextPreprocessor, text_processor_from_proto, BidiTextProcessor,\
default_text_normalizer_params, default_text_regularizer_params
from calamari_ocr.proto import CheckpointParams, DataPreprocessorParams, TextProcessorParams, \
network_params_from_definition_string, NetworkParams


def params_from_args(args):
    params = CheckpointParams()
    for a in ["max_iters", "stats_size", "batch_size", "checkpoint_frequency", "output_dir",
             "output_model_prefix", "display",  "early_stopping_nbest",
             "early_stopping_best_model_prefix"]:
        setattr(params, a, getattr(args, a))

    params.processes = args.num_threads
    params.skip_invalid_gt = not args.no_skip_invalid_gt
    params.early_stopping_frequency = args.early_stopping_frequency if args.early_stopping_frequency >= 0 else args.checkpoint_frequency
    params.early_stopping_best_model_output_dir = \
        args.early_stopping_best_model_output_dir if args.early_stopping_best_model_output_dir else args.output_dir

    params.model.data_preprocessor.type = DataPreprocessorParams.DEFAULT_NORMALIZER
    params.model.data_preprocessor.line_height = args.line_height
    params.model.data_preprocessor.pad = args.pad

    # Text pre processing (reading)
    params.model.text_preprocessor.type = TextProcessorParams.MULTI_NORMALIZER
    default_text_normalizer_params(params.model.text_preprocessor.children.add(), default=args.text_normalization)
    default_text_regularizer_params(params.model.text_preprocessor.children.add(), groups=args.text_regularization)
    strip_processor_params = params.model.text_preprocessor.children.add()
    strip_processor_params.type = TextProcessorParams.STRIP_NORMALIZER

    # Text post processing (prediction)
    params.model.text_postprocessor.type = TextProcessorParams.MULTI_NORMALIZER
    default_text_normalizer_params(params.model.text_postprocessor.children.add(), default=args.text_normalization)
    default_text_regularizer_params(params.model.text_postprocessor.children.add(), groups=args.text_regularization)
    strip_processor_params = params.model.text_postprocessor.children.add()
    strip_processor_params.type = TextProcessorParams.STRIP_NORMALIZER

    if args.seed > 0:
        params.model.network.backend.random_seed = args.seed

    if args.bidi_dir:
        # change bidirectional text direction if desired
        bidi_dir_to_enum = {"rtl": TextProcessorParams.BIDI_RTL, "ltr": TextProcessorParams.BIDI_LTR,
                            "auto": TextProcessorParams.BIDI_AUTO}

        bidi_processor_params = params.model.text_preprocessor.children.add()
        bidi_processor_params.type = TextProcessorParams.BIDI_NORMALIZER
        bidi_processor_params.bidi_direction = bidi_dir_to_enum[args.bidi_dir]

        bidi_processor_params = params.model.text_postprocessor.children.add()
        bidi_processor_params.type = TextProcessorParams.BIDI_NORMALIZER
        bidi_processor_params.bidi_direction = TextProcessorParams.BIDI_AUTO

    params.model.line_height = args.line_height

    network_params_from_definition_string(args.network, params.model.network)
    params.model.network.clipping_mode = NetworkParams.ClippingMode.Value("CLIP_" + args.gradient_clipping_mode.upper())
    params.model.network.clipping_constant = args.gradient_clipping_const
    params.model.network.backend.fuzzy_ctc_library_path = args.fuzzy_ctc_library_path
    params.model.network.backend.num_inter_threads = args.num_inter_threads
    params.model.network.backend.num_intra_threads = args.num_intra_threads

    return params


def cutout(pageimg, coordstring, scale=1):
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
    return box


class NashiDataSet(DataSet):
    def __init__(self, linelist, session, baseurl):
        super().__init__()
        self.session = session
        self.baseurl = baseurl
        for l in linelist:
            self._samples.append(l)

    def load_samples(self, processes=1, progress_bar=False):
        if self.loaded:
            return self._samples

        imgfiles = list(set([l["imgfile"] for l in self._samples]))

        data = parallel_map(self._load_page, imgfiles, desc="Loading Dataset",
                            processes=processes, progress_bar=progress_bar)

        for images, imgfile in zip(data, imgfiles):
            for image, sample in zip(images, [s for s in self._samples
                                              if s["imgfile"] == imgfile]):
                sample["image"] = image

        self.loaded = True

        return self._samples

    def _load_page(self, imgfile):
        imgresp = self.session.get(self.baseurl+"/books/{}".format(imgfile),
                                   params={"upgrade": "nrm"})
        f = BytesIO(imgresp.content)
        im = Image.open(f)
        pageimg = np.array(im)
        f.close()
        if len(pageimg.shape) > 2:
            pageimg = pageimg[:, :, 0]
        return [self._load_line(pageimg, s["coords"],
                                pageimg.shape[1] / s["img_width"])
                for s in self._samples if s["imgfile"] == imgfile]

    def _load_line(self, pageimg, coords, scale=1):
        return cutout(pageimg, coords, scale=scale)

    def _load_sample(self, sample):
        pass


class NashiClient():
    def __init__(self, baseurl):
        self.baseurl = baseurl
        self.session = None
        self.traindata = None
        self.recogdata = None
        self.valdata = None
        self.bookcache = {}

    def login(self, email, pw):
        s = requests.Session()
        r = s.get(self.baseurl + "/login")
        res = html.document_fromstring(r.text)
        csrf = res.get_element_by_id("csrf_token").attrib["value"]
        lg = s.post(self.baseurl+"/login", data={
               "csrf_token": csrf,
               "email": email,
               "password": pw,
               "submit": "Login"
            })
        self.session = s

    def add_gt(self, books, layer=0, skipcommented=True):
        if type(books) == str:
            books = [books]
        lines = []
        for b in books:
            if b not in self.bookcache:
                self.bookcache[b] = self.getbook(b)
            lines = self._lines_gt_frombook(self.bookcache[b], b, lines,
                                            layer=layer,
                                            skipcommented=skipcommented)
        if not self.traindata:
            self.traindata = NashiDataSet(lines, self.session, self.baseurl)
        else:
            for l in lines:
                if l not in self.traindata._samples:
                    self.traindata._samples.append(l)

    def add_lines(self, books):
        if type(books) == str:
            books = [books]
        lines = []
        for b in books:
            if b not in self.bookcache:
                self.bookcache[b] = self.getbook(b)
            lines = self._lines_frombook(self.bookcache[b], b, lines)
        if not self.recogdata:
            self.recogdata = NashiDataSet(lines, self.session, self.baseurl)
        else:
            for l in lines:
                if l not in self.recogdata._samples:
                    self.recogdata._samples.append(l)


    def getbook(self, bookname):
        pagezip = self.session.get(self.baseurl+"/books/{}_PageXML.zip"
                                   .format(bookname))
        f = BytesIO(pagezip.content)
        zf = zipfile.ZipFile(f)
        book = {}
        for fn in zf.namelist():
            filename = fn
            pagename = path.splitext(path.split(filename)[1])[0]
            with zf.open(fn) as fo:
                book[pagename] = etree.parse(fo).getroot()
        f.close()
        return(book)

    def _lines_gt_frombook(self, bookdict, bookname, lines=[], layer=0,
                           skipcommented=True):
        for pagename, root in bookdict.items():
            ns = {"ns": root.nsmap[None]}
            imgfile = root.xpath('//ns:Page',
                                 namespaces=ns)[0].attrib["imageFilename"]
            img_w = int(root.xpath('//ns:Page',
                                   namespaces=ns)[0].attrib["imageWidth"])
            tequivs = root.xpath('//ns:TextEquiv[@index="{}"]'.format(layer),
                                 namespaces=ns)
            for l in tequivs:
                parat = l.getparent().attrib
                if skipcommented and "comments" in parat and parat["comments"]:
                    continue
                lines.append({
                    "rtype": l.xpath('../../@type', namespaces=ns).pop(),
                    "book": bookname,
                    "page": pagename,
                    "imgfile": bookname+"/"+imgfile,
                    "id": l.xpath('../@id', namespaces=ns).pop(),
                    "text": l.xpath('./ns:Unicode',
                                    namespaces=ns).pop().text,
                    "coords": l.xpath('../ns:Coords/@points',
                                      namespaces=ns).pop(),
                    "img_width": img_w
                })
        return lines

    def _lines_frombook(self, bookdict, bookname, lines=[]):
        for pagename, root in bookdict.items():
            ns = {"ns": root.nsmap[None]}
            imgfile = root.xpath('//ns:Page',
                                 namespaces=ns)[0].attrib["imageFilename"]
            img_w = int(root.xpath('//ns:Page',
                                   namespaces=ns)[0].attrib["imageWidth"])
            for l in root.xpath('//ns:TextLine', namespaces=ns):
                lines.append({
                    "rtype": l.xpath('../@type', namespaces=ns).pop(),
                    "book": bookname,
                    "page": pagename,
                    "imgfile": bookname+"/"+imgfile,
                    "id": l.xpath('./@id', namespaces=ns).pop(),
                    "coords": l.xpath('./ns:Coords/@points',
                                      namespaces=ns).pop(),
                    "img_width": img_w
                })
        return lines

    def train(self, params, weights=None, training_to_validation=1):
        if 0<training_to_validation<1 and not self.valdata:
            valsamples = random.sample(self.traindata._samples,
                                       int((1-training_to_validation)*len(self.traindata)))
            for s in valsamples:
                self.traindata._samples.remove(s)
            self.valdata = NashiDataSet([], self.session, self.baseurl)
            self.valdata._samples = valsamples
        trainer = Trainer(params, self.traindata, validation_dataset=self.valdata, weights=weights)
        trainer.train(progress_bar=True)

    def predict(self, checkpoint, progress_bar=True):
        predictor = Predictor(checkpoint=checkpoint)
        self.recogdata.load_samples(progress_bar=progress_bar)
        #lines = [l for l in ncl.recogdata._samples if l["rtype"] == "paragraph"]
        res = predictor.predict_raw([l["image"] for l in self.recogdata._samples])
        for sample, r in zip(self.recogdata._samples, res):
            if r.sentence:
                sample["pred"] = r.sentence

    def results_write_xml(self, xmldir=".", index=1):
        savelines = [l for l in self.recogdata._samples if "pred" in l]
        books = sorted(list(set([l["book"] for l in savelines])))
        for b in books:
            if b not in self.bookcache:
                self.bookcache[b] = self.getbook(b)
            if not path.isdir(xmldir+"/"+b):
                mkdir(xmldir+"/"+b)
            pages = sorted(list(set([l["page"] for l in savelines
                                         if l["book"] == b])))
            for p in pages:
                root = self.bookcache[b][p]
                ns = {"ns": root.nsmap[None]}
                for l in [l for l in savelines if l["book"] == b and l["page"] == p]:
                    linexml = root.find('.//ns:TextLine[@id="'+l["id"]+'"]',
                                        namespaces=ns)
                    textequivxml = linexml.find('./ns:TextEquiv[@index="{}"]'.format(index),
                                                namespaces=ns)
                    if textequivxml is None:
                        textequivxml = etree.SubElement(linexml, "TextEquiv",
                                                        attrib={"index": str(index)})
                    unicodexml = textequivxml.find('./ns:Unicode', namespaces=ns)
                    if unicodexml is None:
                        unicodexml = etree.SubElement(textequivxml, "Unicode")
                    unicodexml.text = l["pred"]
                with open(xmldir + "/" + b + "/" + p + ".xml", "w") as f:
                    f.write(etree.tounicode(root.getroottree()))

    def results_write_server(self, index=1):
        ocrdata = {}
        savelines = [l for l in ncl.recogdata._samples if "pred" in l]
        books = sorted(list(set([l["book"] for l in savelines])))
        for b in books:
            if b not in ocrdata:
                ocrdata[b] = {}
            pages = sorted(list(set([l["page"] for l in savelines
                                     if l["book"] == b])))
            for p in pages:
                if p not in ocrdata[b]:
                    ocrdata[b][p] = {}
                for l in [l for l in savelines if l["book"] == b and l["page"] == p]:
                    ocrdata[b][p][l["id"]] = l["pred"]
        data = {"ocrdata": ocrdata, "index": index}
        self.session.post(self.baseurl+"/_ocrdata",
                          data=gzip.compress(json.dumps(data).encode("utf-8")),
                          headers={"Content-Type": "application/json;charset=UTF-8",
                                   "Content-Encoding": "gzip"})
