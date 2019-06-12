"""
A simple client to OCR texts transcribed in nashi
"""

import argparse
import requests
import zipfile
import json
import gzip
import random
import h5py
import numpy as np
from lxml import etree, html
from io import BytesIO
from os import path
from PIL import Image, ImageFile
from getpass import getpass
from skimage.draw import polygon

from calamari_ocr.ocr import MultiPredictor, Trainer, Predictor, Evaluator
from calamari_ocr.ocr.data_processing import MultiDataProcessor, DataRangeNormalizer,\
    FinalPreparation, CenterNormalizer, NoopDataPreprocessor
from calamari_ocr.ocr.datasets import DataSet, DataSetMode, DatasetGenerator, RawDataSet
from calamari_ocr.ocr.text_processing import default_text_normalizer_params,\
    default_text_regularizer_params, text_processor_from_proto, NoopTextProcessor
from calamari_ocr.ocr.voting import voter_from_proto
from calamari_ocr.proto import DataPreprocessorParams, TextProcessorParams, VoterParams,\
    CheckpointParams, network_params_from_definition_string, NetworkParams
from calamari_ocr.scripts.train import setup_train_args

ImageFile.LOAD_TRUNCATED_IMAGES = True # Otherwise problems with larger images.

def params_from_args(args):
    """
    Turn args to calamari into params
    """
    params = CheckpointParams()
    for attr in ["max_iters", "stats_size", "batch_size", "checkpoint_frequency", "output_dir",
                 "output_model_prefix", "display", "early_stopping_nbest",
                 "early_stopping_best_model_prefix"]:
        setattr(params, attr, getattr(args, attr))

    params.processes = args.num_threads
    params.skip_invalid_gt = not args.no_skip_invalid_gt
    params.early_stopping_frequency = args.early_stopping_frequency\
        if args.early_stopping_frequency >= 0 else args.checkpoint_frequency
    params.early_stopping_best_model_output_dir = args.early_stopping_best_model_output_dir\
        if args.early_stopping_best_model_output_dir else args.output_dir

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


def cutout(pageimg, coordstring, scale=1, rect=False):
    """ Cut region from image
    Parameters
    ----------
    pageimg : image (numpy array)
    coordstring : coordinates from PAGE as one string
    scale : factor to scale the coordinates with
    rect : cut out rectangle instead of polygons
    """
    coords = [p.split(",") for p in coordstring.split()]
    coords = np.array([(int(scale*int(c[1])), int(scale*int(c[0])))
                       for c in coords])
    if rect:
        return pageimg[min(c[0] for c in coords):max(c[0] for c in coords),
                       min(c[1] for c in coords):max(c[1] for c in coords)]
    rr, cc = polygon(coords[:, 0], coords[:, 1], pageimg.shape)
    offset = (min([x[0] for x in coords]), min([x[1] for x in coords]))
    box = np.ones(
        (max([x[0] for x in coords]) - offset[0],
         max([x[1] for x in coords]) - offset[1]),
        dtype=pageimg.dtype) * 255
    box[rr-offset[0], cc-offset[1]] = pageimg[rr, cc]
    return box

            
class Nash5DataSetGenerator(DatasetGenerator):
    def __init__(self, mp_context, output_queue, mode, samples, cachefile):
        super().__init__(mp_context, output_queue, mode, samples)
        self.cachefile = cachefile

    def cacheopen(self):
        if isinstance(self.cachefile, str):
            self.cachefile = h5py.File(self.cachefile, "r", libver='latest', swmr=True)
        return self.cachefile

    def _load_sample(self, sample, text_only=False):
        s = self.cacheopen()[sample["id"]]
        if text_only:
            yield None, s.attrs.get("text")
        else:
            yield s[()], s.attrs.get("text")


class Nash5DataSet(DataSet):
    def __init__(self, mode: DataSetMode, ncache, books):
        """ Create a dataset from nashi cache
        Parameters
        ----------
        ncache : for training: filename of hdf5-cache
                 for prediction: h5py File object
        books : book title or list of titles
        """
        super().__init__(mode)
        self.predictions = {} if self.mode == DataSetMode.PREDICT else None
        self.cachefile = ncache
        if isinstance(books, str):
            books = [books]

        with h5py.File(self.cachefile, "r", libver='latest', swmr=True) as cache:
            for b in books:
                for p in cache[b]:
                    for s in cache[b][p]:
                        if self.mode == DataSetMode.TRAIN and "comments" in cache[b][p][s].attrs:
                            continue
                        if mode in [DataSetMode.TRAIN, DataSetMode.EVAL]\
                                and cache[b][p][s].attrs.get("text") is not None:
                            self.add_sample(cache[b][p][s])
                        elif mode == DataSetMode.PREDICT and cache[b][p][s].attrs.get("text") is None:
                            self.add_sample(cache[b][p][s])

    def add_sample(self, sample):
        self._samples.append({"id": sample.name})

    def store_text(self, sentence, sample, output_dir, extension):
        self.predictions[sample["id"]] = sentence

    def store(self):
        with h5py.File(self.cachefile, "a", libver='latest') as cache:
            cache.swmr_mode = True
            for p in self.predictions:
                cache[p].attrs["pred"] = self.predictions[p]

    def create_generator(self, mp_context, output_queue):
        return Nash5DataSetGenerator(mp_context, output_queue, self.mode, self.samples(), self.cachefile)
    
    

class NashiClient():
    def __init__(self, baseurl, cachefile, login, password=None):
        """ Create a nashi client
        Parameters
        ----------
        baseurl : web address of nashi instance
        cachefile : filename of hdf5-cache
        login : user for nashi
        password : asks for user input if empty
        """
        self.baseurl = baseurl
        self.session = None
        self.traindata = None
        self.recogdata = None
        self.valdata = None
        self.bookcache = {}
        self.cachefile = cachefile
        self.login(login, password)

        params = DataPreprocessorParams()
        params.line_height = 48
        params.pad = 16
        params.pad_value = 1
        params.no_invert = False
        params.no_transpose = False
        self.data_proc = MultiDataProcessor([
            DataRangeNormalizer(),
            CenterNormalizer(params),
            FinalPreparation(params, as_uint8=True),
        ])

        # Text pre processing (reading)
        preproc = TextProcessorParams()
        preproc.type = TextProcessorParams.MULTI_NORMALIZER
        default_text_normalizer_params(preproc.children.add(), default="NFC")
        default_text_regularizer_params(preproc.children.add(), groups=["extended"])
        strip_processor_params = preproc.children.add()
        strip_processor_params.type = TextProcessorParams.STRIP_NORMALIZER
        self.txt_preproc = text_processor_from_proto(preproc, "pre")

        # Text post processing (prediction)
        postproc = TextProcessorParams()
        postproc.type = TextProcessorParams.MULTI_NORMALIZER
        default_text_normalizer_params(postproc.children.add(), default="NFC")
        default_text_regularizer_params(postproc.children.add(), groups=["extended"])
        strip_processor_params = postproc.children.add()
        strip_processor_params.type = TextProcessorParams.STRIP_NORMALIZER
        self.text_postproc = text_processor_from_proto(postproc, "post")

        # BIDI text preprocessing
        bidi_processor_params = preproc.children.add()
        bidi_processor_params.type = TextProcessorParams.BIDI_NORMALIZER
        bidi_processor_params.bidi_direction = TextProcessorParams.BIDI_RTL
        self.bidi_preproc = text_processor_from_proto(preproc, "pre")

        bidi_processor_params = postproc.children.add()
        bidi_processor_params.type = TextProcessorParams.BIDI_NORMALIZER
        bidi_processor_params.bidi_direction = TextProcessorParams.BIDI_AUTO
        self.bidi_postproc = text_processor_from_proto(postproc, "post")


    def login(self, email, pw):
        if pw is None:
            pw = getpass("Password: ")
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
        if "<li>Invalid password</li>" in lg.text:
            raise Exception("Login failed.")
        self.session = s


    def update_books(self, books, gt_layer=0, rect=False, rtl=False):
        """ Update books cache
        Parameters
        ----------
        books : book title or list of titles
        gt_layer : index of ground truth in PAGE files
        rect : cut out rectangles instead of line polygons
        rtl : set text direction to rtl
        """
        cache = self.cache = h5py.File(self.cachefile, 'a', libver='latest')
        if isinstance(books, str):
            books = [books]
        icnt, tcnt = 0, 0
        for b in books:
            print("Updating {}…".format(b))
            if b.endswith("_ar") and not rtl:
                print("Warning: Title ends with _ar but rtl is not set!")
            if b.endswith("_ar") and not rect:
                print("Warning: Title ends with _ar but rect is not set!")
            book = self.getbook(b)
            # remove pages not contained in the nashi book
            if b in cache:
                for p in cache.get(b):
                    if p not in book:
                        _ = cache[b].pop(p)
            else:
                cache.create_group(b)
            cache[b].attrs["dir"] = "rtl" if rtl else "ltr"
            for p, root in book.items():
                icnt_this, tcnt_this = 0, 0
                print(p, end="… ")
                ns = {"ns": root.nsmap[None]}
                if p not in cache[b]:
                    cache.create_group(b+"/"+p)
                cache[b][p].attrs["img_w"] = int(root.xpath('//ns:Page',
                                                            namespaces=ns)[0].attrib["imageWidth"])
                cache[b][p].attrs["image_file"] = root.xpath('//ns:Page',
                                                             namespaces=ns)[0].attrib["imageFilename"]
                pageimg = None
                lines = root.xpath('//ns:TextLine', namespaces=ns)
                lids = [l.attrib["id"] for l in lines]

                # remove lines not contained in the page anymore
                for lid in cache[b][p]:
                    if lid not in lids:
                        _ = cache[b][p].pop(lid)

                for l in lines:
                    coords = l.xpath('./ns:Coords', namespaces=ns).pop().attrib.get("points")
                    lid = l.attrib["id"]
                    # update line image if coords changed
                    if lid not in cache[b][p] \
                            or cache[b][p][lid].attrs.get("coords") != coords:
                        icnt_this += 1
                        if pageimg is None:
                            imgresp = self.session.get(self.baseurl+"/books/{}/{}".format(
                                b, cache[b][p].attrs["image_file"]), params={"upgrade": "nrm"})
                            f = BytesIO(imgresp.content)
                            im = Image.open(f)
                            pageimg = np.array(im)
                            f.close()
                            if len(pageimg.shape) > 2:
                                pageimg = pageimg[:, :, 0]
                            if pageimg.dtype == bool:
                                pageimg = pageimg.astype("uint8") * 255
                        limg = cutout(pageimg, coords,
                                      scale=pageimg.shape[1] / cache[b][p].attrs["img_w"], rect=rect)

                        limg = self.data_proc.apply(limg)[0]

                        if lid not in cache[b][p]:
                            cache[b][p].create_dataset(lid, data=limg, maxshape=(None, 48))
                        else:
                            if cache[b][p][lid].shape != limg.shape:
                                cache[b][p][lid].resize(limg.shape)
                            cache[b][p][lid][:, :] = limg
                        cache[b][p][lid].attrs["coords"] = coords
                        
                    comments = l.attrib.get("comments")
                    if comments is not None and comments.strip():
                        cache[b][p][lid].attrs["comments"] = comments.strip()
                    rtype = l.getparent().attrib.get("type")
                    cache[b][p][lid].attrs["rtype"] = rtype if rtype is not None else ""

                    ucd = l.xpath('./ns:TextEquiv[@index="{}"]/ns:Unicode'.format(gt_layer),
                                  namespaces=ns)
                    rawtext = ucd[0].text if ucd else None
                    if rawtext != cache[b][p][lid].attrs.get("text_raw"):
                        tcnt_this += 1
                        cache[b][p][lid].attrs["text_raw"] = rawtext
                        preproc = self.bidi_preproc if rtl else self.txt_preproc
                        cache[b][p][lid].attrs["text"] = preproc.apply(rawtext)
                icnt += icnt_this
                tcnt += tcnt_this
                print("i: {} / t: {}".format(icnt_this, tcnt_this))
                cache.flush()
        cache.close()


    def train_books(self, books, output_model_prefix, weights=None, train_to_val=1, codec_whitelist=None,
                    max_iters=100000, display=500, checkpoint_frequency=-1, preload=False):
        if isinstance(books, str):
            books = [books]
        dset = Nash5DataSet(DataSetMode.TRAIN, self.cachefile, books)
        if 0 < train_to_val < 1:
            valsamples = random.sample(dset._samples,
                                       int((1-train_to_val)*len(dset)))
            for s in valsamples:
                dset._samples.remove(s)
            vdset = Nash5DataSet(DataSetMode.TRAIN, self.cachefile, [])
            vdset._samples = valsamples
        else:
            vdset = None

        parser = argparse.ArgumentParser()
        setup_train_args(parser, omit=["files", "validation"])
        args = parser.parse_known_args()[0]
        with h5py.File(self.cachefile, 'r', libver='latest', swmr=True) as cache:
            if all(cache[b].attrs.get("dir") == "rtl" for b in books):
                args.bidi_dir = "rtl"
        params = params_from_args(args)
        params.output_model_prefix = output_model_prefix
        params.early_stopping_best_model_prefix = "best_" + output_model_prefix
        params.max_iters = max_iters
        params.display = display
        params.checkpoint_frequency = checkpoint_frequency

        trainer = Trainer(params, dset, txt_preproc=NoopTextProcessor(), data_preproc=NoopDataPreprocessor(),
                  validation_dataset=vdset, weights=weights, preload_training=preload, preload_validation=True,
                  codec_whitelist=codec_whitelist)

        trainer.train(progress_bar=True, auto_compute_codec=True)


    def predict_books(self, books, models, pageupload=False, text_index=1):
        if type(books) == str:
            books = [books]
        if type(models) == str:
            models = [models]
        dset = Nash5DataSet(DataSetMode.PREDICT, self.cachefile, books)

        voter_params = VoterParams()
        voter_params.type = VoterParams.Type.Value("confidence_voter_default_ctc".upper())
        voter = voter_from_proto(voter_params)

        # predict for all models
        predictor = MultiPredictor(checkpoints=models, data_preproc=NoopDataPreprocessor(), batch_size=1, processes=1)
        do_prediction = predictor.predict_dataset(dset, progress_bar=True)

        avg_sentence_confidence = 0
        n_predictions = 0
        # output the voted results to the appropriate files
        for result, sample in do_prediction:
            n_predictions += 1
            for i, p in enumerate(result):
                p.prediction.id = "fold_{}".format(i)

            # vote the results (if only one model is given, this will just return the sentences)
            prediction = voter.vote_prediction_result(result)
            prediction.id = "voted"
            sentence = prediction.sentence
            avg_sentence_confidence += prediction.avg_char_probability

            dset.store_text(sentence, sample, output_dir=None, extension=".pred.txt")
        print("Average sentence confidence: {:.2%}".format(avg_sentence_confidence / n_predictions))

        dset.store()
        print("All files written")

        
    def evaluate_books(self, books, models, rtl=False, mode="auto", sample=-1):
        if type(books) == str:
            books = [books]
        if type(models) == str:
            models = [models]
        results = {}
        if mode == "auto":
            with h5py.File(self.cachefile, 'r', libver='latest', swmr=True) as cache:
                for b in books:
                    for p in cache[b]:
                        for s in cache[b][p]:
                            if "text" in cache[b][p][s].attrs:
                                mode = "eval"
                                break
                        if mode != "auto":
                            break
                    if mode != "auto":
                        break
            if mode == "auto":
                mode = "conf"
                
        if mode == "conf":
            dset = Nash5DataSet(DataSetMode.PREDICT, self.cachefile, books)
        else:
            dset = Nash5DataSet(DataSetMode.TRAIN, self.cachefile, books)
            dset.mode = DataSetMode.PREDICT # otherwise results are randomised
        
        if 0 < sample < len(dset):
            delsamples = random.sample(dset._samples, len(dset) - sample)
            for s in delsamples:
                dset._samples.remove(s)

        if mode == "conf":
            dset = dset.to_raw_input_dataset(processes=1, progress_bar=True)
            for model in models:
                if isinstance(model, str):
                    model = [model]
                predictor = MultiPredictor(checkpoints=model, data_preproc=NoopDataPreprocessor(), batch_size=1, processes=1)
                voter_params = VoterParams()
                voter_params.type = VoterParams.Type.Value("confidence_voter_default_ctc".upper())
                voter = voter_from_proto(voter_params)
                do_prediction = predictor.predict_dataset(dset, progress_bar=True)
                avg_sentence_confidence = 0
                n_predictions = 0
                for result, sample in do_prediction:
                    n_predictions += 1
                    prediction = voter.vote_prediction_result(result)
                    avg_sentence_confidence += prediction.avg_char_probability
                
                results["/".join(model)] = avg_sentence_confidence / n_predictions

        else:
            for model in models:
                if isinstance(model, str):
                    model = [model]
                    
                predictor = MultiPredictor(checkpoints=model, data_preproc=NoopDataPreprocessor(), 
                                           batch_size=1, processes=1)

                voter_params = VoterParams()
                voter_params.type = VoterParams.Type.Value("confidence_voter_default_ctc".upper())
                voter = voter_from_proto(voter_params)

                out_gen = predictor.predict_dataset(dset, progress_bar=True)

                preproc = self.bidi_preproc if rtl else self.txt_preproc
                
                pred_dset = RawDataSet(DataSetMode.EVAL, texts=preproc.apply([
                    voter.vote_prediction_result(d[0]).sentence for d in out_gen
                ]))


                evaluator = Evaluator(text_preprocessor=NoopTextProcessor(), skip_empty_gt=False)
                r = evaluator.run(gt_dataset=dset, pred_dataset=pred_dset, processes=1,
                                  progress_bar=True)    
                    
                results["/".join(model)] = 1 - r["avg_ler"]
        return results
            


    def upload_books(self, books, text_index=1):
        """ Upload books from the cachefile to the server
        Parameters
        ----------
        bookname : title of the book or list of titles
        text_index : index of the TextEquiv to write to

        Returns
        ----------
        dict mapping page names to lxml etree instances
        """
        cache = h5py.File(self.cachefile, 'r', libver='latest', swmr=True)
        ocrdata = {}
        if type(books) == str:
            books = [books]
        savelines = [cache[b][p][l] for b in books for p in cache[b] for l in cache[b][p]
                     if cache[b][p][l].attrs.get("pred") is not None]
        for line in savelines:
            _, b, p, l = line.name.split("/")
            if b not in ocrdata:
                ocrdata[b] = {}
            if p not in ocrdata[b]:
                ocrdata[b][p] = {}
            ocrdata[b][p][l] = line.attrs.get("pred")

        data = {"ocrdata": ocrdata, "index": text_index}
        self.session.post(self.baseurl+"/_ocrdata",
                          data=gzip.compress(json.dumps(data).encode("utf-8")),
                          headers={"Content-Type": "application/json;charset=UTF-8",
                                   "Content-Encoding": "gzip"})
        cache.close()


    def getbook(self, bookname):
        """ Download a book from the nashi server
        Parameters
        ----------
        bookname : title of the book to load

        Returns
        ----------
        dict mapping page names to lxml etree instances
        """
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
        return book
