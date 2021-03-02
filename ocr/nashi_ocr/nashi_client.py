"""
A simple client to OCR texts transcribed in nashi
"""

import cv2 as cv
import gzip
import h5py
import json
import numpy as np
import requests
import random
import zipfile
from bidi.algorithm import get_base_level
from getpass import getpass
from io import BytesIO
from lxml import etree, html
from multiprocessing import Pool
from os import path
from tqdm import tqdm

from calamari_ocr.ocr import PipelineParams
from calamari_ocr.ocr.augmentation.dataaugmentationparams import DataAugmentationAmount
from calamari_ocr.ocr.dataset.data import Data
from calamari_ocr.scripts.eval import print_confusions
from calamari_ocr.ocr.dataset.datareader.base import DataReader
from calamari_ocr.ocr.dataset.datareader.factory import DataReaderFactory, FileDataReaderArgs
from calamari_ocr.ocr.dataset.datareader.pagexml.reader import PageXMLReader, CutMode
from calamari_ocr.ocr.dataset.imageprocessors import AugmentationProcessor, PrepareSampleProcessor
from calamari_ocr.ocr.dataset.imageprocessors.default_image_processors import default_image_processors
from calamari_ocr.ocr.dataset.params import InputSample, DataParams, SampleMeta
from calamari_ocr.ocr.dataset.postprocessors.ctcdecoder import CTCDecoderProcessor
from calamari_ocr.ocr.dataset.postprocessors.reshape import ReshapeOutputsProcessor
from calamari_ocr.ocr.dataset.textprocessors import TextNormalizer, TextRegularizer, StripTextProcessor, BidiTextProcessor
from calamari_ocr.ocr.dataset.textprocessors.text_regularizer import default_text_regularizer_replacements
from calamari_ocr.ocr.model.ctcdecoder.ctc_decoder import CTCDecoderParams
from calamari_ocr.ocr.predict.params import PredictorParams
from calamari_ocr.ocr.scenario import Scenario
from calamari_ocr.ocr.training.params import params_from_definition_string, TrainerParams
from calamari_ocr.ocr.voting.params import VoterParams, VoterType
from calamari_ocr.utils import glob_all
from tfaip.base.data.pipeline.definitions import DataProcessorFactoryParams
from tfaip.base.data.pipeline.datapipeline import SamplePipelineParams

from tfaip.base.data.pipeline.definitions import INPUT_PROCESSOR, PipelineMode, TARGETS_PROCESSOR
import tfaip.util.logging
logger = tfaip.util.logging.logger(__name__)


def lids_from_books(books, cachefile, new_only=False, complete_only=False, skip_commented=False):
    with h5py.File(cachefile, "r", libver='latest', swmr=True) as cache:
        for b in set(books):
            for p in cache[b]:
                for s in cache[b][p]:
                    sample = cache[b][p][s]
                    if skip_commented and "comments" in sample.attrs:
                        continue
                    if complete_only and sample.attrs.get("text") in (None, ""):
                        continue
                    if new_only and sample.attrs.get("text") not in (None, ""):
                        continue
                    yield sample.name


class Nsh5DataReader(DataReader):
    def __init__(self, mode: PipelineMode, images=[], texts=[],
                 skip_invalid=False,
                 remove_invalid=False,
                 non_existing_as_empty=False,
                 args=None):
        """ Create a dataset from nashi cache
        Parameters
        ----------
        images : [path to h5py file]
        texts : ["/path/to/line1", "/path/to/line2", ...] (iterable of sample ids in h5py file)
        """
        super().__init__(mode, skip_invalid, remove_invalid)

        self.predictions = {} if self.mode == PipelineMode.Prediction else None
        self.cachefile = images[0]

        for sid in texts:
            self.add_sample({"id": sid})

    def store_text(self, sentence, sample, output_dir, extension):
        self.predictions[sample["id"]] = sentence

    def store(self):
        self.cacheclose()
        with h5py.File(self.cachefile, "r+", libver='latest') as cache:
            cache.swmr_mode = True
            for p in self.predictions:
                cache[p].attrs["pred"] = self.predictions[p]

    def cacheopen(self):
        if isinstance(self.cachefile, str):
            self.cachefile = h5py.File(self.cachefile, "r",
                                       libver='latest', swmr=True)
        return self.cachefile

    def cacheclose(self):
        if not isinstance(self.cachefile, str):
            fn = self.cachefile.filename
            self.cachefile.close()
            self.cachefile = fn

    def _load_sample(self, sample, text_only):
        cache = self.cacheopen()
        s = cache[sample["id"]]
        text = s.attrs.get("text")
        if text_only:
            yield InputSample(None,
                              text,
                              SampleMeta(sample['id'], fold_id=sample['fold_id']),
                              )
        else:
            im = np.empty(s.shape, s.dtype)
            s.read_direct(im)
            yield InputSample(im.T,
                              text,
                              SampleMeta(
                                  sample['id'], fold_id=sample['fold_id']),
                              )

        def __del__(self):
            self.cacheclose()


DataReaderFactory.CUSTOM_READERS["Nsh5"] = Nsh5DataReader


def cutout(pageimg, coordstring, scale=1, rect=False, rrect=False):
    """ Cut region from image
    Parameters
    ----------
    pageimg : image (numpy array)
    coordstring : coordinates from PAGE as one string
    scale : factor to scale the coordinates with
    rect : cut out rectangle instead of polygons
    rrect : cut minimum enclosing rectangle instead of polygons
    """
    mode = CutMode.POLYGON
    if rect:
        mode = CutMode.BOX
    if rrect:
        mode = CutMode.MBR
    return PageXMLReader.cutout(pageimg, coordstring, mode=mode, angle=0, cval=None, scale=scale)


def get_preprocs(bidi_dir="L", pad=0):
    '''Construct preprocessor functions.
     bidi_dir in ("" (no bidi), None (->from input), L, R).'''

    params: TrainerParams = Scenario.default_trainer_params()

    # =================================================================================================================
    # Data Params
    data_params: DataParams = params.scenario_params.data_params
    data_params.train = PipelineParams(
        skip_invalid=False,
        remove_invalid=True,
        batch_size=1,
        num_processes=1,
    )

    data_params.pre_processors_ = SamplePipelineParams(run_parallel=True)

    for p in [p.name for p in default_image_processors()]:
        p_p = Data.data_processor_factory().processors[p].default_params()
        if 'pad' in p_p:
            p_p['pad'] = pad
        data_params.pre_processors_.sample_processors.append(DataProcessorFactoryParams(p, INPUT_PROCESSOR, p_p))

    # Text pre processing (reading)
    data_params.pre_processors_.sample_processors.extend(
        [
            DataProcessorFactoryParams(TextNormalizer.__name__, TARGETS_PROCESSOR, {'unicode_normalization': 'NFC'}),
            DataProcessorFactoryParams(TextRegularizer.__name__, TARGETS_PROCESSOR, {'replacements': default_text_regularizer_replacements(['extended'])}),
            DataProcessorFactoryParams(StripTextProcessor.__name__, TARGETS_PROCESSOR)
        ])
    if bidi_dir != "":
        data_params.pre_processors_.sample_processors.append(
            DataProcessorFactoryParams(BidiTextProcessor.__name__, TARGETS_PROCESSOR, {'bidi_direction': bidi_dir})
        )

    data_params.line_height_ = 48
    text_preproc_bidi = Data.data_processor_factory().create_sequence(data_params.pre_processors_.sample_processors, data_params,
                                                                      mode=PipelineMode.Targets)
    img_preproc = Data.data_processor_factory().create_sequence(data_params.pre_processors_.sample_processors, data_params,
                                                                mode=PipelineMode.Prediction)

    def text_preproc(text):
        return text_preproc_bidi.apply_on_sample(
            InputSample(None, text, None).to_input_target_sample()).targets

    global image_preproc

    def image_preproc(image):
        return img_preproc.apply_on_sample(
            InputSample(image, None, None).to_input_target_sample()).inputs

    return text_preproc, image_preproc


class ImgProc(object):
    def __init__(self, book, session, img_preproc, rect, rrect):
        self.book = book
        self.s = session
        self.dataproc = img_preproc
        self.rect = rect
        self.rrect = rrect

    def __call__(self, item):
        # b = self.book
        pno, url, img_w, lines = item
        res = []

        imgresp = self.s.get(url, params={"upgrade": "nrm"})
        f = BytesIO(imgresp.content)
        pageimg = cv.imdecode(np.frombuffer(f.read(), np.uint8), flags=cv.IMREAD_GRAYSCALE)
        f.close()

        for lid, coords in lines:
            limg = cutout(pageimg, coords,
                          scale=pageimg.shape[1] / img_w,
                          rect=self.rect, rrect=self.rrect)
            try:
                limg = self.dataproc(limg)
            except ValueError as err:
                raise ValueError("Error on page {}, line {}: {}".format(pno, lid, err))
            if len(limg.shape) != 2:
                continue
            res.append((lid, limg))
        return pno, res


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
        if not path.isfile(cachefile):
            cache = h5py.File(cachefile, "w", libver='latest', swmr=True)
            cache.close()

        self.login(login, password)

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

    def update_books(self, books, gt_layer=0, rect=False, rrect=False, rtl=False):
        """ Update books cache
        Parameters
        ----------
        books : book title or list of titles or "title/pagenumber" for single specific page
        gt_layer : index of ground truth in PAGE files
        rect : cut out rectangles instead of line polygons
        rtl : set text direction to rtl
        """

        if isinstance(books, str):
            books = [books]
        cache = h5py.File(self.cachefile, 'a', libver='latest')

        bidi_dir = 'R' if rtl else ''

        text_preproc, image_preproc = get_preprocs(bidi_dir=bidi_dir)

        pool = Pool()

        for b in books:
            singlepage = None
            if "/" in b:
                b, singlepage = b.split("/")
            if b.endswith("_ar") and not rtl:
                print("Warning: Title ends with _ar but rtl is not set!")
            if b.endswith("_ar") and not (rect or rrect):
                print("Warning: Title ends with _ar but neither rect nor rrect is set!")
            if singlepage is None:
                bookiter, booklen = self.genbook(b)
            else:
                book = self.getbook(b)
                bookiter = self.getbook(b).items()
                booklen = 1
                if singlepage not in book:
                    raise ValueError(f'Page "{singlepage}" not found!')
            pagelist = []

            if b not in cache:
                cache.create_group(b)

            cache[b].attrs["dir"] = "rtl" if rtl else "ltr"

            def bookconv(book):
                for pno, root in book:
                    if singlepage is not None and pno != singlepage:
                        continue
                    if pno not in cache[b]:
                        cache.create_group(b+"/"+pno)
                    ns = f'{{{root.nsmap[None]}}}'
                    xmlPage = root.find(f'./{ns}Page')

                    imglist = []

                    linelist = []
                    img_w = int(xmlPage.attrib.get("imageWidth"))
                    cache[b][pno].attrs["img_w"] = img_w
                    image_file = xmlPage.attrib.get("imageFilename")
                    cache[b][pno].attrs["image_file"] = image_file

                    for tl in root.iterfind(f'.//{ns}TextLine'):
                        coords = tl.find(f'./{ns}Coords').attrib.get("points")
                        lid = tl.attrib.get("id")
                        linelist.append(lid)
                        newl = False
                        if lid not in cache[b][pno]:
                            newl = True
                            cache[b][pno].create_dataset(lid, (0, 48), maxshape=(None, 48),
                                                         dtype="uint8", chunks=(256, 48))
                        if newl or cache[b][pno][lid].attrs.get("coords") != coords:
                            cache[b][pno][lid].attrs["coords"] = coords
                            imglist.append((lid, coords))

                        comments = tl.attrib.get("comments")
                        if comments is not None and comments.strip():
                            cache[b][pno][lid].attrs["comments"] = comments.strip()
                        cache[b][pno][lid].attrs["rtype"] = tl.getparent().attrib["type"]

                        rawtext = tl.findtext(f'./{ns}TextEquiv[@index="{gt_layer}"]/{ns}Unicode', default="")

                        if newl or rawtext != cache[b][pno][lid].attrs.get("text_raw"):
                            cache[b][pno][lid].attrs["text_raw"] = rawtext
                            cache[b][pno][lid].attrs["text"] = text_preproc(rawtext)

                    for lid in cache[b][pno]:
                        if lid not in linelist:
                            _ = cache[b][pno].pop(lid)

                    url = f'{self.baseurl}/books/{b}/{image_file}'

                    yield pno, url, img_w, imglist

            r = pool.imap_unordered(ImgProc(b, self.session, image_preproc, rect, rrect),
                                    bookconv(bookiter))

            for res in tqdm(r, total=booklen, desc=b.ljust(max(len(x) for x in books))):
                pno, lines = res
                pagelist.append(pno)
                for lid, limg in lines:
                    limg = np.ascontiguousarray(limg, dtype=np.uint8)
                    if cache[b][pno][lid].shape != limg.shape:
                        cache[b][pno][lid].resize(limg.shape)
                    cache[b][pno][lid].write_direct(limg)

            # remove pages not contained in the nashi book
            for pno in cache[b]:
                if pno not in pagelist:
                    _ = cache[b].pop(pno)

            cache.flush()

        pool.close()
        pool.join()
        cache.flush()
        cache.close()

    def train_books(self, books,
                    cachefile=None,
                    name="model",
                    skip_commented=True,
                    validation_split_ratio=1,
                    bidi="",
                    n_augmentations=0,
                    ema_weights=False,
                    train_verbose=1,
                    debug=False,
                    epochs=100,
                    whitelist="",
                    keep_loaded_codec=False,
                    preload=True,
                    weights=None,
                    ensemble=-1):

        if isinstance(books, str):
            books = [books]
        if cachefile is None:
            cachefile = self.cachefile

        dataset_args = FileDataReaderArgs(
            line_generator_params=None,
            text_generator_params=None,
            pad=None,
            text_index=0,
        )

        params: TrainerParams = Scenario.default_trainer_params()

        # =================================================================================================================
        # Data Params
        # resolve lines
        lids = list(lids_from_books(books, self.cachefile, complete_only=True, skip_commented=skip_commented))
        data_params: DataParams = params.scenario_params.data_params

        if 0 < validation_split_ratio < 1:
            valsamples = random.sample(lids, int((1-validation_split_ratio)*len(lids)))
            for s in valsamples:
                lids.remove(s)
            # validation
            data_params.val = PipelineParams(
                 type="Nsh5",
                 files=[cachefile],
                 text_files=valsamples,
                 skip_invalid=False,
                 gt_extension=None,
                 data_reader_args=dataset_args,
                 batch_size=1,
                 num_processes=1,
             )
            params.use_training_as_validation = False
        else:
            params.use_training_as_validation = True
            data_params.val = None

        data_params.train = PipelineParams(
            type="Nsh5",
            skip_invalid=False,
            remove_invalid=False,
            files=[cachefile],
            text_files=lids,
            gt_extension=None,
            data_reader_args=dataset_args,
            batch_size=1,
            num_processes=1,
        )

        data_params.pre_processors_ = SamplePipelineParams(run_parallel=True)
        data_params.post_processors_.run_parallel = SamplePipelineParams(
            run_parallel=False, sample_processors=[
                DataProcessorFactoryParams(ReshapeOutputsProcessor.__name__),
                DataProcessorFactoryParams(CTCDecoderProcessor.__name__),
            ])

        data_params.pre_processors_.sample_processors.append(DataProcessorFactoryParams("FinalPreparation", INPUT_PROCESSOR, {
                'normalize': True,
                'invert': False,
                'transpose': True,
                'pad': 16,
                'pad_value': False}))

        # Text post processing (prediction)
        data_params.post_processors_.sample_processors.extend(
            [
                DataProcessorFactoryParams(TextNormalizer.__name__, TARGETS_PROCESSOR,
                                           {'unicode_normalization': "NFC"}),
                DataProcessorFactoryParams(TextRegularizer.__name__, TARGETS_PROCESSOR,
                                           {'replacements': default_text_regularizer_replacements(["extended"])}),
                DataProcessorFactoryParams(StripTextProcessor.__name__, TARGETS_PROCESSOR)
            ])
        if bidi:
            data_params.post_processors_.sample_processors.append(
                DataProcessorFactoryParams(BidiTextProcessor.__name__, TARGETS_PROCESSOR, {'bidi_direction': "R"})
            )

        data_params.pre_processors_.sample_processors.extend([
            DataProcessorFactoryParams(AugmentationProcessor.__name__, {PipelineMode.Training}, {'augmenter_type': 'simple'}),
            DataProcessorFactoryParams(PrepareSampleProcessor.__name__, INPUT_PROCESSOR),
        ])

        data_params.data_aug_params = DataAugmentationAmount.from_factor(n_augmentations)
        data_params.line_height_ = 48

        # =================================================================================================================
        # Trainer Params
        params.calc_ema = ema_weights
        params.verbose = train_verbose
        params.force_eager = debug
        params.skip_model_load_test = not debug
        params.scenario_params.debug_graph_construction = debug
        params.epochs = epochs
        # params.samples_per_epoch = int(args.samples_per_epoch) if args.samples_per_epoch >= 1 else -1
        params.samples_per_epoch = -1
        # params.scale_epoch_size = abs(args.samples_per_epoch) if args.samples_per_epoch < 1 else 1
        params.scale_epoch_size = 1
        params.skip_load_model_test = True
        params.scenario_params.export_frozen = False
        # params.checkpoint_save_freq_ = args.checkpoint_frequency if args.checkpoint_frequency >= 0 else args.early_stopping_frequency
        params.checkpoint_save_freq_ = 1
        params.checkpoint_dir = "checkpoints"
        params.test_every_n = 1
        params.skip_invalid_gt = False
        params.data_aug_retrain_on_original = True

        # if args.seed > 0:
        #    params.random_seed = args.seed

        params.optimizer_params.clip_grad = 5
        params.codec_whitelist = whitelist
        params.keep_loaded_codec = keep_loaded_codec
        params.preload_training = preload
        params.preload_validation = preload
        params.warmstart_params.model = weights

        params.auto_compute_codec = True
        params.progress_bar = True

        params.early_stopping_params.frequency = 1
        params.early_stopping_params.upper_threshold = 0.9
        params.early_stopping_params.lower_threshold = 1.0 - 1.0
        params.early_stopping_params.n_to_go = 5
        params.early_stopping_params.best_model_name = ''
        params.early_stopping_params.best_model_output_dir = "checkpoints"
        params.scenario_params.default_serve_dir_ = f'best_{name}.ckpt.h5'
        params.scenario_params.trainer_params_filename_ = f'best_{name}.ckpt.json'

        # =================================================================================================================
        # Model params
        params_from_definition_string("cnn=40:3x3,pool=2x2,cnn=60:3x3,pool=2x2,lstm=200,dropout=0.5", params)
        params.scenario_params.model_params.ensemble = ensemble
        params.scenario_params.model_params.no_masking_out_during_training = False

        scenario = Scenario(params.scenario_params)
        trainer = scenario.create_trainer(params)
        trainer.train()

    def predict_books(self, books, checkpoint, cachefile=None, pageupload=True, text_index=1, pred_all=False):
        if not pageupload:
            print("""Warning: trying to save results to the hdf5-Cache may fail due to some issue
                  with file access from multiple threads. It should work, however, if you set
                  export HDF5_USE_FILE_LOCKING='FALSE'.""")
        if type(books) == str:
            books = [books]
        if type(checkpoint) == str:
            checkpoint = [checkpoint]
        checkpoint = [(cp if cp.endswith(".json") else cp + ".json") for cp in checkpoint]
        checkpoint = glob_all(checkpoint)
        checkpoint = [cp[:-5] for cp in checkpoint]
        if cachefile is None:
            cachefile = self.cachefile

        def create_ctc_decoder_params():
            params = CTCDecoderParams()
            params.beam_width = 25
            params.word_separator = ' '
            # args.dictionary = None
            return params

        verbose = False

        # add json as extension, resolve wildcard, expand user, ... and remove .json again
        checkpoint = [(cp if cp.endswith(".json") else cp + ".json") for cp in checkpoint]
        checkpoint = glob_all(checkpoint)
        checkpoint = [cp[:-5] for cp in checkpoint]
        # extension = None

        # create ctc decoder
        # ctc_decoder_params = create_ctc_decoder_params()

        # create voter
        voter_params = VoterParams()
        voter_params.type = VoterType("confidence_voter_default_ctc")

        # skip invalid files and remove them, there wont be predictions of invalid files
        lids = list(lids_from_books(books, cachefile,
                    complete_only=False, skip_commented=False, new_only=not pred_all))
        predict_params = PipelineParams(
            type="Nsh5",
            skip_invalid=False,
            remove_invalid=False,
            files=glob_all([cachefile]),
            text_files=lids,
            data_reader_args=FileDataReaderArgs(
                pad=None,
                text_index=1,
            ),
            batch_size=1,
            num_processes=1,
        )

        # predict for all models
        # TODO: Use CTC Decoder params
        from calamari_ocr.ocr.predict.predictor import MultiPredictor
        predictor = MultiPredictor.from_paths(checkpoints=checkpoint, voter_params=voter_params,
                                              predictor_params=PredictorParams(silent=True, progress_bar=True))

        preprocs = [p for p in predictor.data.params().pre_processors_.sample_processors
                    if PipelineMode.Prediction in p.modes and not p.name == 'PrepareSampleProcessor']
        for p in preprocs:
            if p.name == "FinalPreparation":
                p.args = {
                        'normalize': True,
                        'invert': False,
                        'transpose': True,
                        'pad': 16,
                        'pad_value': False}
            else:
                p.modes.remove(PipelineMode.Prediction)

        do_prediction = predictor.predict(predict_params)
        pipeline = predictor.data.get_predict_data(predict_params)
        reader = pipeline.reader()
        if len(reader) == 0:
            raise Exception("Empty dataset provided. Check your files argument!")

        avg_sentence_confidence = 0
        n_predictions = 0

        reader.prepare_store()

        samples = []
        sentences = []
        # output the voted results to the appropriate files
        for s in do_prediction:
            _, (_, prediction), meta = s.inputs, s.outputs, s.meta
            sample = reader.sample_by_id(meta['id'])
            n_predictions += 1
            sentence = prediction.sentence

            avg_sentence_confidence += prediction.avg_char_probability
            if verbose:
                lr = "\u202A\u202B"
                logger.info("{}: '{}{}{}'".format(meta['id'], lr[get_base_level(sentence)], sentence, "\u202C"))

            samples.append(sample)
            sentences.append(sentence)
            reader.store_text(sentence, sample, output_dir=None, extension=None)

        logger.info("Average sentence confidence: {:.2%}".format(avg_sentence_confidence / n_predictions))

        # reader.store(args.extension)

        if pageupload:
            ocrdata = {}
            for lname, text in reader.predictions.items():
                _, b, p, ln = lname.split("/")
                if b not in ocrdata:
                    ocrdata[b] = {}
                if p not in ocrdata[b]:
                    ocrdata[b][p] = {}
                ocrdata[b][p][ln] = text

            data = {"ocrdata": ocrdata, "index": text_index}
            self.session.post(self.baseurl+"/_ocrdata",
                              data=gzip.compress(json.dumps(data).encode("utf-8")),
                              headers={"Content-Type": "application/json;charset=UTF-8",
                                       "Content-Encoding": "gzip"})
            logger.info("Results uploaded")
        else:
            reader.store()
            logger.info("All prediction files written")

    def evaluate_books(self, books, checkpoint, cachefile=None, output_individual_voters=False, n_confusions=10):
        if type(books) == str:
            books = [books]
        if type(checkpoint) == str:
            checkpoint = [checkpoint]
        checkpoint = [(cp if cp.endswith(".json") else cp + ".json") for cp in checkpoint]
        checkpoint = glob_all(checkpoint)
        checkpoint = [cp[:-5] for cp in checkpoint]
        if cachefile is None:
            cachefile = self.cachefile
        lids = list(lids_from_books(books, cachefile,
                    complete_only=True, skip_commented=True))

        pipeline_params = PipelineParams(
            type="Nsh5",
            skip_invalid=False,
            remove_invalid=False,
            files=[cachefile],
            gt_extension=None,
            text_files=lids,
            data_reader_args=FileDataReaderArgs(
                pad=None,
                text_index=1,
            ),
            batch_size=1,
            num_processes=1,
        )

        from calamari_ocr.ocr.predict.predictor import MultiPredictor
        voter_params = VoterParams()
        predictor = MultiPredictor.from_paths(checkpoints=checkpoint, voter_params=voter_params,
                                              predictor_params=PredictorParams(silent=True, progress_bar=True))

        preprocs = [p for p in predictor.data.params().pre_processors_.sample_processors
                    if PipelineMode.Prediction in p.modes and not p.name == 'PrepareSampleProcessor']
        for p in preprocs:
            if p.name == "FinalPreparation":
                p.args = {
                        'normalize': True,
                        'invert': False,
                        'transpose': True,
                        'pad': 16,
                        'pad_value': False}
            else:
                p.modes.remove(PipelineMode.Prediction)

        do_prediction = predictor.predict(pipeline_params)

        all_voter_sentences = []
        all_prediction_sentences = {}

        for s in do_prediction:
            _, (_, prediction), _ = s.inputs, s.outputs, s.meta
            sentence = prediction.sentence
            if prediction.voter_predictions is not None and output_individual_voters:
                for i, p in enumerate(prediction.voter_predictions):
                    if i not in all_prediction_sentences:
                        all_prediction_sentences[i] = []
                    all_prediction_sentences[i].append(p.sentence)
            all_voter_sentences.append(sentence)

        # evaluation
        from calamari_ocr.ocr.evaluator import Evaluator
        evaluator = Evaluator(predictor.data)
        evaluator.preload_gt(gt_dataset=pipeline_params, progress_bar=True)

        def single_evaluation(label, predicted_sentences):
            if len(predicted_sentences) != len(evaluator.preloaded_gt):
                raise Exception("Mismatch in number of gt and pred files: {} != {}. Probably, the prediction did "
                                "not succeed".format(len(evaluator.preloaded_gt), len(predicted_sentences)))

            r = evaluator.evaluate(gt_data=evaluator.preloaded_gt, pred_data=predicted_sentences,
                                   progress_bar=True, processes=1)

            print("=================")
            print(f"Evaluation result of {label}")
            print("=================")
            print("")
            print("Got mean normalized label error rate of {:.2%} ({} errs, {} total chars, {} sync errs)".format(
                r["avg_ler"], r["total_char_errs"], r["total_chars"], r["total_sync_errs"]))
            print()
            print()

            # sort descending
            print_confusions(r, n_confusions)

            return r

        full_evaluation = {}
        for id, data in [(str(i), sent) for i, sent in all_prediction_sentences.items()] + [('voted', all_voter_sentences)]:
            full_evaluation[id] = {"eval": single_evaluation(id, data), "data": data}
        return full_evaluation

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
        savelines = [cache[b][p][ln] for b in books for p in cache[b] for ln in cache[b][p]
                     if cache[b][p][ln].attrs.get("pred") is not None]
        for line in savelines:
            _, b, p, ln = line.name.split("/")
            if b not in ocrdata:
                ocrdata[b] = {}
            if p not in ocrdata[b]:
                ocrdata[b][p] = {}
            ocrdata[b][p][ln] = line.attrs.get("pred")

        data = {"ocrdata": ocrdata, "index": text_index}
        self.session.post(self.baseurl+"/_ocrdata",
                          data=gzip.compress(json.dumps(data).encode("utf-8")),
                          headers={"Content-Type": "application/json;charset=UTF-8",
                                   "Content-Encoding": "gzip"})
        cache.close()

    def getbook(self, bookname, as_string=False):
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
                if not as_string:
                    book[pagename] = etree.parse(fo).getroot()
                else:
                    book[pagename] = fo.read()
        f.close()
        return book

    def genbook(self, bookname):
        """ Download a book from the nashi server
        Parameters
        ----------
        bookname : title of the book to load

        Returns
        ----------
        (pagename, etree-root), number of pages
        """
        pagezip = self.session.get(self.baseurl+"/books/{}_PageXML.zip"
                                   .format(bookname))
        f = BytesIO(pagezip.content)
        zf = zipfile.ZipFile(f)
        # book = {}
        namelist = zf.namelist()
        booklen = len(namelist)

        def zf_to_etree(fn):
            with zf.open(fn) as fo:
                et = etree.parse(fo)
            return et.getroot()

        return (((path.splitext(path.split(fn)[1])[0]), zf_to_etree(fn))
                for fn in namelist), booklen
