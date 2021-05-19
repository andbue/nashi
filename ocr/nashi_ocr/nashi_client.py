"""
A simple client to OCR texts transcribed in nashi
"""
from tensorflow import keras
from tensorflow.config import list_physical_devices

import cv2 as cv
import gzip
import h5py
import json
import numpy as np
import requests
import zipfile
from bidi.algorithm import get_base_level
from getpass import getpass
from io import BytesIO
from lxml import etree, html
from multiprocessing import Pool
from os import path
from tqdm import tqdm
from typing import List
from copy import deepcopy

from dataclasses import dataclass, field

from paiargparse import pai_dataclass, pai_meta
from calamari_ocr.ocr.dataset.datareader.base import (
    CalamariDataGenerator,
    CalamariDataGeneratorParams,
    InputSample,
    SampleMeta,
)

from calamari_ocr.ocr.dataset.data import Data
from calamari_ocr.ocr.dataset.textprocessors.basic_text_processors import (
    BidiTextProcessorParams,
)
from calamari_ocr.ocr.dataset.textprocessors.basic_text_processors import BidiDirection
from calamari_ocr.ocr.dataset.imageprocessors.final_preparation import (
    FinalPreparationProcessorParams,
)
from calamari_ocr.ocr.dataset.imageprocessors.preparesample import (
    PrepareSampleProcessorParams,
)
from calamari_ocr.ocr.dataset.imageprocessors.augmentation import (
    AugmentationProcessorParams,
)
from calamari_ocr.ocr.scenario import CalamariScenario
from calamari_ocr.ocr.training.pipeline_params import (
    CalamariSplitTrainerPipelineParams,
    CalamariTrainOnlyPipelineParams,
)
from calamari_ocr.ocr.predict.predictor import MultiPredictor
from calamari_ocr.ocr.predict.params import PredictorParams
from calamari_ocr.ocr.voting import VoterParams
from calamari_ocr.utils import glob_all
from calamari_ocr.scripts.eval import print_confusions

from calamari_ocr.ocr.training.cross_fold_trainer import (
    CrossFoldTrainer,
    CrossFoldTrainerParams,
)

from calamari_ocr.ocr.dataset.datareader.pagexml.reader import PageXMLReader, CutMode
from tfaip.util.tfaipargparse import post_init
from tfaip.data.databaseparams import DataPipelineParams
from tfaip.data.pipeline.definitions import PipelineMode

import tfaip.util.logging

logger = tfaip.util.logging.logger(__name__)


def lids_from_books(
    books, cachefile, new_only=False, complete_only=False, skip_commented=False
):
    with h5py.File(cachefile, "r", libver="latest", swmr=True) as cache:
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


@pai_dataclass
@dataclass
class Nsh5(CalamariDataGeneratorParams):
    cachefile: str = field(
        default="nashi_cache.h5", metadata=pai_meta(help="Path to the h5py file")
    )
    lines: List[str] = field(
        default_factory=list,
        metadata=pai_meta(help="Iterable of sample ids in h5py file"),
    )

    def __len__(self):
        return len(self.lines)

    def select(self, indices: List[int]):
        self.lines = [self.lines[i] for i in indices]

    def to_prediction(self):
        return self

    @staticmethod
    def cls():
        return Nsh5DataReader


class Nsh5DataReader(CalamariDataGenerator[Nsh5]):
    def __init__(self, mode: PipelineMode, params: Nsh5):
        super().__init__(mode, params)
        self.predictions = {} if self.mode == PipelineMode.PREDICTION else None
        self.cachefile = params.cachefile
        for sid in params.lines:
            self.add_sample({"id": sid})

    def store_text(self, sentence, sample, output_dir, extension):
        self.predictions[sample["id"]] = sentence

    def store(self):
        self.cacheclose()
        with h5py.File(self.cachefile, "r+", libver="latest") as cache:
            cache.swmr_mode = True
            for p in self.predictions:
                cache[p].attrs["pred"] = self.predictions[p]

    def cacheopen(self):
        if isinstance(self.cachefile, str):
            self.cachefile = h5py.File(self.cachefile, "r", libver="latest", swmr=True)
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
            yield InputSample(
                None,
                text,
                SampleMeta(sample["id"], fold_id=sample["fold_id"]),
            )
        else:
            im = np.empty(s.shape, s.dtype)
            s.read_direct(im)
            yield InputSample(
                im.T,
                text,
                SampleMeta(sample["id"], fold_id=sample["fold_id"]),
            )

    def __del__(self):
        self.cacheclose()


def cutout(pageimg, coordstring, scale=1, rect=False, rrect=False):
    """Cut region from image
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
    return PageXMLReader.cutout(
        pageimg, coordstring, mode=mode, angle=0, cval=None, scale=scale
    )


def get_preproc_text(rtl=False):
    data_params = Data.default_params()
    data_params.skip_invalid_gt = False
    data_params.pre_proc.run_parallel = False

    if rtl:
        for p in data_params.pre_proc.processors_of_type(BidiTextProcessorParams):
            p.bidi_direction = BidiDirection.RTL
    post_init(data_params)

    pl = Data(data_params).create_pipeline(DataPipelineParams, None)
    pl.mode = PipelineMode.TARGETS
    preproc = data_params.pre_proc.create(pl)

    def pp(text):
        its = InputSample(
            None, text, SampleMeta("001", fold_id="01")
        ).to_input_target_sample()
        s = preproc.apply_on_sample(its)
        return s.targets

    return pp


def get_preproc_image():
    data_params = Data.default_params()
    data_params.skip_invalid_gt = False
    data_params.pre_proc.run_parallel = False
    data_params.pre_proc.processors = data_params.pre_proc.processors[:-1]
    for p in data_params.pre_proc.processors_of_type(FinalPreparationProcessorParams):
        p.pad = 0
    post_init(data_params)
    pl = Data(data_params).create_pipeline(DataPipelineParams, None)
    pl.mode = PipelineMode.PREDICTION
    preproc = data_params.pre_proc.create(pl)

    def pp(image):
        its = InputSample(
            image, None, SampleMeta("001", fold_id="01")
        ).to_input_target_sample()
        s = preproc.apply_on_sample(its)
        return s.inputs

    return pp


class ImgProc(object):
    def __init__(self, session, rect, rrect):
        self.s = session
        self.dataproc = None
        self.rect = rect
        self.rrect = rrect

    def get_dataproc(self):
        if self.dataproc is None:
            self.dataproc = get_preproc_image()
        return self.dataproc

    def __call__(self, item):
        dataproc = self.get_dataproc()
        # b = self.book
        pno, url, img_w, lines = item
        res = []

        imgresp = self.s.get(url, params={"upgrade": "nrm"})
        f = BytesIO(imgresp.content)
        pageimg = cv.imdecode(
            np.frombuffer(f.read(), np.uint8), flags=cv.IMREAD_GRAYSCALE
        )
        f.close()

        for lid, coords in lines:
            limg = cutout(
                pageimg,
                coords,
                scale=pageimg.shape[1] / img_w,
                rect=self.rect,
                rrect=self.rrect,
            )
            try:
                limg = dataproc(limg)
            except ValueError as err:
                raise ValueError("Error on page {}, line {}: {}".format(pno, lid, err))
            if len(limg.shape) != 2:
                continue
            res.append((lid, limg))
        return pno, res


class NashiClient:
    def __init__(
        self, cachefile="nashi_cache.h5", baseurl="", login=None, password=None
    ):
        """Create a nashi client
        Parameters
        ----------
        cachefile : filename of hdf5-cache
        baseurl : web address of nashi instance
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
            cache = h5py.File(cachefile, "w", libver="latest", swmr=True)
            cache.close()
        self.creds = (login, password)

    def get_session(self):
        if self.session is None:
            self.login(*self.creds)
        return self.session

    def login(self, email, pw):
        if email is None:
            raise Exception("Login information is needed to access nashi server!")
        if pw is None:
            pw = getpass("Password: ")
        s = requests.Session()
        r = s.get(self.baseurl + "/login")
        res = html.document_fromstring(r.text)
        csrf = res.get_element_by_id("csrf_token").attrib["value"]
        lg = s.post(
            self.baseurl + "/login",
            data={
                "csrf_token": csrf,
                "email": email,
                "password": pw,
                "submit": "Login",
            },
        )
        if "<li>Invalid password</li>" in lg.text:
            raise Exception("Login failed.")
        self.session = s

    def update_books(self, books, gt_layer=0, rect=False, rrect=False, rtl=False):
        """Update books cache
        Parameters
        ----------
        books : book title or list of titles or "title/pagenumber" for single specific page
        gt_layer : index of ground truth in PAGE files
        rect : cut out rectangles instead of line polygons
        rtl : set text direction to rtl
        """

        if isinstance(books, str):
            books = [books]
        cache = h5py.File(self.cachefile, "a", libver="latest")

        text_preproc = get_preproc_text(rtl)

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
                        cache.create_group(b + "/" + pno)
                    ns = f"{{{root.nsmap[None]}}}"
                    xmlPage = root.find(f"./{ns}Page")

                    imglist = []

                    linelist = []
                    img_w = int(xmlPage.attrib.get("imageWidth"))
                    cache[b][pno].attrs["img_w"] = img_w
                    image_file = xmlPage.attrib.get("imageFilename")
                    cache[b][pno].attrs["image_file"] = image_file

                    for tl in root.iterfind(f".//{ns}TextLine"):
                        coords = tl.find(f"./{ns}Coords").attrib.get("points")
                        lid = tl.attrib.get("id")
                        linelist.append(lid)
                        newl = False
                        if lid not in cache[b][pno]:
                            newl = True
                            cache[b][pno].create_dataset(
                                lid,
                                (0, 48),
                                maxshape=(None, 48),
                                dtype="uint8",
                                chunks=(256, 48),
                            )
                        if newl or cache[b][pno][lid].attrs.get("coords") != coords:
                            cache[b][pno][lid].attrs["coords"] = coords
                            imglist.append((lid, coords))

                        comments = tl.attrib.get("comments")
                        if comments is not None and comments.strip():
                            cache[b][pno][lid].attrs["comments"] = comments.strip()
                        cache[b][pno][lid].attrs["rtype"] = tl.getparent().attrib[
                            "type"
                        ]

                        rawtext = tl.findtext(
                            f'./{ns}TextEquiv[@index="{gt_layer}"]/{ns}Unicode',
                            default="",
                        )

                        if newl or rawtext != cache[b][pno][lid].attrs.get("text_raw"):
                            cache[b][pno][lid].attrs["text_raw"] = rawtext
                            cache[b][pno][lid].attrs["text"] = text_preproc(rawtext)

                    for lid in cache[b][pno]:
                        if lid not in linelist:
                            _ = cache[b][pno].pop(lid)

                    url = f"{self.baseurl}/books/{b}/{image_file}"

                    yield pno, url, img_w, imglist

            r = pool.imap_unordered(
                ImgProc(self.get_session(), rect, rrect), bookconv(bookiter)
            )

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

    def train_books(
        self,
        books,
        cachefile=None,
        name="model",
        skip_commented=True,
        validation_split_ratio=1,
        bidi="",
        n_augmentations=0,
        ema_decay=0.0,
        train_verbose=1,
        debug=False,
        epochs=100,
        whitelist="",
        keep_loaded_codec=False,
        preload=True,
        weights=None,
        ensemble=0,
    ):

        keras.backend.clear_session()

        if isinstance(books, str):
            books = [books]
        if cachefile is None:
            cachefile = self.cachefile

        p = CalamariScenario.default_trainer_params()
        lids = list(
            lids_from_books(
                books, cachefile, complete_only=True, skip_commented=skip_commented
            )
        )
        train = Nsh5(cachefile=cachefile, lines=lids)

        newprcs = []
        for prc in p.scenario.data.pre_proc.processors:
            prc = deepcopy(prc)
            if PipelineMode.TRAINING in prc.modes:
                if isinstance(prc, FinalPreparationProcessorParams):
                    prc.normalize, prc.invert, prc.transpose = False, False, True
                elif isinstance(prc, AugmentationProcessorParams):
                    prc.n_augmentations = n_augmentations
                elif not isinstance(prc, PrepareSampleProcessorParams):
                    prc.modes.discard(PipelineMode.TRAINING)
            elif isinstance(prc, PrepareSampleProcessorParams):
                prc.modes.add(PipelineMode.TRAINING)
            newprcs.append(prc)
        p.scenario.data.pre_proc.processors = newprcs

        p.device.gpus = [n for n, _ in enumerate(list_physical_devices("GPU"))]

        if validation_split_ratio < 1:
            p.gen = CalamariSplitTrainerPipelineParams(
                validation_split_ratio=validation_split_ratio, train=train
            )
        else:
            p.gen = CalamariTrainOnlyPipelineParams(train=train)

        if bidi:
            for prc in p.scenario.data.post_proc.processors_of_type(
                BidiTextProcessorParams
            ):
                prc.bidi_direction = BidiDirection.RTL

        p.epochs = epochs
        p.codec.keep_loaded = keep_loaded_codec
        p.gen.train.preload = preload
        p.warmstart.model = weights
        p.scenario.model.ensemble = ensemble
        p.ema_decay = ema_decay

        p.scenario.data.__post_init__()
        p.scenario.__post_init__()
        p.__post_init__()

        p.output_dir = name
        trainer = p.scenario.cls().create_trainer(p)
        return trainer.train()

    def cftrain_books(
        self,
        books,
        n_folds=5,
        cachefile=None,
        name="models",
        tempdir=None,
        keep_temporary_files=False,
        max_parallel_models=-1,
        skip_commented=True,
        validation_split_ratio=1,
        bidi="",
        n_augmentations=0,
        ema_decay=0.0,
        train_verbose=1,
        debug=False,
        epochs=100,
        whitelist="",
        keep_loaded_codec=False,
        preload=True,
        weights=[],
        ensemble=0,
    ):

        keras.backend.clear_session()
        if isinstance(weights, str):
            weights = [weights]
        if isinstance(books, str):
            books = [books]
        if cachefile is None:
            cachefile = self.cachefile
        if max_parallel_models < 1:
            max_parallel_models = n_folds
        lids = list(
            lids_from_books(
                books, cachefile, complete_only=True, skip_commented=skip_commented
            )
        )
        train = Nsh5(cachefile=cachefile, lines=lids)

        cfparams = CrossFoldTrainerParams()
        cfparams.weights = weights
        cfparams.temporary_dir = tempdir
        cfparams.best_models_dir = name
        cfparams.n_folds = n_folds
        cfparams.keep_temporary_files = False
        cfparams.max_parallel_models = max_parallel_models

        newprcs = []
        for prc in cfparams.trainer.scenario.data.pre_proc.processors:
            prc = deepcopy(prc)
            if PipelineMode.TRAINING in prc.modes:
                if isinstance(prc, FinalPreparationProcessorParams):
                    prc.normalize, prc.invert, prc.transpose = False, False, True
                elif isinstance(prc, AugmentationProcessorParams):
                    prc.n_augmentations = n_augmentations
                elif not isinstance(prc, PrepareSampleProcessorParams):
                    prc.modes = set()
            elif isinstance(prc, PrepareSampleProcessorParams):
                prc.modes.add(PipelineMode.TRAINING)
            newprcs.append(prc)
        cfparams.trainer.scenario.data.pre_proc.processors = newprcs

        cfparams.trainer.device.gpus = [
            n for n, _ in enumerate(list_physical_devices("GPU"))
        ]

        cfparams.trainer.gen.train = train

        if bidi:
            for prc in cfparams.trainer.scenario.data.post_proc.processors_of_type(
                BidiTextProcessorParams
            ):
                prc.bidi_direction = BidiDirection.RTL

        cfparams.trainer.epochs = epochs
        cfparams.trainer.codec.keep_loaded = keep_loaded_codec
        cfparams.trainer.gen.train.preload = preload
        cfparams.trainer.scenario.model.ensemble = ensemble
        cfparams.trainer.ema_decay = ema_decay

        cfparams.trainer.scenario.data.__post_init__()
        cfparams.trainer.scenario.__post_init__()
        cfparams.trainer.__post_init__()

        print(cfparams.to_json())
        with open("debug.json", "w") as f:
            f.write(cfparams.to_json())
        trainer = CrossFoldTrainer(cfparams)
        return trainer.run()

    def predict_books(
        self,
        books,
        checkpoint,
        cachefile=None,
        pageupload=True,
        text_index=1,
        pred_all=False,
    ):
        keras.backend.clear_session()
        if type(books) == str:
            books = [books]
        if type(checkpoint) == str:
            checkpoint = [checkpoint]
        checkpoint = [
            (cp if cp.endswith(".json") else cp + ".json") for cp in checkpoint
        ]
        checkpoint = glob_all(checkpoint)
        checkpoint = [cp[:-5] for cp in checkpoint]
        if cachefile is None:
            cachefile = self.cachefile
        verbose = False
        lids = list(
            lids_from_books(
                books,
                cachefile,
                complete_only=False,
                skip_commented=False,
                new_only=not pred_all,
            )
        )
        data = Nsh5(cachefile=cachefile, lines=lids)

        predparams = PredictorParams()
        predparams.device.gpus = [n for n, _ in enumerate(list_physical_devices("GPU"))]

        predictor = MultiPredictor.from_paths(
            checkpoints=checkpoint,
            voter_params=VoterParams(),
            predictor_params=predparams,
        )

        newprcs = []
        for prc in predictor.data.params.pre_proc.processors:
            prc = deepcopy(prc)
            if isinstance(prc, FinalPreparationProcessorParams):
                prc.normalize, prc.invert, prc.transpose = False, False, True
                newprcs.append(prc)
            elif isinstance(prc, PrepareSampleProcessorParams):
                newprcs.append(prc)
        predictor.data.params.pre_proc.processors = newprcs

        do_prediction = predictor.predict(data)
        pipeline = predictor.data.get_or_create_pipeline(
            predictor.params.pipeline, data
        )
        reader = pipeline.reader()
        if len(reader) == 0:
            raise Exception(
                "Empty dataset provided. Check your lines (got {})!".format(lids)
            )

        avg_sentence_confidence = 0
        n_predictions = 0

        reader.prepare_store()

        samples = []
        sentences = []
        # output the voted results to the appropriate files
        for s in do_prediction:
            _, (_, prediction), meta = s.inputs, s.outputs, s.meta
            sample = reader.sample_by_id(meta["id"])
            n_predictions += 1
            sentence = prediction.sentence

            avg_sentence_confidence += prediction.avg_char_probability
            if verbose:
                lr = "\u202A\u202B"
                logger.info(
                    "{}: '{}{}{}'".format(
                        meta["id"], lr[get_base_level(sentence)], sentence, "\u202C"
                    )
                )

            samples.append(sample)
            sentences.append(sentence)
            reader.store_text(sentence, sample, output_dir=None, extension=None)

        logger.info(
            "Average sentence confidence: {:.2%}".format(
                avg_sentence_confidence / n_predictions
            )
        )

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
            self.get_session().post(
                self.baseurl + "/_ocrdata",
                data=gzip.compress(json.dumps(data).encode("utf-8")),
                headers={
                    "Content-Type": "application/json;charset=UTF-8",
                    "Content-Encoding": "gzip",
                },
            )
            logger.info("Results uploaded")
        else:
            reader.store()
            logger.info("All prediction files written")

    def evaluate_books(
        self,
        books,
        checkpoint,
        cachefile=None,
        output_individual_voters=False,
        n_confusions=10,
        silent=True,
    ):
        keras.backend.clear_session()
        if type(books) == str:
            books = [books]
        if type(checkpoint) == str:
            checkpoint = [checkpoint]
        checkpoint = [
            (cp if cp.endswith(".json") else cp + ".json") for cp in checkpoint
        ]
        checkpoint = glob_all(checkpoint)
        checkpoint = [cp[:-5] for cp in checkpoint]
        if cachefile is None:
            cachefile = self.cachefile

        lids = list(
            lids_from_books(books, cachefile, complete_only=True, skip_commented=True)
        )
        data = Nsh5(cachefile=cachefile, lines=lids)

        predparams = PredictorParams()
        predparams.device.gpus = [n for n, _ in enumerate(list_physical_devices("GPU"))]
        predparams.silent = silent

        predictor = MultiPredictor.from_paths(
            checkpoints=checkpoint,
            voter_params=VoterParams(),
            predictor_params=predparams,
        )

        newprcs = []
        for prc in predictor.data.params.pre_proc.processors:
            prc = deepcopy(prc)
            if isinstance(prc, FinalPreparationProcessorParams):
                prc.normalize, prc.invert, prc.transpose = False, False, True
                newprcs.append(prc)
            elif isinstance(prc, PrepareSampleProcessorParams):
                newprcs.append(prc)
        predictor.data.params.pre_proc.processors = newprcs

        do_prediction = predictor.predict(data)

        all_voter_sentences = {}
        all_prediction_sentences = {}

        for s in do_prediction:
            _, (_, prediction), _ = s.inputs, s.outputs, s.meta
            sentence = prediction.sentence
            if prediction.voter_predictions is not None and output_individual_voters:
                for i, p in enumerate(prediction.voter_predictions):
                    if i not in all_prediction_sentences:
                        all_prediction_sentences[i] = {}
                    all_prediction_sentences[i][s.meta["id"]] = p.sentence
            all_voter_sentences[s.meta["id"]] = sentence

        # evaluation
        from calamari_ocr.ocr.evaluator import Evaluator, EvaluatorParams

        evaluator_params = EvaluatorParams(
            setup=predparams.pipeline,
            progress_bar=True,
            skip_empty_gt=True,
        )
        evaluator = Evaluator(evaluator_params, predictor.data)
        evaluator.preload_gt(gt_dataset=data, progress_bar=True)

        def single_evaluation(label, predicted_sentences):
            r = evaluator.evaluate(
                gt_data=evaluator.preloaded_gt, pred_data=predicted_sentences
            )

            print("=================")
            print(f"Evaluation result of {label}")
            print("=================")
            print("")
            print(
                "Got mean normalized label error rate of {:.2%} ({} errs, {} total chars, {} sync errs)".format(
                    r["avg_ler"],
                    r["total_char_errs"],
                    r["total_chars"],
                    r["total_sync_errs"],
                )
            )
            print()
            print()

            # sort descending
            print_confusions(r, n_confusions)

            return r

        full_evaluation = {}
        for id, data in [
            (str(i), sent) for i, sent in all_prediction_sentences.items()
        ] + [("voted", all_voter_sentences)]:
            full_evaluation[id] = {"eval": single_evaluation(id, data), "data": data}

        if not predparams.silent:
            print(full_evaluation)

        return full_evaluation

    def upload_books(self, books, text_index=1):
        """Upload books from the cachefile to the server
        Parameters
        ----------
        bookname : title of the book or list of titles
        text_index : index of the TextEquiv to write to

        Returns
        ----------
        dict mapping page names to lxml etree instances
        """
        cache = h5py.File(self.cachefile, "r", libver="latest", swmr=True)
        ocrdata = {}
        if type(books) == str:
            books = [books]
        savelines = [
            cache[b][p][ln]
            for b in books
            for p in cache[b]
            for ln in cache[b][p]
            if cache[b][p][ln].attrs.get("pred") is not None
        ]
        for line in savelines:
            _, b, p, ln = line.name.split("/")
            if b not in ocrdata:
                ocrdata[b] = {}
            if p not in ocrdata[b]:
                ocrdata[b][p] = {}
            ocrdata[b][p][ln] = line.attrs.get("pred")

        data = {"ocrdata": ocrdata, "index": text_index}
        self.get_session().post(
            self.baseurl + "/_ocrdata",
            data=gzip.compress(json.dumps(data).encode("utf-8")),
            headers={
                "Content-Type": "application/json;charset=UTF-8",
                "Content-Encoding": "gzip",
            },
        )
        cache.close()

    def getbook(self, bookname, as_string=False):
        """Download a book from the nashi server
        Parameters
        ----------
        bookname : title of the book to load

        Returns
        ----------
        dict mapping page names to lxml etree instances
        """
        pagezip = self.get_session().get(
            self.baseurl + "/books/{}_PageXML.zip".format(bookname)
        )
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
        """Download a book from the nashi server
        Parameters
        ----------
        bookname : title of the book to load

        Returns
        ----------
        (pagename, etree-root), number of pages
        """
        pagezip = self.get_session().get(
            self.baseurl + "/books/{}_PageXML.zip".format(bookname)
        )
        f = BytesIO(pagezip.content)
        zf = zipfile.ZipFile(f)
        namelist = zf.namelist()
        booklen = len(namelist)

        def zf_to_etree(fn):
            with zf.open(fn) as fo:
                et = etree.parse(fo)
            return et.getroot()

        return (
            ((path.splitext(path.split(fn)[1])[0]), zf_to_etree(fn)) for fn in namelist
        ), booklen
