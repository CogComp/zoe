import os
import pickle
import sys

from zoe_utils import DataReader
from zoe_utils import ElmoProcessor
from zoe_utils import EsaProcessor
from zoe_utils import Evaluator
from zoe_utils import InferenceProcessor


class ZoeRunner:

    """
    @allow_tensorflow sets whether the system will do run-time ELMo processing.
                      It's set to False in experiments as ELMo results are cached,
                      but please set it to default True when running on new sentences.
    """
    def __init__(self, allow_tensorflow=True):
        self.elmo_processor = ElmoProcessor(allow_tensorflow)
        self.esa_processor = EsaProcessor()
        self.inference_processor = InferenceProcessor("figer")
        self.evaluator = Evaluator()
        self.evaluated = []

    """
    Process a single sentence
    @sentence: a sentence in zoe_utils.Sentence structure
    @return: a sentence in zoe_utils that has predicted types set
    """
    def process_sentence(self, sentence, inference_processor=None):
        esa_candidates = self.esa_processor.get_candidates(sentence)
        elmo_candidates = self.elmo_processor.rank_candidates(sentence, esa_candidates)
        if len(elmo_candidates) > 0 and elmo_candidates[0][0] == self.elmo_processor.stop_sign:
            return -1
        if inference_processor is None:
            inference_processor = self.inference_processor
        inference_processor.inference(sentence, elmo_candidates, esa_candidates)
        return sentence

    def process_sentence_vec(self, sentence, inference_processor=None):
        esa_candidates = self.esa_processor.get_candidates(sentence)
        elmo_candidates = self.elmo_processor.rank_candidates_vec(sentence, esa_candidates)
        if len(elmo_candidates) > 0 and elmo_candidates[0][0] == self.elmo_processor.stop_sign:
            return -1
        if inference_processor is None:
            inference_processor = self.inference_processor
        inference_processor.inference(sentence, elmo_candidates, esa_candidates)
        return sentence

    """
    Helper function to evaluate on a dataset that has multiple sentences
    @file_name: A string indicating the data file. 
                Note the format needs to be the common json format, see examples
    @mode: A string indicating the mode. This adjusts the inference mode, and set caches etc.
    @return: None
    """
    def evaluate_dataset(self, file_name, mode, do_inference=True, use_prior=True, use_context=True, size=-1):
        if not os.path.isfile(file_name):
            print("[ERROR] Invalid input data file.")
            return
        self.inference_processor = InferenceProcessor(mode, do_inference, use_prior, use_context)
        dataset = DataReader(file_name, size)
        for sentence in dataset.sentences:
            processed = self.process_sentence(sentence)
            if processed == -1:
                continue
            self.evaluated.append(processed)
            processed.print_self()
            evaluator = Evaluator()
            evaluator.print_performance(self.evaluated)

    """
    Helper function that saves the predicted sentences list to a file.
    @file_name: A string indicating the target file path. 
                Note it will override the content
    @return: None
    """
    def save(self, file_name):
        with open(file_name, "wb") as handle:
            pickle.dump(self.evaluated, handle, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def evaluate_saved_runlog(log_name):
        with open(log_name, "rb") as handle:
            sentences = pickle.load(handle)
        evaluator = Evaluator()
        evaluator.print_performance(sentences)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[ERROR] choose from 'figer', 'bbn', 'ontonotes' or 'eval'")
        exit(0)
    if sys.argv[1] == "figer":
        runner = ZoeRunner(allow_tensorflow=False)
        runner.elmo_processor.load_cached_embeddings("data/FIGER/target.min.embedding.pickle", "data/FIGER/wikilinks.min.embedding.pickle")
        runner.evaluate_dataset("data/FIGER/test_sampled.json", "figer")
        runner.save("data/log/runlog_figer.pickle")
    if sys.argv[1] == "bbn":
        runner = ZoeRunner(allow_tensorflow=False)
        runner.elmo_processor.load_cached_embeddings("data/BBN/target.min.embedding.pickle", "data/BBN/wikilinks.min.embedding.pickle")
        runner.evaluate_dataset("data/BBN/test.json", "bbn")
        runner.save("data/log/runlog_bbn.pickle")
    if sys.argv[1] == "ontonotes":
        runner = ZoeRunner(allow_tensorflow=False)
        runner.elmo_processor.load_cached_embeddings("data/ONTONOTES/target.min.embedding.pickle", "data/ONTONOTES/wikilinks.min.embedding.pickle")
        runner.evaluate_dataset("data/ONTONOTES/test.json", "ontonotes", size=1000)
        runner.save("data/log/runlog_ontonotes.pickle")
    if sys.argv[1] == "eval" and len(sys.argv) > 2:
        ZoeRunner.evaluate_saved_runlog(sys.argv[2])
