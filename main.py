import os
import pickle
import sys

from zoe_utils import DataReader
from zoe_utils import ElmoProcessor
from zoe_utils import EsaProcessor
from zoe_utils import Evaluator
from zoe_utils import InferenceProcessor


class ZoeRunner:

    def __init__(self):
        self.elmo_processor = ElmoProcessor()
        self.esa_processor = EsaProcessor()
        self.inference_processor = InferenceProcessor("figer")
        self.evaluator = Evaluator()
        self.evaluated = []

    """
    Process a single sentence
    @sentence: a sentence in zoe_utils.Sentence structure
    @return: a sentence in zoe_utils that has predicted types set
    """
    def process_sentence(self, sentence):
        esa_candidates = self.esa_processor.get_candidates(sentence)
        elmo_candidates = self.elmo_processor.rank_candidates(sentence, esa_candidates)
        self.inference_processor.inference(sentence, elmo_candidates, esa_candidates)
        return sentence

    """
    Helper function to evaluate on a dataset that has multiple sentences
    @file_name: A string indicating the data file. 
                Note the format needs to be the common json format, see examples
    @mode: A string indicating the mode. This adjusts the inference mode, and set caches etc.
    @return: None
    """
    def evaluate_dataset(self, file_name, mode):
        if not os.path.isfile(file_name):
            print("[ERROR] Invalid input data file.")
            return
        self.inference_processor = InferenceProcessor(mode)
        if mode == "figer":
            self.elmo_processor.load_cached_embeddings("data/FIGER/target.embedding.pickle", "data/FIGER/wikilinks.embedding.pickle")
        dataset = DataReader(file_name)
        for sentence in dataset.sentences:
            processed = self.process_sentence(sentence)
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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[ERROR] No running mode given.")
        exit(0)
    if sys.argv[1] == "figer":
        runner = ZoeRunner()
        runner.evaluate_dataset("data/FIGER/test_sampled.json", "figer")
        runner.save("data/FIGER/runlog.pickle")
