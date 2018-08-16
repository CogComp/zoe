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

    #
    # sentence: a sentence in Sentence data-structure
    def process_sentence(self, sentence):
        esa_candidates = self.esa_processor.get_candidates(sentence)
        elmo_candidates = self.elmo_processor.rank_candidates(sentence, esa_candidates)
        types = self.inference_processor.inference(sentence, elmo_candidates, esa_candidates)
        sentence.set_predictions(types)
        return sentence

    def evaluate_dataset(self, file_name, mode):
        self.inference_processor = InferenceProcessor(mode)
        if mode == "figer":
            self.elmo_processor.load_cached_embeddings("data/FIGER/target.embedding.pickle", "data/FIGER/wikilinks.embedding.pickle")
        dataset = DataReader(file_name)
        to_evaluate = []
        for sentence in dataset.sentences:
            processed = self.process_sentence(sentence)
            to_evaluate.append(processed)
            evaluator = Evaluator()
            evaluator.print_performance(to_evaluate)

    def fun(self):
        self.inference_processor = InferenceProcessor("figer")
        t = self.inference_processor.get_mapped_types_of_title("United_States")
        print(t)


runner = ZoeRunner()
runner.evaluate_dataset("data/FIGER/test_single.json", "figer")
