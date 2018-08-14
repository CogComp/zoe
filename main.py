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
        print(esa_candidates)
        elmo_candidates = self.elmo_processor.rank_candidates(sentence, esa_candidates)
        print(elmo_candidates)
        types = self.inference_processor.inference(sentence, elmo_candidates, esa_candidates)
        print(types)
        sentence.set_predictions(types)
        return sentence

    def evaluate_dataset(self, file_name, mode):
        self.inference_processor = InferenceProcessor(mode)
        dataset = DataReader(file_name)
        to_evaluate = []
        for sentence in dataset.sentences:
            to_evaluate.append(self.process_sentence(sentence))
        self.evaluator.print_performance(to_evaluate)


runner = ZoeRunner()
runner.evaluate_dataset("data/FIGER/test_sampled.json", "figer")
