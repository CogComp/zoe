from zoe_utils import ElmoProcessor
from zoe_utils import EsaProcessor
from zoe_utils import InferenceProcessor
from zoe_utils import Sentence


class ZoeRunner:

    def __init__(self):
        self.elmo_processor = ElmoProcessor()
        self.esa_processor = EsaProcessor()
        self.inference_processor = InferenceProcessor("figer")

    #
    # tokens: a list of tokens
    def process_sentence(self, tokens):
        pass

    def fun(self):
        fun_sentence = Sentence(['Barack', 'Obama', 'is', 'a', 'good', 'president', '.'], 0, 2, "")
        print(fun_sentence.get_mention_surface())
        print(fun_sentence.get_sent_str())
        print(self.inference_processor.get_prob_title("usa"))
        print(self.inference_processor.get_prob_title("barack"))
        candidates = self.esa_processor.get_candidates(fun_sentence)
        selected = self.elmo_processor.rank_candidates(fun_sentence, candidates)
        print(selected)


runner = ZoeRunner()
runner.fun()
