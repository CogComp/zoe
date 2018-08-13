from zoe_utils import ElmoProcessor
from zoe_utils import EsaProcessor
from zoe_utils import Sentence


class ZoeRunner:

    def __init__(self):
        self.elmo_processor = ElmoProcessor()
        self.esa_processor = EsaProcessor()

    #
    # tokens: a list of tokens
    def process_sentence(self, tokens):
        pass

    def fun(self):
        print(self.elmo_processor.process_single('I like basketball .'))
        print(self.elmo_processor.process_batch(['I like basketball .', 'I like basketball .']))
        fun_sentence = Sentence(['Barack', 'Obama', 'is', 'a', 'good', 'president', '.'], 0, 2)
        candidates = self.esa_processor.get_candidates(fun_sentence)
        selected = self.elmo_processor.rank_candidates(fun_sentence, candidates)
        print(selected)

runner = ZoeRunner()
runner.fun()
