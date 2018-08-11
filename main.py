from zoe_utils import ElmoProcessor
from zoe_utils import EsaProcessor


class ZoeRunner:

    def __init__(self):
        self.elmo_processor = ElmoProcessor()
        self.esa_processor = EsaProcessor()

    #
    # tokens: a list of tokens
    def process_sentence(self, tokens):
        pass

    def fun(self):
        candidates = self.esa_processor.get_candidates(['Barack', 'Obama', 'is', 'a', 'good', 'president', '.'])
        print(candidates)


runner = ZoeRunner()
runner.fun()
