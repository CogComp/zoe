import os

import tensorflow as tf

from bilm import dump_bilm_embeddings_inner, dump_bilm_embeddings


class ElmoProcessor:

    def __init__(self):
        self.datadir = os.path.join('bilm-tf', 'model')
        self.vocab_file = os.path.join(self.datadir, 'vocab_test.txt')
        self.options_file = os.path.join(self.datadir, 'options.json')
        self.weight_file = os.path.join(self.datadir, 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5')

    def process_batch(self, sentences):
        tokenized_context = [sentence.split() for sentence in sentences]
        SENT_COUNT = len(tokenized_context)
        FREQ_MAP = {}
        EMBEDDING_MAP_0 = {}
        EMBEDDING_MAP_1 = {}
        EMBEDDING_MAP_2 = {}
        EMBEDDING_MAP = [EMBEDDING_MAP_0, EMBEDDING_MAP_1, EMBEDDING_MAP_2]

        for i in range(0, len(tokenized_context)):
            for j in range(0, len(tokenized_context[i])):
                key = tokenized_context[i][j]
                if key in FREQ_MAP:
                    FREQ_MAP[key] = FREQ_MAP[key] + 1.0
                else:
                    FREQ_MAP[key] = 1.0

        embedding_map = dump_bilm_embeddings(
            self.vocab_file, sentences, self.options_file, self.weight_file
        )

        for i in range(0, SENT_COUNT):
            sentence_embeddings = embedding_map[i]
            for vecidx in range(0, 3):
                sentence_embeddings_curvec = sentence_embeddings[vecidx]
                curMap = EMBEDDING_MAP[vecidx]
                for tok in range(0, len(sentence_embeddings_curvec)):
                    key = tokenized_context[i][tok]
                    if key in curMap:
                        curMap[key] = curMap[key] + sentence_embeddings_curvec[tok]
                    else:
                        curMap[key] = sentence_embeddings_curvec[tok]

        ret_map = {}
        for key in EMBEDDING_MAP[0]:
            dividend = FREQ_MAP[key]
            ret_map[key] = list(EMBEDDING_MAP[0][key] / dividend) + list(EMBEDDING_MAP[1][key] / dividend) + list(EMBEDDING_MAP[2][key] / dividend)
            assert(len(ret_map[key]) == 3 * 1024)
        tf.reset_default_graph()
        return ret_map

    def process_single(self, sentence):
        tokens = sentence.split()
        embedding = dump_bilm_embeddings_inner(
            self.vocab_file, sentence, self.options_file, self.weight_file
        )
        assert(len(embedding[0]) == len(tokens))
        ret_map = {}
        for i in range(0, len(tokens)):
            ret_map[tokens[i]] = list(embedding[0][i]) + list(embedding[1][i]) + list(embedding[2][i])
            assert(len(ret_map[tokens[i]]) == 3 * 1024)
        tf.reset_default_graph()
        return ret_map
