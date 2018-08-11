import math
import os
import pickle

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
        sent_count = len(tokenized_context)
        freq_map = {}

        for i in range(0, len(tokenized_context)):
            for j in range(0, len(tokenized_context[i])):
                key = tokenized_context[i][j]
                if key in freq_map:
                    freq_map[key] = freq_map[key] + 1.0
                else:
                    freq_map[key] = 1.0

        embedding_map = dump_bilm_embeddings(
            self.vocab_file, sentences, self.options_file, self.weight_file
        )

        for i in range(0, sent_count):
            sentence_embeddings = embedding_map[i]
            for vecidx in range(0, 3):
                sentence_embeddings_curvec = sentence_embeddings[vecidx]
                curMap = embedding_map[vecidx]
                for tok in range(0, len(sentence_embeddings_curvec)):
                    key = tokenized_context[i][tok]
                    if key in curMap:
                        curMap[key] = curMap[key] + sentence_embeddings_curvec[tok]
                    else:
                        curMap[key] = sentence_embeddings_curvec[tok]

        ret_map = {}
        for key in embedding_map[0]:
            dividend = freq_map[key]
            ret_map[key] = list(embedding_map[0][key] / dividend) + list(embedding_map[1][key] / dividend) + list(embedding_map[2][key] / dividend)
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


class EsaProcessor:

    N_DOCUMENTS = 24504233.0
    RETURN_NUM = 300

    def __init__(self):
        with open('data/esa/esa.pickle', 'rb') as handle:
            self.esa_map = pickle.load(handle)
        with open('data/esa/freq.pickle', 'rb') as handle:
            self.freq_map = pickle.load(handle)
        with open('data/esa/invcount.pickle', 'rb') as handle:
            self.invcount_map = pickle.load(handle)

    @staticmethod
    def str2map(map_val):
        ret_map = {}
        entries = map_val.split("|")
        for entry in entries:
            key = entry.split("::")[0]
            val = entry.split("::")[1]
            ret_map[key] = float(val)
        return ret_map

    def get_candidates(self, tokens):
        overall_map = {}
        doc_freq_map = {}
        max_acc = 0
        for token in tokens:
            if token in doc_freq_map:
                acc = doc_freq_map[token] + 1
            else:
                acc = 1
            if acc > max_acc:
                max_acc = acc
            doc_freq_map[token] = acc
        for token in tokens:
            if token in self.esa_map:
                idf_score = math.log(self.N_DOCUMENTS / float(self.freq_map[token]))
                tf_score = 0.5 + 0.5 * (float(doc_freq_map[token]) / float(max_acc))
                inv_freq = float(self.invcount_map[token])
                sub_map = EsaProcessor.str2map(self.esa_map[token])
                for key in sub_map:
                    weight = idf_score * tf_score * sub_map[key] / inv_freq
                    if key in overall_map:
                        overall_map[key] = overall_map[key] + weight
                    else:
                        overall_map[key] = weight
        sorted_overall_map = sorted(overall_map.items(), key=lambda kv: kv[1], reverse=True)
        return [x[0] for x in sorted_overall_map][:self.RETURN_NUM]

