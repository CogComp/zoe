import math
import os
import pickle

import numpy as np
import tensorflow as tf

from bilm import dump_bilm_embeddings_inner, dump_bilm_embeddings


class ElmoProcessor:

    RANKED_RETURN_NUM = 20

    def __init__(self):
        self.datadir = os.path.join('bilm-tf', 'model')
        self.vocab_file = os.path.join(self.datadir, 'vocab_test.txt')
        self.options_file = os.path.join(self.datadir, 'options.json')
        self.weight_file = os.path.join(self.datadir, 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5')
        with open('data/sent_example.pickle', 'rb') as handle:
            self.sent_example_map = pickle.load(handle)

    def process_batch(self, sentences):
        tokenized_context = [sentence.split() for sentence in sentences]
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

        ret_map = {}
        for sent_id in range(0, len(sentences)):
            sent_embedding = embedding_map[sent_id]
            for i in range(0, len(tokenized_context[sent_id])):
                key = tokenized_context[sent_id][i]
                concat = np.concatenate([
                    sent_embedding[0][i],
                    sent_embedding[1][i],
                    sent_embedding[2][i]
                ])
                if key in ret_map:
                    ret_map[key] = ret_map[key] + concat
                else:
                    ret_map[key] = concat
                assert(len(ret_map[key]) == 3 * 1024)
        ret_map_avg = {}
        for key in ret_map:
            dividend = freq_map[key]
            ret_map_avg[key] = list(ret_map[key] / dividend)
        tf.reset_default_graph()
        return ret_map_avg

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

    @staticmethod
    def cosine(vec_a, vec_b):
        assert(len(vec_a) == len(vec_b))
        square_a = 0.0
        square_b = 0.0
        score_mul = 0.0
        for i in range(0, len(vec_a)):
            square_a += float(vec_a[i]) * float(vec_a[i])
            square_b += float(vec_b[i]) * float(vec_b[i])
            score_mul += float(vec_a[i]) * float(vec_b[i])
        return score_mul / math.sqrt(square_a * square_b)

    def rank_candidates(self, sentence, candidates):
        sentences_to_process = [sentence.get_sent_str()]
        for candidate in candidates:
            if candidate not in self.sent_example_map:
                continue
            example_sentences_str = self.sent_example_map[candidate]
            example_sentences = example_sentences_str.split("|")
            for i in range(0, min(len(example_sentences), 10)):
                sentences_to_process.append(example_sentences[i])
        elmo_map = self.process_batch(sentences_to_process)

        target_vec = elmo_map[sentence.get_mention_surface()]
        results = {}
        for candidate in candidates:
            if candidate in elmo_map:
                results[candidate] = ElmoProcessor.cosine(target_vec, elmo_map[candidate])
            else:
                results[candidate] = 0.0
        sorted_results = sorted(results.items(), key=lambda kv: kv[1], reverse=True)
        return [x[0] for x in sorted_results][:self.RANKED_RETURN_NUM]


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

    def get_candidates(self, sentence):
        tokens = sentence.tokens
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


class Sentence:

    def __init__(self, tokens, mention_start, mention_end):
        self.tokens = tokens
        self.mention_start = mention_start
        self.mention_end = mention_end

    def get_mention_surface(self):
        concat = ""
        for i in range(self.mention_start, self.mention_end):
            concat += self.tokens[i] + "_"
        if len(concat) > 0:
            concat = concat[:-1]
        return concat

    def get_sent_str(self):
        concat = ""
        for i in range(0, len(self.tokens)):
            if i == self.mention_start:
                concat += self.get_mention_surface()
                i = self.mention_end
            else:
                concat += self.tokens[i]
            concat += " "
        if len(concat) > 0:
            concat = concat[:-1]
        return concat
