import os

from bilm import dump_bilm_embeddings_inner


def generate_embedding(sentence):
    FREQ_MAP = {}

    tokens = sentence.split()

    for i in range(0, len(tokens)):
        key = tokens[i]
        if key in FREQ_MAP:
            FREQ_MAP[key] = FREQ_MAP[key] + 1.0
        else:
            FREQ_MAP[key] = 1.0

    datadir = os.path.join('bilm-tf', 'model')
    vocab_file = os.path.join(datadir, 'vocab_test.txt')
    options_file = os.path.join(datadir, 'options.json')
    weight_file = os.path.join(datadir, 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5')

    embedding = dump_bilm_embeddings_inner(
        vocab_file, sentence, options_file, weight_file
    )
    return embedding

