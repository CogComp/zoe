import os
import pickle
import sys
import sqlite3

from zoe_utils import ElmoProcessor

def convert_esa_map(esa_file_name, freq_file_name, invcount_file_name):
    esa_map = {}
    with open(esa_file_name) as f:
        for line in f:
            line = line.strip()
            if len(line.split("\t")) <= 1:
                continue
            key = line.split("\t")[0]
            value = line.split("\t")[1]
            esa_map[key] = value
    with open('data/esa/esa.pickle', 'wb') as handle:
        pickle.dump(esa_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

    freq_map = {}
    with open(freq_file_name) as f:
        for line in f:
            line = line.strip()
            if len(line.split("\t")) <= 1:
                continue
            key = line.split("\t")[0]
            value = int(line.split("\t")[1])
            freq_map[key] = value
    with open('data/esa/freq.pickle', 'wb') as handle:
        pickle.dump(freq_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

    invcount_map = {}
    with open(invcount_file_name) as f:
        for line in f:
            line = line.strip()
            if len(line.split("\t")) <= 1:
                continue
            key = line.split("\t")[0]
            value = int(line.split("\t")[1])
            invcount_map[key] = value
    with open('data/esa/invcount.pickle', 'wb') as handle:
        pickle.dump(invcount_map, handle, protocol=pickle.HIGHEST_PROTOCOL)


def convert_wikilinks_sent_examples(sent_example_file_name):
    sent_example_map = {}
    max_bytes = 2 ** 31 - 1
    with open(sent_example_file_name) as f:
        for line in f:
            line = line.strip()
            if len(line.split("\t")) <= 1:
                continue
            key = line.split("\t")[0]
            value = line.split("\t")[1]
            sent_example_map[key] = value
    bytes_out = pickle.dumps(sent_example_map, protocol=pickle.HIGHEST_PROTOCOL)
    with open('data/sent_example.pickle', 'wb') as handle:
        for idx in range(0, len(bytes_out), max_bytes):
            handle.write(bytes_out[idx:idx + max_bytes])


def convert_freebase(freebase_file_name, freebase_sup_file_name):
    ret_map = {}
    with open(freebase_file_name) as f:
        for line in f:
            line = line.strip()
            if len(line.split("\t")) <= 1:
                continue
            key = line.split("\t")[0]
            val = line.split("\t")[1]
            ret_map[key] = val
    with open(freebase_sup_file_name) as f:
        for line in f:
            line = line.strip()
            if len(line.split("\t")) <= 1:
                continue
            key = line.split("\t")[0]
            val = line.split("\t")[1]
            if key not in ret_map:
                ret_map[key] = val
    with open('data/title2freebase.pickle', 'wb') as handle:
        pickle.dump(ret_map, handle, protocol=pickle.HIGHEST_PROTOCOL)


def convert_prob(prob_file_name, n2c_fil2_name):
    prob_map = {}
    n2c_map = {}
    with open(n2c_fil2_name) as f:
        for line in f:
            line = line.strip()
            n2c_map[line.split("\t")[0]] = line.split("\t")[1]
    with open(prob_file_name) as f:
        for line in f:
            line = line.strip()
            key = line.split("\t")[0]
            val = float(line.split("\t")[1])
            surface = key.split("|")[0]
            title = key.split("|")[1]
            if title in n2c_map:
                title = n2c_map[title]
            if surface in prob_map:
                cur_highest = prob_map[surface][1]
                if val > cur_highest:
                    prob_map[surface] = (title, val)
            else:
                prob_map[surface] = (title, val)
    with open('data/prior_prob.pickle', 'wb') as handle:
        pickle.dump(prob_map, handle, protocol=pickle.HIGHEST_PROTOCOL)


def convert_cached_embeddings(raw_file_name, output_file_name):
    ret_map = {}
    max_bytes = 2 ** 31 - 1
    with open(raw_file_name) as f:
        for line in f:
            line = line.strip()
            token = line.split("\t")[0]
            if line.split("\t")[1] == "null":
                continue
            vals = line.split("\t")[1].split(",")
            vec = []
            for val in vals:
                vec.append(float(val))
            ret_map[token] = vec
    bytes_out = pickle.dumps(ret_map, protocol=pickle.HIGHEST_PROTOCOL)
    with open(output_file_name, 'wb') as handle:
        for idx in range(0, len(bytes_out), max_bytes):
            handle.write(bytes_out[idx:idx + max_bytes])


def reduce_cache_file_size(cache_pickle_file_name, title_file_name, out_file_name):
    with open(cache_pickle_file_name, "rb") as handle:
        cache_map = pickle.load(handle)
    title_set = set()
    with open(title_file_name, "r") as f:
        for line in f:
            line = line.strip()
            title_set.add(line)
    ret_map = {}
    for key in cache_map:
        if key in title_set:
            ret_map[key] = cache_map[key]
    with open(out_file_name, "wb") as handle:
        pickle.dump(ret_map, handle, protocol=pickle.HIGHEST_PROTOCOL)


def check_data_file_integrity(mode=""):
    file_list = [
        'data/esa/esa.pickle',
        'data/esa/freq.pickle',
        'data/esa/invcount.pickle',
        'data/prior_prob.pickle',
        'data/sent_example.pickle',
        'data/title2freebase.pickle',
    ]
    corpus_supplements = []
    if mode == "figer":
        corpus_supplements = [
            'data/FIGER/target.embedding.pickle',
            'data/FIGER/wikilinks.embedding.pickle',
            'mapping/figer.mapping',
            'mapping/figer.logic.mapping'
        ]
    passed = True
    for file in file_list + corpus_supplements:
        if not os.path.isfile(file):
            print("[ERROR]: Missing " + file)
            passed = False
    if not passed:
        print("You have one or more file missing. Please refer to README for solution.")
    else:
        print("All required or suggested files are here. Go ahead and run experiments!")


def compare_runlogs(runlog_file_a, runlog_file_b):
    if not os.path.isfile(runlog_file_a) or not os.path.isfile(runlog_file_b):
        print("Invalid input file names")
    with open (runlog_file_a, "rb") as handle:
        log_a = pickle.load(handle)
    with open (runlog_file_b, "rb") as handle:
        log_b = pickle.load(handle)
    for sentence in log_a:
        for compare_sentence in log_b:
            if sentence.get_sent_str() == compare_sentence.get_sent_str():
                if sentence.get_mention_surface() == compare_sentence.get_mention_surface():
                    if sentence.predicted_types != compare_sentence.predicted_types:
                        print(sentence.get_sent_str())
                        print(sentence.get_mention_surface())
                        print(sentence.gold_types)
                        print("Log A prediction: " + str(sentence.predicted_types))
                        print("Log B prediction: " + str(compare_sentence.predicted_types))


def produce_cache():
    elmo_processor = ElmoProcessor(allow_tensorflow=True)
    to_process = []
    to_process_concepts = []
    sorted_pairs = sorted(elmo_processor.sent_example_map.items())
    cur_processing_file_num = ord(sorted_pairs[0][0][0])
    sub_map_index = 0
    max_bytes = 2 ** 31 - 1
    for pair in sorted_pairs:
        concept = pair[0]
        file_num = ord(concept[0])
        if file_num != cur_processing_file_num or len(to_process) > 10000:
            new_start = False
            if file_num != cur_processing_file_num:
                new_start = True
            out_file_name = "data/cache/batch_" + str(cur_processing_file_num) + "_" + str(sub_map_index) + ".pickle"
            if new_start:
                out_file_name = "data/cache/batch_" + str(cur_processing_file_num) + ".pickle"
            if not os.path.isfile(out_file_name) and cur_processing_file_num >= 65:
                print("Prepared to run ELMo on " + chr(cur_processing_file_num))
                print("This batch contains " + str(len(to_process_concepts)) + " concepts, and " + str(len(to_process)) + " sentences.")
                elmo_map = elmo_processor.process_batch(to_process)
                batch_map = {}
                for processed_concept in to_process_concepts:
                    if processed_concept in elmo_map:
                        batch_map[processed_concept] = elmo_map[processed_concept]
                bytes_out = pickle.dumps(batch_map, protocol=pickle.HIGHEST_PROTOCOL)
                with open(out_file_name, "wb") as handle:
                    for idx in range(0, len(bytes_out), max_bytes):
                        handle.write(bytes_out[idx:idx + max_bytes])
                print("Processed all concepts start with " + chr(cur_processing_file_num))
                print()
            to_process = []
            to_process_concepts = []
            if new_start:
                cur_processing_file_num = file_num
                sub_map_index = 0
            else:
                sub_map_index += 1
        example_sentences_str = elmo_processor.sent_example_map[concept]
        example_sentences = example_sentences_str.split("|||")
        for i in range(0, min(len(example_sentences), 10)):
            to_process.append(example_sentences[i])
        to_process_concepts.append(concept)


def combine_caches():
    conn = sqlite3.connect('data/elmo_cache.db')
    cur = conn.cursor()
    cur.execute("CREATE TABLE data (title VARCHAR(256), value TEXT)")
    for file_name in os.listdir('data/cache'):
        with open("data/cache/" + file_name, "rb") as handle:
            m = pickle.load(handle)
            for key in m:
                cur.execute("INSERT INTO data(title, value) VALUES(?, ?)", [key, str(m[key])])
        conn.commit()


def test_caches(title):
    conn = sqlite3.connect('data/elmo_cache.db')
    cur = conn.cursor()
    cur.execute("SELECT * FROM data WHERE title=?", [title])
    print(cur.fetchone())

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("[ERROR]: No command given.")
        exit(0)
    if sys.argv[1] == "CHECKFILE":
        if len(sys.argv) == 2:
            check_data_file_integrity()
        else:
            check_data_file_integrity(sys.argv[2])
    if sys.argv[1] == "COMPARE":
        if len(sys.argv) < 4:
            print("Need two files for comparison.")
        compare_runlogs(sys.argv[2], sys.argv[3])
    if sys.argv[1] == "CACHE":
        produce_cache()
    if sys.argv[1] == "COMBINECACHE":
        combine_caches()
    if sys.argv[1] == "TESTCACHE":
        test_caches(sys.argv[2])
