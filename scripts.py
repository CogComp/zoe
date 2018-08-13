import pickle


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
