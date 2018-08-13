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
    ret_map = {}
    for key in prob_map:
        ret_map[key] = prob_map[key][0]
    with open('data/prior_prob.pickle', 'wb') as handle:
        pickle.dump(ret_map, handle, protocol=pickle.HIGHEST_PROTOCOL)


convert_freebase('/Volumes/Research/Process/title2freebase.txt', '/Volumes/Research/Process/title2freebase_sup.txt')


