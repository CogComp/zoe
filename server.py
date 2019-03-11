import json
import signal
import time
import traceback

import requests
from ccg_nlpy import local_pipeline
from flask import Flask
from flask import request
from flask import send_from_directory
from flask_cors import CORS

from cache import ServerCache
from cache import SurfaceCache
from main import ZoeRunner
from zoe_utils import InferenceProcessor
from zoe_utils import Sentence


class CogCompLoggerClient:
    def __init__(self, demo_name, base_url="http://127.0.0.1:5000"):
        self.demo_name = demo_name
        self.base_url = base_url
        if self.base_url.endswith("/"):
            self.url = self.base_url + "log"
        else:
            self.url = self.base_url + "/log"

    def log(self, content=""):
        params = {
            'entry_name': self.demo_name,
            'content': content
        }
        result = requests.post(url=self.url, params=params).json()
        if result['result'] == 'SUCCESS':
            return True
        return False

    def log_dict(self, d=None):
        if d is None:
            return self.log()
        else:
            return self.log(content=json.dumps(d))


class Server:

    """
    Initialize the server with needed resources
    @sql_db_path: The path pointing to the ELMo caches sqlite file
    """
    def __init__(self, sql_db_path, surface_cache_path):
        self.app = Flask(__name__)
        CORS(self.app)
        self.mem_cache = ServerCache()
        self.surface_cache = SurfaceCache(surface_cache_path)
        self.pipeline = local_pipeline.LocalPipeline()
        self.pipeline_initialize_helper(['.'])
        self.logger = CogCompLoggerClient("zoe", base_url="http://macniece.seas.upenn.edu:4005")
        self.runner = ZoeRunner(allow_tensorflow=True)
        status = self.runner.elmo_processor.load_sqlite_db(sql_db_path, server_mode=True)
        if not status:
            print("ELMo cache file is not found. Server mode is prohibited without it.")
            print("Please contact the author for this cache, or modify this code if you know what you are doing.")
            exit(1)
        self.runner.elmo_processor.rank_candidates_vec()
        signal.signal(signal.SIGINT, self.grace_end)

    @staticmethod
    def handle_root(path):
        return send_from_directory('./frontend', path)

    @staticmethod
    def handle_redirection():
        return Server.handle_root("index.html")

    def parse_custom_rules(self, rules):
        type_to_titles = {}
        freebase_freq_total = {}
        for rule in rules:
            title = rule.split("|||")[0]
            freebase_types = []
            if title in self.runner.inference_processor.freebase_map:
                freebase_types = self.runner.inference_processor.freebase_map[title].split(",")
            for ft in freebase_types:
                if ft in freebase_freq_total:
                    freebase_freq_total[ft] += 1
                else:
                    freebase_freq_total[ft] = 1
            custom_type = rule.split("|||")[1]
            if custom_type in type_to_titles:
                type_to_titles[custom_type].append(title)
            else:
                type_to_titles[custom_type] = [title]
        counter = 0
        ret = {}
        for custom_type in type_to_titles:
            freebase_freq = {}
            for title in type_to_titles[custom_type]:
                freebase_types = []
                if title in self.runner.inference_processor.freebase_map:
                    freebase_types = self.runner.inference_processor.freebase_map[title].split(",")
                    counter += 1
                for freebase_type in freebase_types:
                    if freebase_type in freebase_freq:
                        freebase_freq[freebase_type] += 1
                    else:
                        freebase_freq[freebase_type] = 1
            for ft in freebase_freq:
                if float(freebase_freq[ft]) > float(counter) * 0.5 and freebase_freq[ft] == freebase_freq_total[ft]:
                    ft = "/" + ft.replace(".", "/")
                    ret[ft] = custom_type
        return ret

    """
    Main request handler
    It requires the request to contain required information like tokens/mentions
    in the format of a json string
    
    @param_override: override API input with a pre-defined dictionary
    """
    def handle_input(self, param_override=None):
        start_time = time.time()
        ret = {}
        r = request.get_json()
        if param_override is not None:
            r = param_override
        if "tokens" not in r or "mention_starts" not in r or "mention_ends" not in r or "index" not in r:
            ret["type"] = [["INVALID_INPUT"]]
            ret["index"] = -1
            ret["mentions"] = []
            ret["candidates"] = [[]]
            return json.dumps(ret)
        sentences = []
        for i in range(0, len(r["mention_starts"])):
            sentence = Sentence(r["tokens"], int(r["mention_starts"][i]), int(r["mention_ends"][i]), "")
            sentences.append(sentence)
        mode = r["mode"]
        predicted_types = []
        predicted_candidates = []
        other_possible_types = []
        selected_candidates = []
        mentions = []
        if mode != "figer":
            if mode != "custom":
                selected_inference_processor = InferenceProcessor(mode, resource_loader=self.runner.inference_processor)
            else:
                rules = r["taxonomy"]
                mappings = self.parse_custom_rules(rules)
                selected_inference_processor = InferenceProcessor(mode, custom_mapping=mappings)
        else:
            selected_inference_processor = self.runner.inference_processor

        for sentence in sentences:
            sentence.set_signature(selected_inference_processor.signature())
            cached = self.mem_cache.query_cache(sentence)
            if cached is not None:
                sentence = cached
            else:
                self.runner.process_sentence(sentence, selected_inference_processor)
                try:
                    self.mem_cache.insert_cache(sentence)
                    self.surface_cache.insert_cache(sentence)
                except:
                    print("Cache insertion exception. Ignored.")
            predicted_types.append(list(sentence.predicted_types))
            predicted_candidates.append(sentence.elmo_candidate_titles)
            mentions.append(sentence.get_mention_surface_raw())
            selected_candidates.append(sentence.selected_title)
            other_possible_types.append(sentence.could_also_be_types)

        elapsed_time = time.time() - start_time
        print("Processed mention " + str([x.get_mention_surface() for x in sentences]) + " in mode " + mode + ". TIME: " + str(elapsed_time) + " seconds.")

        # Post logging request to Cogcomp Logger
        self.logger.log_dict(r)

        ret["type"] = predicted_types
        ret["candidates"] = predicted_candidates
        ret["mentions"] = mentions
        ret["index"] = r["index"]
        ret["selected_candidates"] = selected_candidates
        ret["other_possible_type"] = other_possible_types
        return json.dumps(ret)

    def pipeline_initialize_helper(self, tokens):
        doc = self.pipeline.doc([tokens], pretokenized=True)
        doc.get_shallow_parse
        doc.get_ner_conll
        doc.get_ner_ontonotes
        doc.get_view("MENTION")

    def handle_tokenizer_input(self):
        r = request.get_json()
        ret = {"tokens": []}
        if "sentence" not in r:
            return json.dumps(ret)
        doc = self.pipeline.doc(r["sentence"])
        token_view = doc.get_tokens
        for cons in token_view:
            ret["tokens"].append(str(cons))
        return json.dumps(ret)

    """
    Handles requests for mention filling
    """
    def handle_mention_input(self):
        r = request.get_json()
        ret = {'mention_spans': []}
        if "tokens" not in r:
            return json.dumps(ret)
        tokens = r["tokens"]
        doc = self.pipeline.doc([tokens], pretokenized=True)
        shallow_parse_view = doc.get_shallow_parse
        ner_conll_view = doc.get_ner_conll
        ner_ontonotes_view = doc.get_ner_ontonotes
        md_view = doc.get_view("MENTION")
        ret_set = set()
        ret_list = []
        additions_views = []
        if ner_ontonotes_view.cons_list is not None:
            additions_views.append(ner_ontonotes_view)
        if md_view.cons_list is not None:
            additions_views.append(md_view)
        if shallow_parse_view.cons_list is not None:
            additions_views.append(shallow_parse_view)
        try:
            if ner_conll_view.cons_list is not None:
                for ner_conll in ner_conll_view:
                    for i in range(ner_conll['start'], ner_conll['end']):
                        ret_set.add(i)
                    ret_list.append((ner_conll['start'], ner_conll['end']))
            for additions_view in additions_views:
                for cons in additions_view:
                    add_to_list = True
                    if additions_view.view_name != "MENTION":
                        if additions_view.view_name == "SHALLOW_PARSE" and cons['label'] != "NP":
                            continue
                        start = int(cons['start'])
                        end = int(cons['end'])
                    else:
                        start = int(cons['properties']['EntityHeadStartSpan'])
                        end = int(cons['properties']['EntityHeadEndSpan'])
                    for i in range(max(start - 1, 0), min(len(tokens), end + 1)):
                        if i in ret_set:
                            add_to_list = False
                            break
                    if add_to_list:
                        for i in range(start, end):
                            ret_set.add(i)
                        ret_list.append((start, end))
        except Exception as e:
            traceback.print_exc()
            print(e)
        ret['mention_spans'] = ret_list
        return json.dumps(ret)

    """
    Handles surface form cached requests
    This is expected to return sooner than actual processing
    """
    def handle_simple_input(self):
        ret = {}
        r = request.get_json()
        if "tokens" not in r or "mention_starts" not in r or "mention_ends" not in r or "index" not in r:
            ret["type"] = [["INVALID_INPUT"]]
            return json.dumps(ret)
        sentences = []
        for i in range(0, len(r["mention_starts"])):
            sentence = Sentence(r["tokens"], int(r["mention_starts"][i]), int(r["mention_ends"][i]), "")
            sentences.append(sentence)
        types = []
        for sentence in sentences:
            surface = sentence.get_mention_surface()
            cached_types = self.surface_cache.query_cache(surface)
            if cached_types is not None:
                distinct = set()
                for t in cached_types:
                    distinct.add("/" + t.split("/")[1])
                types.append(list(distinct))
            else:
                types.append([])
        ret["type"] = types
        ret["index"] = r["index"]
        return json.dumps(ret)

    def handle_word2vec_input(self):
        ret = {}
        r = request.get_json()
        if "tokens" not in r or "mention_starts" not in r or "mention_ends" not in r or "index" not in r:
            ret["type"] = [["INVALID_INPUT"]]
            return json.dumps(ret)
        sentences = []
        for i in range(0, len(r["mention_starts"])):
            sentence = Sentence(r["tokens"], int(r["mention_starts"][i]), int(r["mention_ends"][i]), "")
            sentences.append(sentence)
        predicted_types = []
        for sentence in sentences:
            self.runner.process_sentence_vec(sentence)
            predicted_types.append(list(sentence.predicted_types))
        ret["type"] = predicted_types
        ret["index"] = r["index"]
        return json.dumps(ret)

    def handle_elmo_input(self):
        ret = {}
        results = []
        r = request.get_json()
        if "sentence" not in r:
            ret["vectors"] = []
            return json.dumps(ret)
        elmo_map = self.runner.elmo_processor.process_single_continuous(r["sentence"])
        for token in r["sentence"].split():
            results.append((token, str(elmo_map[token])))
        ret["vectors"] = results
        return json.dumps(ret)

    def handle_logger_test(self):
        params = {
            "tokens": ["Iced", "Earth", "\\u2019", "s", "musical", "style", "is", "influenced", "by", "many", "traditional", "heavy", "metal", "groups", "such", "as", "Black", "Sabbath", "."],
            "index": 0,
            "mention_starts": [0],
            "mention_ends": [2],
            "mode": "figer",
            "taxonomy": [],
        }
        self.handle_input(param_override=params)
        return "finished"

    """
    Handler to start the Flask app
    @localhost: Whether the server lives only in localhost
    @port: A port number, default to 80 (Web)
    """
    def start(self, localhost=False, port=80):
        self.app.add_url_rule("/", "", self.handle_redirection)
        self.app.add_url_rule("/<path:path>", "<path:path>", self.handle_root)
        self.app.add_url_rule("/annotate", "annotate", self.handle_input, methods=['POST'])
        self.app.add_url_rule("/annotate_token", "annotate_token", self.handle_tokenizer_input, methods=['POST'])
        self.app.add_url_rule("/annotate_mention", "annotate_mention", self.handle_mention_input, methods=['POST'])
        self.app.add_url_rule("/annotate_cache", "annotate_cache", self.handle_simple_input, methods=['POST'])
        self.app.add_url_rule("/annotate_vec", "annotate_vec", self.handle_word2vec_input, methods=['POST'])
        self.app.add_url_rule("/annotate_elmo", "annotate_elmo", self.handle_elmo_input, methods=['POST'])
        # Specifically saved for logger test
        self.app.add_url_rule("/test", "test", self.handle_logger_test, methods=['POST', 'GET'])
        if localhost:
            self.app.run()
        else:
            self.app.run(host='0.0.0.0', port=port)

    def grace_end(self, signum, frame):
        print("Gracefully Existing...")
        if self.runner.elmo_processor.db_loaded:
            self.runner.elmo_processor.db_conn.close()
        print("Resource Released. Existing.")
        exit(0)


if __name__ == '__main__':
    # First argument is a placeholder. Please ask for the actual file.
    server = Server("elmo_cache_correct.db", "./data/surface_cache_new.db")
    server.start(localhost=True)

