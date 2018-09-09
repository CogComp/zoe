import json
from main import ZoeRunner
from zoe_utils import Sentence
from zoe_utils import InferenceProcessor
from flask import Flask
from flask import request
from flask import send_from_directory
from flask_cors import CORS


class Server:

    """
    Initialize the server with needed resources
    @sql_db_path: The path pointing to the ELMo caches sqlite file
    """
    def __init__(self, sql_db_path):
        self.app = Flask(__name__)
        CORS(self.app)
        self.runner = ZoeRunner(allow_tensorflow=True)
        self.runner.elmo_processor.load_sqlite_db(sql_db_path, server_mode=True)

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
    """
    def handle_input(self):
        ret = {}
        r = request.get_json()
        if "tokens" not in r or "mention_start" not in r or "mention_end" not in r:
            ret["type"] = ["INVALID_INPUT"]
            ret["candidates"] = []
            return json.dumps(ret)
        sentence = Sentence(r["tokens"], r["mention_start"], r["mention_end"], "")
        mode = r["mode"]
        if mode != "figer":
            if mode != "custom":
                selected_inference_processor = InferenceProcessor(mode, resource_loader=self.runner.inference_processor)
                self.runner.process_sentence(sentence, selected_inference_processor)
            else:
                rules = r["taxonomy"]
                mappings = self.parse_custom_rules(rules)
                custom_inference_processor = InferenceProcessor(mode, custom_mapping=mappings)
                self.runner.process_sentence(sentence, custom_inference_processor)
        else:
            self.runner.process_sentence(sentence)
        print("Processed mention " + sentence.get_mention_surface() + " in mode " + mode)
        ret["type"] = list(sentence.predicted_types)
        ret["candidates"] = sentence.elmo_candidate_titles
        return json.dumps(ret)

    """
    Handler to start the Flask app
    @localhost: Whether the server lives only in localhost
    @port: A port number, default to 80 (Web)
    """
    def start(self, localhost=False, port=80):
        self.app.add_url_rule("/", "", self.handle_redirection)
        self.app.add_url_rule("/<path:path>", "<path:path>", self.handle_root)
        self.app.add_url_rule("/annotate", "annotate", self.handle_input, methods=['POST'])
        if localhost:
            self.app.run()
        else:
            self.app.run(host='0.0.0.0', port=port)


if __name__ == '__main__':
    server = Server("/Users/xuanyuzhou/Downloads/elmo_cache_correct.db")
    server.start(localhost=True)

