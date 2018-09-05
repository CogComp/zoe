import json
from main import ZoeRunner
from zoe_utils import Sentence
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
    def handle_root():
        return send_from_directory('./frontend', 'index.html')

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
        self.runner.process_sentence(sentence)
        print("Processed mention " + sentence.get_mention_surface())
        ret["type"] = list(sentence.predicted_types)
        ret["candidates"] = sentence.elmo_candidate_titles
        return json.dumps(ret)

    """
    Handler to start the Flask app
    @localhost: Whether the server lives only in localhost
    @port: A port number, default to 80 (Web)
    """
    def start(self, localhost=False, port=80):
        self.app.add_url_rule("/", "", self.handle_root)
        self.app.add_url_rule("/annotate", "annotate", self.handle_input, methods=['POST'])
        if localhost:
            self.app.run()
        else:
            self.app.run(host='0.0.0.0', port=port)


if __name__ == '__main__':
    server = Server("/Users/xuanyuzhou/Downloads/elmo_cache_correct.db")
    server.start(localhost=True)

