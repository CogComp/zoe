import time
import sqlite3
from flask import g


class ElmoProcessorCache:

    CLEANUP_THRESHOLD = 10000

    def __init__(self, elmo_processor):
        self.elmo_processor = elmo_processor
        self.added_count = 0
        self.initialized = False

    @staticmethod
    def compute_sig(sentence):
        return sentence.get_sent_str() + "|||" + sentence.get_mention_surface()

    @staticmethod
    def get_mem_db():
        if 'mem_db' not in g:
            g.mem_db = sqlite3.connect("./shared_cache.db")
        return g.mem_db

    def initialize_cache(self):
        db = ElmoProcessorCache.get_mem_db()
        cursor = db.cursor()
        cursor.execute("DROP TABLE IF EXISTS memcache")
        cursor.execute("CREATE TABLE memcache (key TEXT PRIMARY KEY, value TEXT, time INTEGER)")
        db.commit()
        self.added_count = 0
        self.initialized = True

    def query_cache(self, sentence):
        if not self.initialized:
            self.initialize_cache()
        db = ElmoProcessorCache.get_mem_db()
        cursor = db.cursor()
        key = ElmoProcessorCache.compute_sig(sentence)
        cursor.execute("SELECT value FROM memcache WHERE key=?", [key])
        data = cursor.fetchone()
        if data is None:
            value_map = self.elmo_processor.process_single_continuous(sentence.get_sent_str())
            if sentence.get_mention_surface() not in value_map:
                return []
            value = value_map[sentence.get_mention_surface()]
            current_timestamp = int(time.time())
            cursor.execute("INSERT INTO memcache VALUES (?, ?, ?)", [key, str(value), current_timestamp])
            self.added_count += 1
            if self.added_count > self.CLEANUP_THRESHOLD:
                self.initialize_cache()
            db.commit()
            return value
        else:
            result_str = data[0]
            assert(result_str[0] == '[')
            assert(result_str[-1] == ']')
            result_str = result_str[1:-1]
            result_arr = [float(x) for x in result_str.split(",")]
            return result_arr


