import hashlib
import pickle
import sqlite3
import time

from flask import g


class ServerCache:

    CLEANUP_THRESHOLD = 10000

    def __init__(self):
        self.added_count = 0
        self.initialized = False

    @staticmethod
    def compute_sig(sentence):
        key_val = str(sentence.get_sent_str() + "|||" + sentence.get_mention_surface() + "|||" + sentence.inference_signature).encode('utf-8')
        return hashlib.sha224(key_val).hexdigest()

    @staticmethod
    def get_mem_db():
        if 'mem_db' not in g:
            g.mem_db = sqlite3.connect("./shared_cache.db")
        return g.mem_db

    def initialize_cache(self):
        db = ServerCache.get_mem_db()
        cursor = db.cursor()
        cursor.execute("DROP TABLE IF EXISTS memcache")
        cursor.execute("CREATE TABLE memcache (key TEXT PRIMARY KEY, value BLOB, time INTEGER)")
        db.commit()
        self.added_count = 0
        self.initialized = True

    def query_cache(self, sentence):
        if not self.initialized:
            self.initialize_cache()
        db = ServerCache.get_mem_db()
        cursor = db.cursor()
        key = ServerCache.compute_sig(sentence)
        cursor.execute("SELECT value FROM memcache WHERE key=?", [key])
        data = cursor.fetchone()
        if data is None:
            return None
        else:
            result_binary = data[0]
            return pickle.loads(result_binary)

    def insert_cache(self, sentence):
        if not self.initialized:
            self.initialize_cache()
        db = ServerCache.get_mem_db()
        cursor = db.cursor()
        key = ServerCache.compute_sig(sentence)
        current_timestamp = int(time.time())
        data = pickle.dumps(sentence)
        cursor.execute("INSERT INTO memcache VALUES (?, ?, ?)", [key, data, current_timestamp])
        db.commit()
        self.added_count += 1
        if self.added_count > self.CLEANUP_THRESHOLD:
            self.initialize_cache()
