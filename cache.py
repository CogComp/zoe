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


class SurfaceCache:
    def __init__(self, cache_file, server_mode=True):
        self.cache_file = cache_file
        self.server_mode = server_mode
        if not self.server_mode:
            self.surface_db = sqlite3.connect(self.cache_file)

    def get_surface_db(self):
        if self.server_mode:
            if 'surface_db' not in g:
                g.surface_db = sqlite3.connect(self.cache_file)
            return g.surface_db
        else:
            return self.surface_db

    def initialize_cache(self):
        db = self.get_surface_db()
        cursor = db.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS cache (surface TEXT PRIMARY KEY, types BLOB)")
        db.commit()

    def query_cache(self, surface, limit=5):
        self.initialize_cache()
        surface = str(surface).lower()
        db = self.get_surface_db()
        cursor = db.cursor()
        cursor.execute("SELECT types FROM cache WHERE surface=?", [surface])
        data = cursor.fetchone()
        if data is None:
            return None
        else:
            result_binary = data[0]
            cache_dict = sorted((pickle.loads(result_binary)).items(), key=lambda x: x[1], reverse=True)
            ret = []
            for i in range(0, min(limit, len(cache_dict))):
                ret.append(cache_dict[i][0])
            return ret

    def insert_cache(self, sentence):
        self.initialize_cache()
        surface = sentence.get_mention_surface().lower()
        db = self.get_surface_db()
        cursor = db.cursor()
        cursor.execute("SELECT types FROM cache WHERE surface=?", [surface])
        data = cursor.fetchone()
        if data is None:
            to_insert_cache = {}
            for t in sentence.predicted_types:
                to_insert_cache[t] = 1
            cursor.execute("INSERT INTO cache VALUES (?, ?)", [surface, pickle.dumps(to_insert_cache)])
            db.commit()
        else:
            previous_cache = pickle.loads(data[0])
            for t in sentence.predicted_types:
                if t in previous_cache:
                    previous_cache[t] += 1
                else:
                    previous_cache[t] = 1
            cursor.execute("UPDATE cache SET types=? WHERE surface=?", [pickle.dumps(previous_cache), surface])
            db.commit()
