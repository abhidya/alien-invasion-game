import hashlib
import os
import socket

import pymongo


def connect_and_collect():
    """Connect to the configured MongoDB high-score collection."""

    uri = os.environ.get("GALAGAI_MONGO_URI", "mongodb://127.0.0.1:27017")
    db_name = os.environ.get("GALAGAI_MONGO_DB", "highscore_db")
    collection_name = os.environ.get("GALAGAI_MONGO_COLLECTION", "scores")
    client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=2500)
    client.admin.command("ping")
    return client[db_name][collection_name]


def add_score(name, score, collection):
    """Add or update one high score keyed by host/name."""

    hostname = socket.gethostname()
    ip_addr = socket.gethostbyname(hostname)
    score_key = hashlib.md5(f"{ip_addr}{name}".encode()).hexdigest()
    query = {"key": score_key}
    existing = collection.find_one(query, {"_id": 0, "score": 1})
    if existing and existing["score"] > score:
        print("User already has a higher score")
        return
    collection.replace_one(query, {"key": score_key, "name": name, "score": score}, upsert=True)
    print("Saved user's score")


def get_top_scores(collection, num):
    """Return top scores as JSON-serializable dictionaries."""

    limit = 0 if num == -1 else num
    docs = collection.find({}, {"_id": 0, "name": 1, "score": 1}).sort("score", -1)
    if limit:
        docs = docs.limit(limit)
    return list(docs)
