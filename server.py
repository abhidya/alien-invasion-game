from pathlib import Path

from flask import Flask, jsonify, send_from_directory
from pymongo.errors import PyMongoError

from alien_invasion.score_db import connect_and_collect, get_top_scores

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=None)

@app.route('/')
def index():
    return send_from_directory(ROOT, 'index.html')


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(ROOT, path)

@app.route('/get_scores')
def getscore():
    try:
        collection = connect_and_collect()
        return jsonify({"scores": get_top_scores(collection, 10)})
    except PyMongoError as error:
        return jsonify({"scores": [], "error": str(error)}), 503

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3000, debug=True)
