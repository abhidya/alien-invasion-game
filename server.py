from flask import Flask
from alien_invasion.score_db import connect_and_collect, get_top_scores
app = Flask(__name__)
from flask import jsonify


@app.route('/')
def getscore():
    my_col = connect_and_collect()
    return jsonify(get_top_scores(my_col, 10))


if __name__ == '__main__':
    app.run()
