from flask import Flask, render_template, jsonify
from alien_invasion.score_db import connect_and_collect, get_top_scores
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_scores')
def getscore():
    my_col = connect_and_collect()
    return jsonify(get_top_scores(my_col, 10))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int("3000"), debug=True)                                                
