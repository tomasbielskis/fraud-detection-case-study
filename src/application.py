import pandas as pd
import numpy as np
import pickle
import os
from flask import Flask, request, render_template
import requests

application = Flask(__name__)

def get_new_data():
    api_key = 'vYm9mTUuspeyAWH1v-acfoTlck-tCxwTw9YfCynC'
    url = 'https://hxobin8em5.execute-api.us-west-2.amazonaws.com/api/'
    sequence_number = 0
    response = requests.post(url, json={'api_key': api_key,
                                'sequence_number': sequence_number})
    raw_data = response.json()
    return raw_data

# home page
@application.route('/')
def index():
    return render_template('index.html', title='Fraud Detector')

# Form page to submit text
@application.route('/submission_page/')
def submission_page():
    return '''
        <form action="/post_recommender" method='POST' >
            <input type="text" name="user_input1" />
            <input type="submit" />
        </form>
        '''

@application.route('/predicted_fraud', methods=['POST'] )
def post_recommender():
    # page = 'These are some events that might be fraudulent: {0}'
    return render_template('predict.html',predictions=predictions)

if __name__ == '__main__':

    # model = pickle.load(open('path to model', 'rb'))

    # check database
    # get_new_data()
    application.run(host='0.0.0.0', port=8080, debug=True)
