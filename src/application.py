from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import pandas as pd
import numpy as np
import pickle
import os
from flask import Flask, request, render_template
import requests
from load_data import split_data


application = Flask(__name__)

def get_new_data():
    api_key = 'vYm9mTUuspeyAWH1v-acfoTlck-tCxwTw9YfCynC'
    url = 'https://hxobin8em5.execute-api.us-west-2.amazonaws.com/api/'
    sequence_number = 0
    response = requests.post(url, json={'api_key': api_key,
                                'sequence_number': sequence_number})
    raw_data = response.json()
    return pd.DataFrame.from_dict(raw_data['data'][0],orient='index')

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
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f, encoding='latin1')

    # check database
    new_events = get_new_data()
    X = split_data(new_events, for_predict=True)
    predictions = model.predict(X)
    # predictions = {'event1':'probability1'}
    application.run(host='0.0.0.0', port=8080, debug=True)
