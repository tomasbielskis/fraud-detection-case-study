import pandas as pd
from sklearn.model_selection import train_test_split
import my_resample as ms
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import requests
import pickle
import os
from pymongo import MongoClient


class CleanData(object):
    def __init__(self, train=True):
        self.df = None
        self.X = None
        self.y = None

    def get_df(self, train=True):
        if train:
            self.get_train_data()
        else:
            self.get_new_data()
        return self.df

    def get_train_data(self):
        self.df = pd.read_json('../data/data.json')
        self.create_x_features()
        self.create_y_features()

    def get_new_data(self):
        api_key = os.environ['FRAUD_CASE_STUDY_API_KEY']
        url = 'https://hxobin8em5.execute-api.us-west-2.amazonaws.com/api/'
        sequence_number = 0
        response = requests.post(url, json={'api_key': api_key,
                                 'sequence_number': sequence_number})
        raw_data = response.json()
        self.df = pd.DataFrame.from_dict(raw_data['data'][0], orient='index').T
        self.create_x_features()

    def create_x_features(self):
        self.df['ticket_types_max_profit'] = self.df.ticket_types.apply(self.get_max_profit)
        self.df['ticket_types_num_types'] = self.df.ticket_types.apply(len)
        self.df['ticket_types_num_tickets'] = self.df.ticket_types.apply(self.get_num_tickets)
        self.df['ticket_types_avg_price'] = self.df.ticket_types_max_profit / self.df.ticket_types_num_tickets
        self.df['previous_payouts_num_payouts'] = self.df.previous_payouts.apply(len)
        self.df['previous_payouts_value_payouts'] = self.df.previous_payouts.apply(self.get_value_payouts)
        self.df['previous_payouts_avg_payout'] = self.df.previous_payouts_value_payouts/self.df.previous_payouts_num_payouts
        X_col = ['body_length', 'event_created', 'fb_published',
                 'has_analytics', 'has_header', 'has_logo', 'name_length',
                 'user_age', 'ticket_types_max_profit',
                 'ticket_types_num_types', 'ticket_types_num_tickets',
                 'ticket_types_avg_price', 'previous_payouts_num_payouts',
                 'previous_payouts_value_payouts', 'previous_payouts_avg_payout']
        X = self.df[X_col]
        self.X = X.fillna(X.mean(axis=0))
        # Fill with 0 if no mean
        self.X = self.X.fillna(0)

    def create_y_features(self):
        fraud_acct = {'fraudster_event','fraudster','fraudster_att'}
        self.df['fraud_flag'] = self.df.acct_type.apply(lambda x: x in fraud_acct)
        y_col = ['fraud_flag']
        self.y = self.df[y_col]

    def get_max_profit(self, row):
        return sum(level['cost']*level['quantity_total'] for level in row)

    def get_num_tickets(self, row):
        return sum(level['quantity_total'] for level in row)

    def get_value_payouts(self, row):
        return sum(level['amount'] for level in row)

    def split_data(self, oversample=False, scale=False):
        # train test split
        X_train, X_test, y_train, y_test = train_test_split(self.X.as_matrix(),
                                                            self.y.as_matrix(),
                                                            random_state=17)
        # resample
        if oversample:
            X_train, y_train = ms.oversample(X_train, y_train[:, 0], .5)
        # scale data
        if scale:
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test


class Model(object):
    def __init__(self):
        self.model = None

    def fit(self):
        max_oversample, max_scale, max_mf, max_nest = (True, False, 'log2', 100)
        cd = CleanData()
        df = cd.get_df(train=True)
        X_train, X_test, y_train, y_test = cd.split_data(oversample=max_oversample, scale=max_scale)
        self.model = RandomForestClassifier(max_features=max_mf,
                                            n_estimators=max_nest)
        self.model.fit(X_train, y_train)

    def pickle_model(self):
        with open("model.pkl", 'wb') as f:
            pickle.dump(self.model, f)

    def import_model(self):
        with open("model.pkl", 'rb') as f_un:
            self.model = pickle.load(f_un)
        return self.model


if __name__ == '__main__':
    # Create and Pickle Model
    # model = Model()
    # model.fit()
    # model.pickle_model()

    # New model
    new_model = Model()
    nm = new_model.import_model()

    # Get new data
    cd = CleanData()
    cd.get_df(train=False)
    # print(cd.X)
    # print(nm.predict(cd.X))
    DB_NAME = "fraud"
    COLLECTION_NAME = "events"

    client = MongoClient()
    db = client[DB_NAME]
    coll = db[COLLECTION_NAME]

    cd.df['Fraud'] = nm.predict(cd.X)
    cd.df['Probability'] = nm.predict_proba(cd.X)[0][1]

    # coll.remove({})

    coll.insert(cd.df.to_dict('records'))
    # print(cd.df.to_dict('records'))
