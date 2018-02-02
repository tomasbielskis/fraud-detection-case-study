import random
import pickle
import pandas as pd

class MyModel():
    def fit():
        pass
    def predict():
        return random.choice([True, False])

def get_data(datafile):
    pd.read_json(datafile)
    return X, y

if __name__ == '__main__':
    X, y = get_data('../data/data.json')
    model = MyModel()
    model.fit(X, y)
    with open('dum_mod.pkl', 'w') as f:
        pickle.dump(model, f)
