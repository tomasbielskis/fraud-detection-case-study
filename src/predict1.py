import pickle
from load_data import split_data


def import_model():
    with open("model.pkl", 'rb') as f_un:
        model = pickle.load(f_un)
    return model


if __name__ == '__main__':
    model = import_model()
    X = split_data(f_name='../data/data.json', for_predict=True):
    model.predict(X)
