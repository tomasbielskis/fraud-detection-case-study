from sklearn.ensemble import RandomForestClassifier
from load_data import split_data
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from sklearn.grid_search import GridSearchCV


def display_importances_trees():
   # show feature importances
   feature_df = pd.DataFrame([np.array(X_col), model.feature_importances_]).T
   feature_df.columns = ['feature','value']
   return feature_df.sort_values('value', ascending=False)


def display_metrics():
   print("\nMETRICS")
   print("Model recall: {}".format(recall_score(y_test, y_pred)))
   print("Model precision: {}".format(precision_score(y_test, y_pred)))
   print("Model accuracy: {}".format(model.score(X_test, y_test)))

   print ("\nCONFUSION MATRIX")
   print (confusion_matrix(y_test, y_pred))
   print ("\nkey:")
   print (" TN   FP ")
   print (" FN   TP ")


def fit_model(X_train, X_test, y_train, y_test,
              max_features='auto', n_estimators=100):
    model = RandomForestClassifier(max_features=max_features,
                                   n_estimators=n_estimators)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print("max_features={}, n_estimators={}".format(max_features, n_estimators))
    print("F1 = {}".format(f1))
    return f1


def grid_search(max_features_list, n_estimators_list):
    max_f1 = 0
    max_oversample = None
    max_scale = None
    max_mf = None
    max_nest = None
    for oversample in [True, False]:
        for scale in [True, False]:
            print("Oversample: {}, Scale: {}".format(oversample, scale))
            X_train, X_test, y_train, y_test = split_data(oversample=oversample,
                                                          scale=scale)
            for mf in max_features_list:
                for nest in n_estimators_list:
                    f1 = fit_model(X_train, X_test, y_train, y_test,
                                   max_features=mf,
                                   n_estimators=nest)
                    if f1 > max_f1:
                        print("NEW MAX!")
                        max_f1 = f1
                        max_oversample = oversample
                        max_scale = scale
                        max_mf = mf
                        max_nest = nest
    return max_f1, max_oversample, max_scale, max_mf, max_nest

if __name__ == '__main__':
    X_col = ['body_length','event_created','fb_published','has_analytics','has_header',
             'has_logo','name_length','user_age','ticket_types_max_profit',
             'ticket_types_num_types','ticket_types_num_tickets','ticket_types_avg_price',
             'previous_payouts_num_payouts','previous_payouts_value_payouts','previous_payouts_avg_payout']

    max_features_list = ['auto', 'log2']
    n_estimators_list = [100, 500, 1000]
    max_f1, max_oversample, max_scale, max_mf, max_nest = grid_search(max_features_list, n_estimators_list)
    print(max_f1, max_oversample, max_scale, max_mf, max_nest)
