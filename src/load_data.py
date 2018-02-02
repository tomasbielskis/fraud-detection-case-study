import pandas as pd
from sklearn.model_selection import train_test_split
import my_resample as ms
from sklearn.preprocessing import StandardScaler

def get_max_profit(row):
    return sum(level['cost']*level['quantity_total'] for level in row)


def get_num_tickets(row):
    return sum(level['quantity_total'] for level in row)


def get_value_payouts(row):
    return sum(level['amount'] for level in row)


def load_data_frame(json_file):
    # df = json_file
    df = pd.read_json(json_file)
    fraud_acct = {'fraudster_event','fraudster','fraudster_att'}
    df['fraud_flag'] = df.acct_type.apply(lambda x: x in fraud_acct)
    df['ticket_types_max_profit'] = df.ticket_types.apply(get_max_profit)
    df['ticket_types_num_types'] = df.ticket_types.apply(len)
    df['ticket_types_num_tickets'] = df.ticket_types.apply(get_num_tickets)
    df['ticket_types_avg_price'] = df.ticket_types_max_profit / df.ticket_types_num_tickets
    df['previous_payouts_num_payouts'] = df.previous_payouts.apply(len)
    df['previous_payouts_value_payouts'] = df.previous_payouts.apply(get_value_payouts)
    df['previous_payouts_avg_payout'] = df.previous_payouts_value_payouts/df.previous_payouts_num_payouts
    return df


def split_data(f_name='../data/data.json', oversample=False, scale=False, for_predict=False):
    df = load_data_frame(f_name)
    X_col = ['body_length','event_created','fb_published','has_analytics','has_header',
             'has_logo','name_length','user_age','ticket_types_max_profit',
             'ticket_types_num_types','ticket_types_num_tickets','ticket_types_avg_price',
             'previous_payouts_num_payouts','previous_payouts_value_payouts','previous_payouts_avg_payout']
    y_col = ['fraud_flag']
    X = df[X_col]
    X = X.fillna(X.mean(axis=0))
    if for_predict:
        return X
    y = df[y_col]
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X.as_matrix(),
                                                        y.as_matrix(),
                                                        random_state=17)
    # resample
    if oversample:
        X_train, y_train = ms.oversample(X_train, y_train[:,0], .5)
    # scale data
    if scale:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = split_data(oversample=False, scale=False)
