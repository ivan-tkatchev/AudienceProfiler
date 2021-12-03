import pandas as pd
import numpy as np
import pickle
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score


def normalization_time_data(data_dict_cur):
    data_dict_cur['hour'] = data_dict_cur['hour'].astype(int)
    data_dict_cur['hour'] = (data_dict_cur['hour'] - np.min(data_dict_cur['hour'])) / np.max(data_dict_cur['hour']) - np.min(data_dict_cur['hour'])
    data_dict_cur['dist_from_msk_in_tz_hours'] = (data_dict_cur['dist_from_msk_in_tz_hours'] - \
                                                      np.min(data_dict_cur['dist_from_msk_in_tz_hours'])) / \
    np.max(data_dict_cur['dist_from_msk_in_tz_hours']) - np.min(data_dict_cur['dist_from_msk_in_tz_hours'])
    return data_dict_cur



def get_train_x_y(cur_data, using_features):
    x = np.empty((cur_data['Segment'].shape[0],len(using_features)-1))
    for i, feature in zip(range(0, 12),['shift', 'oblast', 'city', 'os', 'interp_game',
       'interp_subgame', 'osv_numerical',  'day_of_week',
       'is_weekend', 'hour', 'dist_from_msk_in_tz_hours']):
        x[:,i] = cur_data[feature]

    y = np.array(pd.get_dummies(cur_data['Segment']))
    return x, y


def prepare_model():
    from_file = CatBoostClassifier()
    model = from_file.load_model("model.h5")
    return model


def prepare_data(data):
    data = data.fillna(0)
    data = data.replace('None', 0)
    data['interp_game'] = data['interp_game'].astype(str)
    data['interp_subgame'] = data['interp_subgame'].astype(str)
    data['city'] = data['interp_subgame'].astype(str)
    data['oblast'] = data['oblast'].astype(str)
    data['os'] = data['os'].astype(str)
    data['shift'] = data['shift'].astype(str)

    features_to_encoding = ['interp_game','oblast','interp_subgame','city','os','shift','day_of_week']
    using_features = ['Segment', 'shift', 'oblast', 'city', 'os', 'interp_game',
           'interp_subgame', 'osv_numerical',  'day_of_week',
           'is_weekend', 'hour', 'dist_from_msk_in_tz_hours']
    data_dict = {}
    for feature in features_to_encoding:
        le = LabelEncoder()
        data_dict[feature] = np.array(le.fit_transform(np.array(data[feature])))

    for category in set(using_features) - set(features_to_encoding):
        data_dict[category] = np.array(data[category])
    test_data_dict = normalization_time_data(data_dict)
    x_train, y_train = get_train_x_y(test_data_dict, using_features)
    return x_train, y_train


def predict(data):
    model = prepare_model()
    x_test, y_test = prepare_data(data)
    preds_proba = model.predict_proba(x_test)
    score = roc_auc_score(y_test, preds_proba,  multi_class='ovr', average=None)

    pass