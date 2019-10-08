from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

def oneHotEncoder(array_1d):
    label = LabelEncoder().fit_transform(array_1d)
    label = label.reshape(len(label), 1)
    one_hot = OneHotEncoder(sparse=False).fit_transform(label)
    return one_hot

def minMaxScale(array_2d):
    return MinMaxScaler().fit_transform(array_2d)

def preprocess(data):
    cat_list =[f for f in data.columns]
    for c in cat_list:
        data[c] = LabelEncoder().fit_transform(list(data[c].values))
    return data

def auc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

def recognize_feature(data, label_encoder = False):
    sparse_features = []
    dense_features = []
    for f in data.columns:
        if data[f].dtype=='object':
            lbl = LabelEncoder()
            lbl.fit(list(data[f].values))
            data[f] = lbl.transform(list(data[f].values))
            sparse_features.append(f)
        elif f.find('cat') >=0 and f.find('bin') <0:
            lbl = LabelEncoder()
            lbl.fit(list(data[f].values))
            data[f] = lbl.transform(list(data[f].values))
            sparse_features.append(f)
        elif data[f].dtype not in ['float16','float32','float64']:
            if(len(data[f].unique()) < 100 and f.find('bin') <0):
                lbl = LabelEncoder()
                lbl.fit(list(data[f].values))
                data[f] = lbl.transform(list(data[f].values))
                sparse_features.append(f)
    print("sparse : unique sum ", sum([len(data[f].unique()) for f in sparse_features]))
        
    dense_features = list(set(data.columns.tolist()) - set(sparse_features))
    return data, sparse_features, dense_features

def hash_encoding(data, sparse_features):
    return ;
def one_hot_for_sparse(data, sparse_features):
    for f in sparse_features:
        one_hot = pd.get_dummies(data[f], prefix =f, dummy_na = True)
        data.drop(f , axis = 1, inplace=True)
        data = data.join(one_hot)
    return data
def scalar_for_dense(data, dense_features):
    for f in dense_features:
        scaler = MinMaxScaler()
        data[f] = scaler.fit_transform(data[f].values.reshape(-1,1))
    return data

def reduce_mem_usage(df, is_first= False):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            if(is_first):
                df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df