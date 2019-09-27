from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def recognize_feature(data):
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