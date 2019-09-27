#coding=utf-8
from sklearn.metrics import roc_auc_score
from keras.models import Sequential
from keras.layers import Input,Dense, concatenate
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def load_data(file):
    df = pd.read_csv("../Data/train.csv.gz")
    return df

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

def eval_matric(y_true, y_prob):
    print(sum(y_true)/ len(y_true))
    print(sum([i>0.5 for i in y_prob])/ len(y_true))

    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
        gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    print("gini:", gini)
    return gini


def main():
    print("---loading and preprocessing the data---")
    data = load_data('./train')
    data = data.set_index("id")
    target = data['target']
    data.drop(['target'], axis=1, inplace=True)
    data = preprocess(data)
    train, test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)
    print("---build the wide&deep model---")
    # wide
    wide = Sequential()
    wide = Input(shape=(train.shape[1],))

    # deep
    deep_data = Input(shape=(train.shape[1],))
    deep = Dense(1024, activation='relu')(deep_data)
    deep = Dense(128, activation='relu')(deep)
    deep = Dense(64, activation='relu')(deep)

    # wide & deep 
    #wide_deep = concatenate([wide, deep])
    wide_deep = deep
   # wide_deep = deep
    wide_deep = Dense(1, activation='sigmoid')(wide_deep)
    model = Model(inputs=[wide, deep_data], outputs=wide_deep)

    print("---starting the training---")
    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit([train, train], y_train, nb_epoch=1, batch_size=32)

    loss, accuracy = model.evaluate([test, test], y_test)
    print('\n', 'test accuracy:', accuracy)
    y_pred = model.predict([test,test])
    print(sum(y_test))
    print(len(y_test))
    print("auc is ", roc_auc_score(y_test, y_pred))
    eval_matric(y_test, y_pred)

if __name__ == '__main__':
    main()

