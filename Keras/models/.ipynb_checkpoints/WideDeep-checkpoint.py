#coding=utf-8
from sklearn.metrics import roc_auc_score
from keras.models import Sequential
from keras.layers import Input,Dense, concatenate
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class WideAndDeep():
   def __init__(self, train, valid):
        self.train = train
        self.valid = valid


   def fit(self, train, valid):
   wide = Sequential()
   wide = Input(shape=(X_train.shape[1],))
   
   # deep
   deep_data = Input(shape=(X_train.shape[1],))
   deep = Dense(input_dim=X_train.shape[1], output_dim=256, activation='relu')(deep_data)
   deep = Dense(128, activation='relu')(deep)
   
   # wide & deep 
   #wide_deep = concatenate([wide, deep])
   wide_deep = deep
   wide_deep = Dense(1, activation='sigmoid')(wide_deep)
   model = Model(inputs=[wide, deep_data], outputs=wide_deep)
    
   print("---starting the training---")
   model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy']
   )
    
   model.fit([X_train, X_train], y_train, nb_epoch=10, batch_size=32)
    
   loss, accuracy = model.evaluate([X_test, X_test], y_test)
   print('\n', 'test accuracy:', accuracy)
   y_pred = model.predict([X_test,X_test])
   print("auc is ", roc_auc_score(y_test, y_pred))
