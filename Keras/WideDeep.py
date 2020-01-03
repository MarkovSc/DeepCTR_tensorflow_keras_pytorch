"""
Created on Jan 01, 2020
@author: markov_alg@163.com

Tensorflow implementation of WideDeep [1]
Reference: 
[1] Wide & deep learning for recommender systems,
    Cheng, Heng-Tze, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson et al.
"""

from keras import optimizers
from keras import backend as K
from keras.models import Sequential
from keras.layers import Input,Dense, concatenate,Dropout,BatchNormalization,Activation,Flatten,Add
from keras.layers import RepeatVector, merge, Subtract, Lambda, Multiply, Embedding, Concatenate, Reshape
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.engine.topology import Layer
from sklearn.metrics import roc_auc_score, recall_score

class Added_Weights(Layer):
    def __init__(self, use_bias = False, **kwargs):
        self.use_bias = use_bias
        
        super(Added_Weights, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], input_shape[2]),
                                      initializer='uniform',  # TODO: Choose your initializer
                                      trainable=True)
        
        if(self.use_bias):
            self.bias = self.add_weight(name='bias',
                                        shape=(1, input_shape[2]),
                                        initializer='uniform',  # TODO: Choose your initializer
                                        trainable=True)
        else:
            self.bias = self.add_weight(name='bias',
                                        shape=(1, input_shape[2]),
                                        initializer='zeros',
                                        trainable=False)
        
        super(Added_Weights, self).build(input_shape)

    def call(self, x, **kwargs):
        # Implicit broadcasting occurs here.
        # Shape x: (BATCH_SIZE, N, M)
        # Shape kernel: (N, M)
        # Shape output: (BATCH_SIZE, N, M)
        #if self.use_bias:
        #if self.use_bias:
        return x * self.kernel + self.bias
        

    def compute_output_shape(self, input_shape):
        return input_shape
    
class WideAndDeep():
    def __init__(self, sparse_features, dense_features, sparse_label_dict, hidden_layer, embed_dim):
        self.sparse_features = sparse_features
        self.dense_features = dense_features
        self.sparse_label_dict = sparse_label_dict
        self.hidden_layer = hidden_layer
        self.embed_dim = embed_dim
        self.sparse_label_dict = sparse_label_dict
        
    def fit(self, train, y_train, n_epoch=100, n_batch = 1000):
        cat_input = []
        cat_output = []
        for col in self.sparse_features:
            input = Input(shape= (1,))
            cat_input.append(input)
            emb = Embedding(self.sparse_label_dict[col], self.embed_dim, input_length =1 ,trainable = True)(input)
            cat_output.append(emb)
         
        cat_output = Concatenate(axis=1)(cat_output)
        
        first_order = Added_Weights(use_bias = True)(cat_output)
        first_order = Flatten()(first_order)
        
        # 需要使用lambda 层封装Backend 的函数操作
        first_order = Lambda(lambda x: K.sum(x, axis =1, keepdims=True))(first_order)
        
        # cat_output shape : s *k, keras 需要把这个list 进行concat 为一个tensor
        # 然后fatten 为一个weight，然后在sum，或者是直接sum, w * x ,w 是tf.variable
        
        # second order for sparse features with fixed dim
        # vx * vx - vx, vx shape: (1, k)
        
        dense_input = Input(shape = (len(self.dense_features), ))
        
        dnn_input = Concatenate(axis=1)([Flatten()(cat_output), dense_input])
        #dnn_input = dense_input 
        dnn_output = dnn_input 
        for layer in self.hidden_layer:
            dnn_output  = BatchNormalization()(dnn_output)
            dnn_output  = Dense(layer, activation='relu')(dnn_output)
            dnn_output  = Dropout(0.2)(dnn_output)
        dnn_output = Dense(1, activation='linear')(dnn_output)
        
        #output  = Concatenate(axis=1)([first_order, second_order, dnn_output])
        output  = Add()([first_order, dnn_output])
        output = Dense(1, activation='sigmoid')(output)
        model = Model(inputs = cat_input + [dense_input], outputs=output)
        print("---starting the training---")
        model.compile(
            optimizer="adam",
            loss='binary_crossentropy',
            metrics=["accuracy"]
        )
        #print(model.summary())
        model.fit([train[f] for f in self.sparse_features] + [train[self.dense_features]], y_train, epochs = n_epoch, batch_size= n_batch)
        self.model = model
        
    def predict(self, test):
        y_pred = self.model.predict([test[f] for f in self.sparse_features] +  [test[self.dense_features]])
        return y_pred
    
    def evaluate(self, test, y_test, metrics):
        #loss, accuracy = self.model.evaluate([test[f] for f in self.sparse_features] +  [test[self.dense_features]], y_test)
        #print('\n', 'test accuracy:', accuracy)
        y_pred = self.predict(test)
        return metrics(y_test, y_pred)