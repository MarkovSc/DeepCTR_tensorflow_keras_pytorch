"""
Created on Jan 01, 2020
@author: markov_alg@163.com

Tensorflow implementation of DeepFM [1]
Reference:
[1] DeepFM: A Factorization-Machine based Neural Network for CTR Prediction,
    Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.
"""

from keras import optimizers
from keras import backend as K
from keras.models import Sequential
from keras.layers import Input,Dense, concatenate,Dropout,BatchNormalization,Activation,Flatten,Add
from keras.layers import RepeatVector, Subtract, Lambda, Multiply, Embedding, Concatenate, Reshape
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
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
    
class DeepFM():
    def __init__(self, sparse_features, dense_features, hidden_layer=[128,256,128], embed_dim=8, sparse_label_dict=None):
        self.sparse_features = sparse_features
        self.dense_features = dense_features
        self.sparse_label_dict = sparse_label_dict
        self.hidden_layer = hidden_layer
        self.embed_dim = embed_dim
        self.voc_dim = sparse_label_dict if type(sparse_label_dict) == type(3) else 1000
        if type(sparse_label_dict) != type([1]):
            self.sparse_label_dict = dict([(col,100) for col in self.sparse_features])
        else:
            self.sparse_label_dict = dict([(col,self.voc_dim) for col in self.sparse_features])    
        
    def build(self):
        num_output = []
        cat_output = []
        input_dict = {}
        for col in self.sparse_features:
            input = Input(shape= (1,), dtype=tf.string)
            input_dict[col] = input
            cat_tensor = tf.keras.layers.experimental.preprocessing.Hashing(num_bins= (self.sparse_label_dict[col] + 1))(input)
            emb = Embedding(self.sparse_label_dict[col]+1, output_dim=self.embed_dim, input_length =1 ,trainable = True)(cat_tensor)
            cat_output.append(emb)
            
        for col in self.dense_features:
            input = Input(shape= (1,), dtype='float64')
            input_dict[col] = input
            num_output.append(input)     
         
        cat_output = Concatenate(axis=1)(cat_output)
        num_output = Concatenate(axis=1)(num_output)
        
        first_order = Added_Weights(use_bias = True)(cat_output)
        first_order = Flatten()(first_order)
        
        # 需要使用lambda 层封装Backend 的函数操作
        first_order = Lambda(lambda x: K.sum(x, axis =1, keepdims=True))(first_order)
        
        # cat_output shape : s *k, keras 需要把这个list 进行concat 为一个tensor
        # 然后fatten 为一个weight，然后在sum，或者是直接sum, w * x ,w 是tf.variable
        
        # second order for sparse features with fixed dim
        # vx * vx - vx, vx shape: (1, k)
        vx = Added_Weights()(cat_output)
        sum_square = Lambda(lambda x: K.sum(x, axis =1))(vx)
        sum_square = Multiply()([sum_square, sum_square])
        square_sum = Multiply()([vx, vx])
        square_sum = Lambda(lambda x: K.sum(x, axis =1))(square_sum)
        second_order = Subtract()([sum_square, square_sum])
        second_order = Lambda(lambda x: K.sum(x/2, axis =1, keepdims=True))(second_order)
        print(second_order.shape)
        
        dnn_input = Concatenate(axis=1)([Flatten()(cat_output), num_output])
        dnn_output = dnn_input 
        for layer in self.hidden_layer:
            dnn_output  = BatchNormalization()(dnn_output)
            dnn_output  = Dense(layer, activation='relu')(dnn_output)
            dnn_output  = Dropout(0.2)(dnn_output)
        dnn_output = Dense(1, activation='linear')(dnn_output)
        
        #output  = Concatenate(axis=1)([first_order, second_order, dnn_output])
        output  = Add()([first_order, second_order, dnn_output])
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
        return model
        
    def fit(self, train, y_train, n_epoch=10, n_batch = 1000):
        self.model = self.build()
        #print(model.summary())
        self.model.fit(dict([(f, train[f]) for f in self.sparse_features + self.dense_features]), y_train, epochs = n_epoch, batch_size= n_batch)
        
    def predict(self, test):
        y_pred = self.model.predict(dict([(f, test[f]) for f in self.sparse_features + self.dense_features]))
        return y_pred
    
    def evaluate(self, test, y_test, metrics):
        #loss, accuracy = self.model.evaluate([test[f] for f in self.sparse_features] +  [test[self.dense_features]], y_test)
        #print('\n', 'test accuracy:', accuracy)
        y_pred = self.predict(test)
        return metrics(y_test, y_pred)
