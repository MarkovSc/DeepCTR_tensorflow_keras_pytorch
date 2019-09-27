"""
Tensorflow implementation of DeepFM [1]
Reference:
[1] DeepFM: A Factorization-Machine based Neural Network for CTR Prediction,
    Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.
"""

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from time import time
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from yellowfin import YFOptimizer

class DeepFM():
    def __init__(self, sparse_features, dense_features, sparse_label_dict, hidden_layer, embed_dim):
        self.sparse_features = sparse_features
        self.dense_features = dense_features
        self.sparse_label_dict = sparse_label_dict
        self.hidden_layer = hidden_layer
        self.embed_dim = embed_dim
        self.embed_feature_size = len(self.sparse_features)
    
    def init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            self.sparse_input = tf.placeholder(tf.int32, shape= [None, None], name = "sparse_input")
            self.dense_input = tf.placeholder(tf.int32, shape= [None, None], name = "dense_input")
            self.label = tf.placeholder(tf.int32, shape=[None, 1], name="label")
            
            self.weights = init_weights()
                  
            sparse_embed = tf.nn.embedding_look_up(self.weights["feature_embedding"], self.sparse_input)
            first_order = tf.reduce_sum(tf.multiply(sparse_embed, self.weights["first_order"]))
            
            vx = tf.multiply(self.sparse_embed, tf.weights["second_order"])
            sum_square = tf.reduce_sum(vx, 1)
            sum_square =  tf.multiply(sum_square, sum_square)
            sum_square = tf.reduce_sum(sum_square, 2, keepdims = True)
            
            square_sum = tf.multiply(vx, vx)
            square_sum = tf.reduce.sum(square_sum, 1)
            square_sum = tf.reduce.sum(square_sum, 2, keepdims = True)
            second_order  = 0.5 * tf.subtract(sum_square , square_sum)
            
            deep_input = tf.flatten(sparse_embed, 2)
            deep_input = tf.concatten([deep_input, self.dense_input])
            deep_output = deep_input
            for index, layer in enumerate(hidden_layer):
                deep_output = tf.mat(self.weights["layer" + index],  deep_output)
            
            deep_output = tf.mat(self.weights["last_layer"],  deep_output)
            deep = deep_output
            
            concat_input = tf.concat([first_order, second_order, deep], axis=1)
            self.out = tf.add(tf.matmul(concat_input, self.weights["concat_projection"]), self.weights["concat_bias"])
            self.out = tf.nn.sigmoid(self.out)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run()
    
    def init_weight(self):
        weights = dict()
        weights["feature_embedding"] = tf.Variable(tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01),
                                                   name = "feature_embedding")
        weights["first_order"] = tf.Variable(tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01),
                                             name = "first_order")
        weights["feature_bias"] = tf.Variable(tf.random_uniform([self.feature_size, 1], 0.0, 1.0), name="feature_bias")
            
        weights["second_order"] = tf.Variable(tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01),
                                              name = "second_order")
        
        weights["concat_projection"] = tf.Variable(tf.random_normal([3], 0.0, 0.01),
                                              name = "concat_projection", dtype=np.float32))
        weights["concat_bias"] = tf.Variable(tf.constant(0.01), dtype=np.float32)
        
        flatten_size = len(self.sparse_feature) * embedding_size + len(self.dense_feature)
        last_layer = flatten_size
        for index, layer in enumerate(hidden_layer):
            weights["layer" + index] = tf.Variable(tf.random_normal([last_layer, layer], 0.0, 0.01),
                                                   name = "layer" + index)
            last_layer = layer
            
        weights["last_layer"] = tf.Variable(tf.random_normal([hidden_layer[-1], 1], 0.0, 0.01),
                                            name = "last_layer")
        
    def fit(self, train, test, y_train, y_test):