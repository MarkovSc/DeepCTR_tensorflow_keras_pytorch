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
from .TFModel import TFModel

class XDeepFM(TFModel):
    def __init__(self, sparse_features, dense_features, sparse_label_dict, hidden_layer, embed_dim):
        super().__init__(sparse_features, dense_features, sparse_label_dict, hidden_layer, embed_dim)
        
        self.sparse_features = sparse_features
        self.dense_features = dense_features
        self.sparse_label_dict = sparse_label_dict
        self.hidden_layer = hidden_layer
        self.embed_dim = embed_dim
        self.embed_feature_size = len(self.sparse_features)
        self.weights=dict()
        self.conv_layer = [25, 20, 20]
        self.build_model()
        
    
    def build_model(self, opt = 'adam'):
        self.sparse_input_list = []
        self.dense_input = tf.placeholder(tf.float32, shape= [None, len(self.dense_features)], name = "dense_input")
        self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")
        self.keep_prob = tf.placeholder(tf.float32)
        
        cat_output = []
        for col in self.sparse_features:
            input = tf.placeholder(tf.int32, shape= [None, 1], name = "sparse_input")
            self.sparse_input_list.append(input)
            weights = tf.Variable(tf.random_normal([self.sparse_label_dict[col], self.embed_dim], 0.0, 0.01),
                                               name = "feature_embedding" + col)
            emb = tf.nn.embedding_lookup(weights, input)
            cat_output.append(emb)

        sparse_embed = tf.concat(cat_output, axis =1)

        emb_flat = tf.layers.flatten(sparse_embed)
        first_order = tf.layers.dense(emb_flat, 1, activation=tf.nn.relu, use_bias=True)
        
        cat_output_expand = tf.expand_dims(sparse_embed, axis = 2)

        # shape: -1, 1, feature_size, dim
        print(cat_output_expand.shape)
        x_0 = tf.transpose(cat_output_expand, perm=[0,3,2,1])
        x_next = cat_output_expand
    
        cin_output = []
        for layer in self.conv_layer:
            x = tf.transpose(x_next, perm=[0,3,1,2])
            z_0 = tf.matmul(x, x_0)
            print("dot shape", z_0.shape)
            z_1 = tf.transpose(z_0, perm=[0,2,3,1])
            
            x_next_list = []
            pooling_output_list = []
            for index in range(layer):
                output = tf.layers.conv2d(z_1, cat_output_expand.shape[-1], (int(z_1.shape[1]), int(z_1.shape[2])))
                print("conv shape", output.shape)
                output = tf.squeeze(output, 2)
                pooling_output = tf.reduce_sum(output, axis = 2)
                pooling_output_list.append(pooling_output)
                x_next_list.append(output)
            x_next = tf.concat(x_next_list, axis = 1)
            x_next = tf.expand_dims(x_next, axis = 2)
            
            x_pooling = tf.concat(pooling_output_list, axis = 1)
            cin_output.append(x_pooling)
        
        cin_output =  tf.concat(cin_output, axis = 1)
        cin_output = tf.layers.dense(cin_output, 1, activation=None, use_bias=True)

        deep_input = tf.concat([emb_flat, self.dense_input], axis =1)
        deep_output = deep_input
        for index, layer in enumerate(self.hidden_layer):
            deep_output = tf.layers.batch_normalization(deep_output)
            deep_output = tf.layers.dense(deep_output, layer, activation=tf.nn.relu, use_bias=True)
            deep_output = tf.layers.dropout(deep_output, self.keep_prob)

        deep_output = tf.layers.dense(deep_output, 1, activation=tf.nn.relu, use_bias=True)
        deep = deep_output

        concat_input = tf.concat([first_order, cin_output, deep], axis=1)
        self.out = tf.layers.dense(concat_input, 1, activation=tf.nn.sigmoid, use_bias=True)


        self.loss = -tf.reduce_mean(
            self.label * tf.log(self.out + 1e-24) + (1 - self.label) * tf.log(1 - self.out + 1e-24))

        self.global_step = tf.Variable(0, trainable=False)
        
        if opt == 'adam':
            self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(0.001)
            trainable_params = tf.trainable_variables()
            print(trainable_params)
            gradients = tf.gradients(self.loss, trainable_params)
            clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
            self.train_op = self.optimizer.apply_gradients(
                zip(clip_gradients, trainable_params), global_step=self.global_step)
    
    def train(self, sess, train, y_train, drop_out = 0.2):
        feed_dict = dict()
        for index, col in enumerate(self.sparse_features):
            feed_dict[self.sparse_input_list[index]] = train[[col]].values
        feed_dict[self.dense_input] = train[self.dense_features].values
        feed_dict[self.label] = y_train.values
        feed_dict[self.keep_prob] = drop_out
        
        loss, _, step = sess.run([self.loss, self.train_op, self.global_step], feed_dict= feed_dict)
        return loss, step

    def to_predict(self, sess, test, drop_out = 0.2):
        feed_dict = dict()
        for index, col in enumerate(self.sparse_features):
            feed_dict[self.sparse_input_list[index]] = test[[col]].values
        feed_dict[self.dense_input] = test[self.dense_features].values
        feed_dict[self.keep_prob] = drop_out
        
        result = sess.run([self.out], feed_dict = feed_dict)
        return result

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)