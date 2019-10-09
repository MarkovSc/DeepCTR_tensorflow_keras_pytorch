import numpy as np
import tensorflow as tf
import sys
from Tensorflow.DeepFM import *

class TFModel():
    def __init__(self, sparse_features, dense_features, sparse_label_dict, hidden_layer, embed_dim):
        self.gpu_config = tf.ConfigProto()
        self.gpu_config.gpu_options.allow_growth = True
        self.sparse_features = sparse_features
        self.dense_features = dense_features
        self.sparse_label_dict = sparse_label_dict
        self.hidden_layer = hidden_layer
        self.embed_dim = embed_dim
        self.checkpoint_dir = "./"
        self.is_training = True
        self.sess = tf.Session(config=self.gpu_config)
        
    def get_batch(self, X, y, batch_size, index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(X) else len(X)
        if len(y) == 0:
            return X.iloc[start: end]
        else:
            return X.iloc[start: end] , y.iloc[start:end]

    def fit(self, train, y_train, epoch = 100, batch_size= 1000):
        self.batch_size = batch_size
        self.Model = DeepFM(self.sparse_features, self.dense_features, self.sparse_label_dict,
                      self.hidden_layer, self.embed_dim)
        # init variables
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        cnt = int(len(train) / batch_size)
        sys.stdout.flush()
        if self.is_training:
            for i in range(epoch):
                print('epoch %s:' % i)
                for j in range(0, cnt):
                    X, y  = self.get_batch(train, y_train, batch_size, j)
                    loss, step = self.Model.train(self.sess, X, y)
                    if j % 100 == 0:
                        print('the times of training is %d, and the loss is %s' % (j, loss))
                        
    def predict(self, test):
        cnt = int(len(test) / self.batch_size) + 1
        result = []
        for j in range(0, cnt):
            X  = self.get_batch(test, [], self.batch_size, j)
            result += self.Model.predict(self.sess, X)
            
        return np.concatenate(result).reshape(-1,)
        
    def evaluate(self, test, y_test, metrics):
        pred = self.predict(test)
        return metrics(y_test, pred)
         
        