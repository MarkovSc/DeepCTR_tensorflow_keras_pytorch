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
        
    def get_batch(self, X, y, batch_size, index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)
        return X.iloc[start: end] , y.iloc[start:end]

    def fit(self, train, test, y_train, y_test, epoch = 100, batch_size= 1000):
        with tf.Session(config=self.gpu_config) as sess:
            Model = DeepFM(self.sparse_features, self.dense_features, self.sparse_label_dict,
                          self.hidden_layer, self.embed_dim)
            # init variables
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            cnt = int(len(train) / batch_size)
            sys.stdout.flush()
            if self.is_training:
                for i in range(epoch):
                    print('epoch %s:' % i)
                    for j in range(0, cnt):
                        X, y  = self.get_batch(train, y_train, batch_size, j)
                        loss, step = Model.train(sess, X, y)
                        if j % 100 == 0:
                            print('the times of training is %d, and the loss is %s' % (j, loss))
                            Model.save(sess, self.checkpoint_dir)
            else:
                Model.restore(sess, self.checkpoint_dir)
                for j in range(0, cnt):
                    X, y  = get_batch(test, y_test, batch_size, j)
                    result = Model.predict(sess, X)
                    print(result)
