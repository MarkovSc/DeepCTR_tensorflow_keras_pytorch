import numpy as np
import tensorflow as tf
import sys

class TFBase():
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

        deep_input = tf.concat([emb_flat, self.dense_input], axis =1)
        deep_output = deep_input
        for index, layer in enumerate(self.hidden_layer):
            deep_output = tf.layers.batch_normalization(deep_output)
            deep_output = tf.layers.dense(deep_output, layer, activation=tf.nn.relu, use_bias=True)
            deep_output = tf.layers.dropout(deep_output, self.keep_prob)

        deep_output = tf.layers.dense(deep_output, 1, activation=tf.nn.relu, use_bias=True)
        deep = deep_output
        self.out = tf.layers.dense(deep, 1, activation=tf.nn.sigmoid, use_bias=True)

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
                    loss, step = self.train(self.sess, X, y)
                    if j % 100 == 0:
                        print('the times of training is %d, and the loss is %s' % (j, loss))
                        
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
    
    def predict(self, test):
        cnt = int(len(test) / self.batch_size) + 1
        result = []
        for j in range(0, cnt):
            X  = self.get_batch(test, [], self.batch_size, j)
            result += self.to_predict(self.sess, X)
            
        return np.concatenate(result).reshape(-1,)
        
    def evaluate(self, test, y_test, metrics):
        pred = self.predict(test)
        return metrics(y_test, pred)
        