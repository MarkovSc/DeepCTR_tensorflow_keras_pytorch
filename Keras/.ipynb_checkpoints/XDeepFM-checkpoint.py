from keras import optimizers
from keras import backend as K
from keras.models import Sequential
from keras.layers import Input,Dense, concatenate,Dropout,BatchNormalization,Activation,Flatten,Add, Conv2D, Dot, dot
from keras.layers import RepeatVector, merge, Subtract, Lambda, Multiply, Embedding, Concatenate, Reshape, DepthwiseConv2D
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.engine.topology import Layer
from sklearn.metrics import roc_auc_score, recall_score
    
def CIN(x, conv_layer):
    cat_output_expand = Lambda(lambda x: K.expand_dims(x, axis = 2))(x)

    # shape: -1, 1, feature_size, dim
    x_0 = Lambda(lambda x: K.permute_dimensions(x, (0,3,2,1)))(cat_output_expand)
    x_next = cat_output_expand

    cin_output = []
    for layer in conv_layer:
        x = Lambda(lambda x: K.permute_dimensions(x, (0,3,1,2)))(x_next)
        z_0 = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=(2,3)))([x_0, x])
        z_1 = Lambda(lambda x: K.permute_dimensions(x, (0,2,3,1)))(z_0)

        x_next_list = []
        pooling_output_list = []
        for index in range(layer):
            output = DepthwiseConv2D((int(z_1.shape[1]), int(z_1.shape[2])))(z_1)
            #output = Conv2D(int(cat_output_expand.shape[-1]), (int(z_1.shape[1]), int(z_1.shape[2])) )(z_1)
            output = Lambda(lambda x: K.squeeze(x, 2))(output)
            pooling_output = Lambda(lambda x: K.sum(output, axis = 2))(output)
            pooling_output_list.append(pooling_output)
            x_next_list.append(output)
        x_next = Concatenate(axis = 1)(x_next_list)
        x_next = Lambda(lambda x: K.expand_dims(x, axis = 2))(x_next)

        x_pooling = Concatenate(axis = 1)(pooling_output_list)
        cin_output.append(x_pooling)

    cin_output =  Concatenate(axis = 1)(cin_output)
    cin_output = Dense(1, activation='linear')(cin_output)
    return cin_output
    
class XDeepFM():
    def __init__(self, sparse_features, dense_features, sparse_label_dict, hidden_layer, conv_layer, embed_dim):
        self.sparse_features = sparse_features
        self.dense_features = dense_features
        self.sparse_label_dict = sparse_label_dict
        self.hidden_layer = hidden_layer
        self.embed_dim = embed_dim
        self.sparse_label_dict = sparse_label_dict
        self.conv_layer = conv_layer
        
    def fit(self, train, y_train, n_epoch=100, n_batch = 1000):
        cat_input = []
        cat_output = []
        for col in self.sparse_features:
            input = Input(shape= (1,))
            cat_input.append(input)
            emb = Embedding(self.sparse_label_dict[col], self.embed_dim, input_length =1 ,trainable = True)(input)
            cat_output.append(emb)
         
        cat_output = Concatenate(axis=1)(cat_output)
        
        #first_order = Added_Weights(use_bias = True)(cat_output)
        first_order = Flatten()(cat_output)
        first_order = Dense(1, activation='linear')(first_order)
        
        # 需要使用lambda 层封装Backend 的函数操作
        #first_order = Lambda(lambda x: K.sum(x, axis =1, keepdims=True))(first_order)
        cin_output = CIN(cat_output, self.conv_layer)
        
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
        output  = Add()([first_order, cin_output, dnn_output])
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