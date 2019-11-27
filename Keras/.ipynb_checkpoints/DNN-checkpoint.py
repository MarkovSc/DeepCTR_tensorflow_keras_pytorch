from keras import optimizers
from keras import backend as K
from keras.models import Sequential
from keras.layers import Input,Dense, concatenate,Dropout,BatchNormalization,Activation,Flatten,Add
from keras.layers import RepeatVector, merge, Subtract, Lambda, Multiply, Embedding, Concatenate, Reshape
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.engine.topology import Layer
    
class DNN():
    def __init__(self, sparse_features, dense_features, sparse_label_dict, hidden_layer, embed_dim):
        self.sparse_features = sparse_features
        self.dense_features = dense_features
        self.sparse_label_dict = sparse_label_dict
        self.hidden_layer = hidden_layer
        self.embed_dim = embed_dim
        
    def fit(self, train, y_train, test, y_test, optimizer="adam", loss='binary_crossentropy',
            metrics=["accuracy"], model_summary = False):
        cat_input = []
        cat_output = []        
        for col in self.sparse_features:
            input = Input(shape= (1,))
            cat_input.append(input)
            emb = Embedding(sparse_label_dict[col], self.embed_dim, input_length =1 ,trainable = True)(input)
            cat_output.append(emb)
         
        cat_output = Concatenate(axis=1)(cat_output)
        
        dense_input = Input(shape = (len(self.dense_features), ))  
        dnn_input = Concatenate(axis=1)([Flatten()(cat_output), dense_input])
        dnn_output = dnn_input 
        for layer in self.hidden_layer:
            dnn_output  = BatchNormalization()(dnn_output)
            dnn_output  = Dense(layer, activation='relu')(dnn_output)
            dnn_output  = Dropout(0.2)(dnn_output)
        output = Dense(1, activation='sigmoid')(dnn_output)
        model = Model(inputs = cat_input + [dense_input], outputs=output)
        model.compile(
            optimizer= optimizer,
            loss= loss,
            metrics=metrics
        )
        if model_summary:
            print(model.summary())
        model.fit([train[f] for f in self.sparse_features] + [train[self.dense_features]], y_train, nb_epoch=50, batch_size=1000)
        loss, accuracy = model.evaluate([test[f] for f in self.sparse_features] +  [test[self.dense_features]], y_test)
        print('\n', 'test accuracy:', accuracy)
        y_pred = model.predict([test[f] for f in self.sparse_features] +  [test[self.dense_features]])
        print("auc is ", roc_auc_score(y_test, y_pred))
    def predict(self, pred):
        y_pred = model.predict([pred[f] for f in self.sparse_features] +  [pred[self.dense_features]])
        return y_pred