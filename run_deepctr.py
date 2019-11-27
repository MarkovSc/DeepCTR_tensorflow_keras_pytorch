import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.models import DeepFM, xDeepFM, DCN
from deepctr.inputs import  SparseFeat, DenseFeat,get_fixlen_feature_names

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

from deepctr.models import DeepFM
from deepctr.inputs import  SparseFeat, DenseFeat,get_fixlen_feature_names

def recognize_feature(data, label_encoder = False):
    sparse_features = []
    dense_features = []
    for f in data.columns:
        if data[f].dtype=='object':
            sparse_features.append(f)
        elif f.find('cat') >=0 and f.find('bin') <0:
            sparse_features.append(f)
        elif data[f].dtype not in ['float16','float32','float64']:
            if(len(data[f].unique()) < 100 and f.find('bin') <0):
                sparse_features.append(f)     
    dense_features = list(set(data.columns.tolist()) - set(sparse_features))
    return data, sparse_features, dense_features

def hash_encoding(data, sparse_features):
    return ;
def one_hot_for_sparse(data, sparse_features):
    for f in sparse_features:
        one_hot = pd.get_dummies(data[f], prefix =f, dummy_na = True)
        data.drop(f , axis = 1, inplace=True)
        data = data.join(one_hot)
    return data
def scalar_for_dense(data, dense_features):
    for f in dense_features:
        scaler = MinMaxScaler()
        data[f] = scaler.fit_transform(data[f].values.reshape(-1,1))
    return data

if __name__ == "__main__":
    data = pd.read_csv("Data/train.csv.gz")
    data = data.set_index("id")
    #target = data['target']
    #data.drop(['target'], axis=1, inplace=True)
    data, sparse_features, dense_features = recognize_feature(data)
    sparse_features = [f for f in sparse_features if f !='target']
    dense_features = [f for f in dense_features if f !='target']
    
    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['target']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                           for feat in sparse_features] + [DenseFeat(feat, 1,)
                          for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns
    fixlen_feature_names = get_fixlen_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.33)
    train_model_input = [train[name] for name in fixlen_feature_names]

    test_model_input = [test[name] for name in fixlen_feature_names]

    # 4.Define Model,train,predict and evaluate
    model = DeepFM(linear_feature_columns, dnn_feature_columns, embedding_size=3, dnn_hidden_units=(2048, 1024, 100), dnn_use_bn=False, task='binary')
    model.compile("adam", "binary_crossentropy",
                  metrics=['accuracy'], )

    history = model.fit(train_model_input, train[target].values,
                        batch_size=1000, epochs=100, verbose=2, validation_split=0.2, )
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
