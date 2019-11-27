# this is for feature encoding
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

def recognize_feature(data, label_encoder = False):
    sparse_features = []
    dense_features = []
    for f in data.columns:
        if data[f].dtype=='object':
            lbl = LabelEncoder()
            lbl.fit(list(data[f].values))
            data[f] = lbl.transform(list(data[f].values))
            sparse_features.append(f)
        elif f.find('cat') >=0 and f.find('bin') <0:
            lbl = LabelEncoder()
            lbl.fit(list(data[f].values))
            data[f] = lbl.transform(list(data[f].values))
            sparse_features.append(f)
        elif data[f].dtype not in ['float16','float32','float64']:
            if(len(data[f].unique()) < 100 and f.find('bin') <0):
                lbl = LabelEncoder()
                lbl.fit(list(data[f].values))
                data[f] = lbl.transform(list(data[f].values))
                sparse_features.append(f)
    print("sparse : unique sum ", sum([len(data[f].unique()) for f in sparse_features]))
        
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