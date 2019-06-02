import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

trainFile = pd.read_csv('./train-v3.csv')
validFile = pd.read_csv('./valid-v3.csv')

trainFile = trainFile.append(validFile)

prediction_target_name = 'price'

X_train = trainFile

X_scaler = preprocessing.MinMaxScaler()

X_train_value = X_scaler.fit_transform(X_train.values)
Y_train_value = pd.DataFrame(X_scaler.fit_transform(X_train.values), columns=trainFile.columns)[prediction_target_name].values.reshape(-1,1)

std = pd.DataFrame(columns=trainFile.columns)
std.loc[0] = [0 for n in range(23)]

for i in range(0,X_train_value.shape[1],1):
    if X_train.columns[i] == prediction_target_name:
        continue
    f = plt.figure(figsize=(4,4));
    # Plot training & validation loss values
    plt.plot(Y_train_value)
    plt.plot(X_train_value[:,i])
    std.loc[0][X_train.columns[i]] = np.array(Y_train_value-np.array([X_train_value[:,i]]).T).std()
    #plt.plot(history.history['val_loss'])
    plt.title('relation')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.legend([prediction_target_name, X_train.columns[i]], loc='upper left')
    #plt.legend(['Train', 'valid'], loc='upper left')
    plt.show()
    
    f.savefig('./figure/'+prediction_target_name+'_'+X_train.columns[i]+'.pdf', bbox_inches='tight')
std = std.T