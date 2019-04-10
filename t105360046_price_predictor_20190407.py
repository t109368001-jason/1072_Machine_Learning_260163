import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import initializers
from keras import activations
from keras import optimizers
import datetime
import time as time
import sys


trainFile = pd.read_csv('./train-v3.csv', index_col=0)
validFile = pd.read_csv('./valid-v3.csv', index_col=0)
testFile = pd.read_csv('./test-v3.csv', index_col=0)

#remove_columns = ['lat', 'long', 'sale_yr', 'sale_month', 'sale_day', 'zipcode', 'floors', 'yr_built', 'yr_renovated']
#trainFile = trainFile.drop(columns=remove_columns)
#validFile = validFile.drop(columns=remove_columns)
#testFile = testFile.drop(columns=remove_columns)

train_valid = False

#batch_size = int((valid_date_as_train_data and trainFile.shape[0]+validFile.shape[0] or trainFile.shape[0])/1+1)
batch_size = 32
#batch_size = 1
epochs = 50
verbose = 1
output = False
prediction_target_name = 'price'
hidden_unit = int((trainFile.shape[1]-1)*2/3)+1
metrics = ['acc', 'mean_squared_error']

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, device_count={'GPU':(batch_size > 500 and 1 or 0), 'CPU':4}))
keras.backend.set_session(sess)

#optimizer = optimizers.adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0)
#optimizer = optimizers.adagrad(lr=0.01, epsilon=None, decay=0)
#optimizer = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
#optimizer = optimizers.adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0)
optimizer = optimizers.nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
#optimizer = optimizers.rmsprop(lr=0.001, rho=0.9, epsilon=None, decay=0)
#optimizer = optimizers.sgd(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

#kernel_initializer = initializers.normal(mean=0, stddev=0.5, seed=None)
#kernel_initializer = initializers.uniform(minval=-0.05, maxval=0.05, seed=None)
#kernel_initializer = initializers.truncated_normal(mean=0.0, stddev=0.05, seed=None)
#kernel_initializer = initializers.orthogonal(gain=1, seed=None)
#kernel_initializer = initializers.identity(gain=1)
kernel_initializer = initializers.he_uniform()

#activation = 'elu'
#activation = 'hard_sigmoid'
#activation = 'linear'
activation = lambda x : activations.relu(x, alpha=-1)
#activation = 'relu'
#activation = 'selu'
#activation = 'sigmoid' 
#activation = 'softmax'
#activation = 'softplus'
#activation = 'softsign'
#activation = 'tanh'
#activation = None

X_train = trainFile.drop(columns=prediction_target_name).values
Y_train = trainFile[prediction_target_name].values.reshape(-1,1)

X_valid = validFile.drop(columns=prediction_target_name).values
Y_valid = validFile[prediction_target_name].values.reshape(-1,1)

X_test = testFile.values

X_train = (X_train - X_train.mean())
X_valid = (X_valid - X_valid.mean())
X_test = (X_test - X_test.mean())

X_train = X_train/(X_train.max() > -(X_train.min()) and X_train.max() or X_train.min())
X_valid = X_valid/(X_valid.max() > -(X_valid.min()) and X_valid.max() or X_valid.min())
X_test = X_test/(X_train.max() > -(X_test.min()) and X_test.max() or X_test.min())

if train_valid:
    X_train = np.append(X_train, X_valid, axis=0)
    Y_train = np.append(Y_train, Y_valid, axis=0)

model = Sequential()
model.add(Dense(hidden_unit, input_dim=X_train.shape[1], kernel_initializer=kernel_initializer))
model.add(Activation(activation))
model.add(Dropout(0.5))
model.add(Dense(1, activation=activation))

model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=metrics)

t = time.time()

if train_valid:
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=verbose)
else:
    history = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), batch_size=batch_size, epochs=epochs, shuffle=True, verbose=verbose)
elapsed = time.time() - t

Y_test = model.predict(X_test)

#Y_test = Y_scaler.inverse_transform(Y_test)

filename = './result/'
filename += datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename += '_'+optimizer.__class__.__name__
filename += '_'+kernel_initializer.__class__.__name__
filename += '_'+'relu'
filename += '_'+str(batch_size)
filename += '_'+str(epochs)
filename += '_'+str(hidden_unit)
filename += '_train_valid_'+str(train_valid)

for i in range(len(metrics)):
    plt.plot(history.history[metrics[i]])
    plt.title('Model '+metrics[i])
    plt.ylabel(metrics)
    plt.xlabel('Epoch')
    plt.legend('Train', loc='upper left')
    plt.show()

f = plt.figure(figsize=(10,10));
# Plot training & validation loss values
plt.plot(history.history['loss'])
if not train_valid:
    plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.ylim(0,np.array(history.history['loss']).mean()*2)
plt.xlim(left=0)
if train_valid:
    plt.legend('Train', loc='upper left')
else:
    plt.legend(['Train', 'valid'], loc='upper left')
plt.show()

if output:
    Y_test_csv_format = pd.DataFrame(Y_test, index=np.linspace(1, Y_test.shape[0], Y_test.shape[0], dtype=int), columns=[prediction_target_name])
    Y_test_csv_format.to_csv(filename+'.csv', index_label='id')
    f.savefig(filename+'.pdf', bbox_inches='tight')
    # serialize model to JSON
    model_json = model.to_json()
    with open(filename+'.json', 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(filename+'.h5')
    
sys.stdout.write(' elapsed : ' + str(elapsed) + 's\n')
for i in range(len(metrics)):
    sys.stdout.write(metrics[i]+' : ' + str(history.history[metrics[i]][-1]))
sys.stdout.write('    loss : ' + str(history.history['loss'][-1])+'\t')
if not train_valid:
    sys.stdout.write('val_loss : ' + str(history.history['val_loss'][-1])+'\n')
else:
    sys.stdout.write('\n')
    

sys.stdout.write('test price max : '+str(Y_test.max())+'\t')
sys.stdout.write('train price max : '+str(Y_train.max())+'\n')
#print(' test sqft_living max : '+str(pd.DataFrame(scaler.inverse_transform(testFile.values), columns=testFile.columns)['sqft_living'].max()))
#print('train sqft_living max : '+str(pd.DataFrame(scaler.inverse_transform(trainFile.values), columns=trainFile.columns)['sqft_living'].max()))

