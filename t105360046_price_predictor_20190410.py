import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras import initializers
from keras import activations
from keras import optimizers
from tensorflow.contrib.opt import AdamWOptimizer
import datetime
import time as time
import sys
import math
from keras import backend as K


trainFile = pd.read_csv('./train-v3.csv', index_col=0)
validFile = pd.read_csv('./valid-v3.csv', index_col=0)
testFile = pd.read_csv('./test-v3.csv', index_col=0)

trainFile.index = np.linspace(0, trainFile.shape[0]-1, trainFile.shape[0], dtype=int)
validFile.index = np.linspace(0, validFile.shape[0]-1, validFile.shape[0], dtype=int)
testFile.index = np.linspace(0, testFile.shape[0]-1, testFile.shape[0], dtype=int)

"""
for i in range(trainFile.shape[0]):
    if trainFile.at[i, 'price'] > 4000000:
        trainFile = trainFile.drop(i, axis=0)
        i -= 1;
"""
"""
#remove_columns = ['sale_yr', 'sale_month', 'sale_day', 'zipcode', 'lat', 'long', 'yr_built', 'yr_renovated', 'floors', 'condition', 'waterfront', 'view', 'sqft_basement', 'sqft_lot15', 'bedrooms', 'bathrooms', 'sqft_lot']
remove_columns = ['sale_yr', 'sale_month', 'sale_day']
trainFile = trainFile.drop(columns=remove_columns)
validFile = validFile.drop(columns=remove_columns)
testFile = testFile.drop(columns=remove_columns)
"""
"""
duplicate_columns = ['sqft_living', 'sqft_above', 'grade', 'sqft_living15', 'bathrooms']
for i in range(len(duplicate_columns)):
    trainFile['c'+str(i)] = trainFile[duplicate_columns[i]]
    validFile['c'+str(i)] = validFile[duplicate_columns[i]]
    testFile['c'+str(i)] = testFile[duplicate_columns[i]]
"""
"""
trainFile['c'] = trainFile['sqft_living']+trainFile['sqft_above']
validFile['c'] = validFile['sqft_living']+validFile['sqft_above']
testFile['c'] = testFile['sqft_living']+testFile['sqft_above']
"""
output = False
force_gpu = False
train_valid = False
use_scaler = True
Y_scaler = False

#batch_size = int((train_valid and trainFile.shape[0]+validFile.shape[0] or trainFile.shape[0])/1+1)
batch_size = 1024
#batch_size = 1
epochs = 1000000
verbose = 2
prediction_target_name = 'price'
#hidden_unit = int((trainFile.shape[1]-1)*2/3)+1
#hidden_unit = int((trainFile.shape[1])*1/2)
hidden_unit = 1024
metrics = ['mean_absolute_error']
loss = 'mean_squared_error'
#loss = lambda y_true, y_pred : keras.backend.mean(keras.backend.square(y_true - y_pred))

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, device_count={'GPU':1 if force_gpu else (hidden_unit*batch_size > 1e5 and 1 or 0), 'CPU':4}))
keras.backend.set_session(sess)

optimizer = optimizers.adadelta(lr=10.0, rho=0.95, epsilon=1e-10, decay=0)
#optimizer = optimizers.adagrad(lr=0.01, epsilon=None, decay=0)
#optimizer = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
#optimizer = AdamWOptimizer(weight_decay=1e-4, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-10, use_locking=False)
#optimizer = optimizers.adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0)
#optimizer = optimizers.nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
#optimizer = optimizers.rmsprop(lr=0.001, rho=0.9, epsilon=None, decay=0)
#optimizer = optimizers.sgd(lr=0.01, momentum=0.0, decay=0, nesterov=False)
#optimizer = optimizers.sgd(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)

#kernel_initializer = initializers.normal(mean=0, stddev=0.5, seed=None)
#kernel_initializer = initializers.uniform(minval=-0.05, maxval=0.05, seed=None)
#kernel_initializer = initializers.truncated_normal(mean=0.0, stddev=0.05, seed=None)
#kernel_initializer = initializers.orthogonal(gain=1, seed=None)
#kernel_initializer = initializers.identity(gain=1)
#kernel_initializer = initializers.he_uniform()
kernel_initializer = initializers.glorot_uniform()

#bias_initializer = initializers.normal(mean=0, stddev=0.5, seed=None)
#bias_initializer = initializers.uniform(minval=-0.5, maxval=0.5, seed=None)
#bias_initializer = initializers.truncated_normal(mean=0.0, stddev=0.05, seed=None)
#bias_initializer = initializers.orthogonal(gain=1, seed=None)
#bias_initializer = initializers.identity(gain=1)
#bias_initializer = initializers.he_uniform()
#bias_initializer = initializers.glorot_uniform()
bias_initializer = initializers.zeros()

kernel_regularizer = regularizers.l2(l=0.01)
#kernel_regularizer = None

#bias_regularizer = regularizers.l2(l=0.01)
bias_regularizer = None

#activation = 'elu'
#activation = 'hard_sigmoid'
#activation = 'linear'
#activation = lambda x : K.abs(x)*x
#activation = lambda x : x*x
#activation = lambda x : activations.relu(x, threshold=10000.0, alpha=0)
activation = 'relu'
#activation = 'selu'
#activation = 'sigmoid'
#activation = 'softmax'
#activation = 'softplus'
#activation = 'softsign'
#activation = 'tanh'
#activation = None

class EarlyStoppingThreshold(keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', value=0, verbose=0):
        super(EarlyStoppingThreshold, self).__init__()
        self.monitor=monitor
        self.value=value
        self.verbose=verbose
    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            return
        if(current < self.value):
            self.stopped_epoch = epoch
            self.model.stop_training = True


class RestoreBestWeights(keras.callbacks.Callback):
    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None):
        super(RestoreBestWeights, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.set_weights(self.best_weights)

callbacks = []
callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', patience=500))
#callbacks.append(RestoreBestWeights(monitor='val_loss', patience=10))
#callbacks.append(EarlyStoppingThreshold(monitor='mean_absolute_error', value=20000))
#callbacks = None

trainFile = trainFile.dropna()
validFile = validFile.dropna()
testFile = testFile.dropna()

X_train = trainFile.drop(columns=prediction_target_name)
Y_train = trainFile[prediction_target_name]

X_valid = validFile.drop(columns=prediction_target_name)
Y_valid = validFile[prediction_target_name]

X_test = testFile

X_train_stats = X_train.describe()
X_train_stats = X_train_stats.transpose()

Y_train_mean = Y_train.mean()
Y_train_std = Y_train.std()


def normX(x):
  return ((x - X_train_stats['mean']) / X_train_stats['std']).values

if use_scaler:
    X_train = normX(X_train)
    X_valid = normX(X_valid)
    X_test = normX(X_test)

if Y_scaler:
    Y_train = (Y_train - Y_train_mean) / Y_train_std
    Y_valid = (Y_valid - Y_train_mean) / Y_train_std

if train_valid:
    X_train = np.append(X_train, X_valid, axis=0)
    X_valid = None
    Y_train = np.append(Y_train, Y_valid, axis=0)
    Y_valid = None

model = Sequential()
model.add(Dense(int(hidden_unit), activation=activation, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, input_dim=X_train.shape[1]))
#model.add(Dropout(0.25))

model.add(Dense(int(hidden_unit/2), activation=activation, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer))
#model.add(Dropout(0.25))

model.add(Dense( 1, activation=activation))
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

t = time.time()

history_o = model.fit(X_train, Y_train, validation_data= None if train_valid else (X_valid, Y_valid), batch_size=batch_size, epochs=epochs, shuffle=True, verbose=verbose, callbacks=None if callbacks == None else callbacks)

elapsed = time.time() - t

history = pd.DataFrame(history_o.history)

Y_test = model.predict(X_test)

if Y_scaler:
    history = history * Y_train_std + Y_train_mean
    
history['epoch'] = history_o.epoch
    
final_loss = history['loss'].values[-1]
    
if not train_valid:
    final_val_loss = history['val_loss'].values[-1]

filename = './result/'
filename += datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename += '_'+optimizer.__class__.__name__
filename += '_'+kernel_initializer.__class__.__name__
filename += '_'+activation
filename += '_'+str(batch_size)
filename += '_'+str(history['epoch'].values[-1])
filename += '_'+str(hidden_unit)
filename += '_train_valid_'+str(train_valid)
filename += '_' + ''

for i in range(len(metrics)):
    plt.plot(history[metrics[i]])
    if train_valid:
        plt.title('Model loss='+str(final_loss))
    else:
        plt.plot(history['val_'+metrics[i]])
        plt.title('Model loss='+str(history[metrics[i]].values[-1])+'\nval_loss='+str(history['val_'+metrics[i]].values[-1]))
    plt.title('Model '+metrics[i])
    plt.ylabel(metrics)
    plt.xlabel('Epoch')
    plt.ylim(0, history[metrics[i]].mean()*2)
    plt.xlim(left=0)
    plt.legend('Train', loc='upper right')
    plt.show()

f = plt.figure(figsize=(10,10));
# Plot training & validation loss values
plt.plot(history['loss'])
if train_valid:
    plt.title('Model loss='+str(final_loss))
else:
    plt.plot(history['val_loss'])
    plt.title('Model loss='+str(final_loss)+'\nval_loss='+str(final_val_loss))
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.ylim(0, history['loss'].mean()*2)
plt.xlim(left=0)
plt.grid()
if train_valid:
    plt.legend('Train', loc='upper left')
else:
    plt.legend(['Train', 'valid'], loc='upper right')
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
    
print(' elapsed : ' + str(elapsed))
for i in range(len(metrics)):
    print('    '+metrics[i]+' : ' + str(history[metrics[i]].values[-1]))
    if not train_valid:
        print('val_'+metrics[i]+' : ' + str(history['val_'+metrics[i]].values[-1]) + '    ' + str(history['val_'+metrics[i]].min()))
print('    loss : ' + str(final_loss))
if not train_valid:
    print('val_loss : ' + str(final_val_loss)+ '    ' + str(history['val_loss'].min()))
else:
    print('\n')
print('test price max : '+str(Y_test.max()))
print('   hidden_unit : '+str(hidden_unit))
print('    batch_size : '+str(batch_size))
print('    mae / mean : '+str(history['mean_absolute_error'].values[-1]/Y_train.mean()*100))
if not train_valid:
    print('val_mae / mean : '+str(history['val_mean_absolute_error'].values[-1]/Y_valid.mean()*100))
#print(' test sqft_living max : '+str(pd.DataFrame(scaler.inverse_transform(testFile.values), columns=testFile.columns)['sqft_living'].max()))
#print('train sqft_living max : '+str(pd.DataFrame(scaler.inverse_transform(trainFile.values), columns=trainFile.columns)['sqft_living'].max()))

