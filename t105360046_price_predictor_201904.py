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
from scipy.signal import butter, lfilter, freqz


output = False
use_gpu = False
train_valid = False

verbose = 2
batch_size = 16
epochs = 1000000
prediction_target_name = 'price'
hidden_unit = 64
metrics = []
loss = 'mean_absolute_error'

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, device_count={'GPU':1 if use_gpu else 0, 'CPU':4}))
keras.backend.set_session(sess)

trainFile = pd.read_csv('./train-v3.csv', index_col=0)
validFile = pd.read_csv('./valid-v3.csv', index_col=0)
testFile = pd.read_csv('./test-v3.csv', index_col=0)


trainFile.index = np.linspace(0, trainFile.shape[0]-1, trainFile.shape[0], dtype=int)
validFile.index = np.linspace(0, validFile.shape[0]-1, validFile.shape[0], dtype=int)
testFile.index = np.linspace(0, testFile.shape[0]-1, testFile.shape[0], dtype=int)
for i in range(trainFile.shape[0]):
    if trainFile.at[i, 'price'] > 4000000:
        trainFile = trainFile.drop(i, axis=0)
        i -= 1;

for i in range(validFile.shape[0]):
    if validFile.at[i, 'price'] > 4000000:
        validFile = validFile.drop(i, axis=0)
        i -= 1;


remove_columns = []                     #641 638 646 644 636 641 625 636 
#remove_columns.append('sale_yr')       #649 645 644 645 644 629 647 649
#remove_columns.append('sale_month')    #645 631 639 638 637 636 640 639
#remove_columns.append('sale_day')      #637 630 648 640 637 642 636 637
#remove_columns.append('zipcode')       #648 650 639 634 650 650 
remove_columns.append('lat')           #
#remove_columns.append('long')          #
#remove_columns.append('yr_built')      #
#remove_columns.append('yr_renovated')  #
#remove_columns.append('floors')        #
#remove_columns.append('condition')     #
#remove_columns.append('waterfront')    #
#remove_columns.append('view')          #
#remove_columns.append('sqft_basement') #
#remove_columns.append('sqft_lot15')    #
#remove_columns.append('bedrooms')      #
#remove_columns.append('sqft_lot')      #

trainFile = trainFile.drop(columns=remove_columns)
validFile = validFile.drop(columns=remove_columns)
testFile = testFile.drop(columns=remove_columns)

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
  return (x - X_train_stats['mean']) / X_train_stats['std']

X_train = normX(X_train)
X_valid = normX(X_valid)
X_test = normX(X_test)

Y_train = (Y_train - Y_train_mean) / Y_train_std
Y_valid = (Y_valid - Y_train_mean) / Y_train_std

if train_valid:
    X_train = X_train.append(X_valid)
    X_valid = X_valid.drop(index=X_valid.index)
    Y_train = Y_train.append(Y_valid)
    Y_valid = Y_valid.drop(index=Y_valid.index)
    
X_train = X_train.values
X_valid = X_valid.values
X_test = X_test.values
Y_train = Y_train.values
Y_valid = Y_valid.values

#optimizer = optimizers.adadelta(lr=1.0, rho=0.95, epsilon=1e-24, decay=0.0)
#optimizer = optimizers.adagrad(lr=0.01, epsilon=None, decay=0)
#optimizer = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-24, decay=0, amsgrad=False)
#optimizer = AdamWOptimizer(weight_decay=1e-6, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-24, use_locking=False)
#optimizer = optimizers.adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0)
#optimizer = optimizers.nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
#optimizer = optimizers.rmsprop(lr=0.001, rho=0.9, epsilon=1e-24, decay=0)
optimizer = optimizers.sgd(lr=0.01, momentum=0, decay=0, nesterov=False)
#optimizer = optimizers.sgd(lr=0.01, momentum=0.9, decay=1e-6, nesterov=False)

#kernel_initializer = initializers.normal(mean=0, stddev=0.5, seed=None)
kernel_initializer = initializers.uniform(minval=-0.05, maxval=0.05, seed=None)
#kernel_initializer = initializers.truncated_normal(mean=0.0, stddev=0.05, seed=None)
#kernel_initializer = initializers.orthogonal(gain=1, seed=None)
#kernel_initializer = initializers.identity(gain=1)
#kernel_initializer = initializers.he_uniform()
#kernel_initializer = initializers.glorot_uniform()

#bias_initializer = initializers.normal(mean=0, stddev=0.5, seed=None)
#bias_initializer = initializers.uniform(minval=-0.05, maxval=0.05, seed=None)
#bias_initializer = initializers.truncated_normal(mean=0.0, stddev=0.05, seed=None)
#bias_initializer = initializers.orthogonal(gain=1, seed=None)
#bias_initializer = initializers.identity(gain=1)
#bias_initializer = initializers.he_uniform()
#bias_initializer = initializers.glorot_uniform()
bias_initializer = initializers.zeros()

#kernel_regularizer = regularizers.l2(l=0.01)
kernel_regularizer = None

#bias_regularizer = regularizers.l2(l=0.01)
bias_regularizer = None

#activation = 'elu'
#activation = 'hard_sigmoid'
#activation = 'linear'
activation = 'relu'
#activation = 'selu'
#activation = 'sigmoid'
#activation = 'softmax'
#activation = 'softplus'
#activation = 'softsign'
#activation = 'tanh'
#activation = None

class EarlyStoppingThreshold(keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', value=0):
        super(EarlyStoppingThreshold, self).__init__()
        self.monitor=monitor
        self.value=value
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
        val_current = logs.get('val_loss')
        current = logs.get('loss')
        if current is None:
            return

        if self.monitor_op(val_current - self.min_delta, self.best) and (val_current < (current*0.99)):
            self.best = val_current
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.set_weights(self.best_weights)

callbacks = []
#callbacks.append(keras.callbacks.EarlyStopping(monitor='loss', patience=100))
callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', patience=200))
#callbacks.append(RestoreBestWeights(patience=1))
#callbacks.append(EarlyStoppingThreshold(monitor='loss', value=0.1400))
#callbacks = None

model = Sequential()
model.add(Dense(int(hidden_unit), activation=activation, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, input_dim=X_train.shape[1]))
#model.add(Dropout(0.25))

model.add(Dense(int(hidden_unit/2), activation=activation, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer))
#model.add(Dropout(0.25))

"""
i=hidden_unit/4
while i > 1 :    
    model.add(Dense(int(i), activation=activation, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer))
    #model.add(Dropout(0.25))
    i =i / 2
"""

model.add(Dense( 1, activation=None))
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model.summary()

t = time.time()

history_o = model.fit(X_train, Y_train, validation_data= None if train_valid else (X_valid, Y_valid), batch_size=batch_size, epochs=epochs, shuffle=True, verbose=verbose, callbacks=None if callbacks == None else callbacks)

elapsed = time.time() - t

history = pd.DataFrame(history_o.history)

Y_test = model.predict(X_test)

Y_test = (Y_test * Y_train_std) + Y_train_mean
history = history * Y_train_std
    
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
filename += '_' + 'normalY'

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
#print(' test sqft_living max : '+str(pd.DataFrame(scaler.inverse_transform(testFile.values), columns=testFile.columns)['sqft_living'].max()))
#print('train sqft_living max : '+str(pd.DataFrame(scaler.inverse_transform(trainFile.values), columns=trainFile.columns)['sqft_living'].max()))

