import datetime
import time as time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import initializers
from keras import optimizers

output = False
use_gpu = False
train_valid = False

verbose = 2
batch_size = 32
epochs = 1000000
prediction_target_name = 'price'
hidden_unit = 64
loss = 'mean_absolute_error'

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(
        gpu_options=gpu_options, 
        device_count={'GPU':1 if use_gpu else 0, 'CPU':4}))
keras.backend.set_session(sess)

trainFile = pd.read_csv('./train-v3.csv', index_col=0)
validFile = pd.read_csv('./valid-v3.csv', index_col=0)
testFile = pd.read_csv('./test-v3.csv', index_col=0)

trainFile.index = np.linspace(0, trainFile.shape[0]-1, trainFile.shape[0], dtype=int)
validFile.index = np.linspace(0, validFile.shape[0]-1, validFile.shape[0], dtype=int)
testFile.index = np.linspace(0, testFile.shape[0]-1, testFile.shape[0], dtype=int)

for i in range(trainFile.shape[0]):
    if (trainFile.at[i, 'price'] > 3000000):
        trainFile = trainFile.drop(i, axis=0)
        i -= 1;

if train_valid:
    for i in range(validFile.shape[0]):
        if (validFile.at[i, 'price'] > 3000000):
            validFile = validFile.drop(i, axis=0)
            i -= 1;

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

def normY(y):
  return (y - Y_train_mean) / Y_train_std

def inormY(y):
  return y * Y_train_std + Y_train_mean

X_train = normX(X_train)
X_valid = normX(X_valid)
X_test = normX(X_test)

Y_train = normY(Y_train)
Y_valid = normY(Y_valid)

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

optimizer = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-24, decay=0, amsgrad=False)

kernel_initializer = initializers.uniform(minval=-0.05, maxval=0.05, seed=None)
activation = 'relu'

class RestoreBestWeightsFinal(keras.callbacks.Callback):
    def __init__(self,
                 min_delta=0,
                 mode='auto',
                 baseline=None):
        super(RestoreBestWeightsFinal, self).__init__()
        self.min_delta = min_delta
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        
    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)

    def on_epoch_end(self, epoch, logs=None):
        val_current = logs.get('val_loss')
        if val_current is None:
            return

        if self.monitor_op(val_current - self.min_delta, self.best):
            self.best = val_current
            self.best_weights = self.model.get_weights()
                
callbacks = []
callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', patience=10))
callbacks.append(RestoreBestWeightsFinal())

model = Sequential()
model.add(Dense(int(hidden_unit), activation=activation, kernel_initializer=kernel_initializer, input_dim=X_train.shape[1]))
model.add(Dense(int(hidden_unit/2), activation=activation, kernel_initializer=kernel_initializer))
model.add(Dense( 1, activation=None))
model.compile(optimizer=optimizer, loss=loss)
model.summary()

t = time.time()
history_o = model.fit(X_train, Y_train, validation_data= None if train_valid else (X_valid, Y_valid), 
                      batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks)
elapsed = time.time() - t

history = pd.DataFrame(history_o.history)

Y_test = model.predict(X_test)

Y_test = inormY(Y_test)
history = history * Y_train_std
    
history['epoch'] = history_o.epoch

filename = './result/'
filename += datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename += '_'+optimizer.__class__.__name__
filename += '_'+kernel_initializer.__class__.__name__
filename += '_'+activation
filename += '_'+str(batch_size)
filename += '_'+str(history['epoch'].values[-1])
filename += '_'+str(hidden_unit)
filename += '_train_valid' if train_valid else ''

f = plt.figure(figsize=(10,10));
train_valid = False
plt.plot(history['loss'])
if not train_valid:
    plt.plot(history['val_loss'])
plt.title('elapsed='+str(elapsed)+'s\n'
          +'loss='+str(history['loss'].values[-1])+
          (train_valid and ' ' or '\nval_loss='+str(history['val_loss'].values[-1])))
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.ylim(0, history['loss'].mean()*2)
plt.xlim(left=0)
plt.grid()
plt.legend(train_valid and 'Train' or ['Train', 'valid'], loc='upper right')
plt.show()

if output:
    Y_test_csv_format = pd.DataFrame(Y_test, 
             index=np.linspace(1, Y_test.shape[0], Y_test.shape[0], dtype=int), 
             columns=[prediction_target_name])
    Y_test_csv_format.to_csv(filename+'.csv', index_label='id')
    f.savefig(filename+'.pdf', bbox_inches='tight')
    model_json = model.to_json()
    with open(filename+'.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(filename+'.h5')
    