import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras import initializers
from keras import activations
from keras import optimizers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from tensorflow.contrib.opt import AdamWOptimizer
import datetime
import time as time
import sys
import inspect, os
import math
from keras import backend as K
from scipy.signal import butter, lfilter, freqz
from scipy import misc
import cv2
import gc
import random


os.chdir(os.path.dirname(os.path.realpath(__file__)))

output = False
use_gpu = True
split_train = True

verbose = 1
batch_size = 32
epochs = 1000
image_width = 128
prediction_target_name = 'character'
metrics = ['acc']
loss = 'categorical_crossentropy'

train_dir = '../input/train/'
test_dir = '../input/test/'
tmp_dir = '../input/tmp/'

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, device_count={'GPU':1 if use_gpu else 0, 'CPU':4}))
keras.backend.set_session(sess)

characters_20 = ['{}'.format(i) for i in os.listdir(train_dir)]
train_path = np.array([]).reshape(0,2)

for index,character in enumerate(characters_20):
    paths = np.array(['../input/train/{}/{}'.format(character,i) for i in os.listdir(train_dir+character)]).T.reshape(-1,1)
    
    train_character = np.append(paths,(index)*np.ones(shape=paths.shape),axis=1)
    train_path = np.append(train_path,train_character, axis=0)
random.shuffle(train_path)

if split_train:
    train_path, valid_path = train_test_split(train_path, test_size=0.1, random_state=2)
    X_valid_path = valid_path[:,0]
    Y_valid = valid_path[:,1]

X_train_path = train_path[:,0]
Y_train = train_path[:,1]
t = time.time()
print('loading images')
try:
    X_train = np.load(tmp_dir+str(image_width)+'_X_train.npy')
except FileNotFoundError:
    X_train = [cv2.resize(cv2.imread(i, cv2.IMREAD_COLOR), (image_width,image_width), interpolation=cv2.INTER_CUBIC) 
            for i in X_train_path]
    np.save(tmp_dir+str(image_width)+'_X_train.npy',X_train)
print('X_train created')
try:
    X_valid = np.load(tmp_dir+str(image_width)+'_X_valid.npy')
except FileNotFoundError:
    X_valid = [cv2.resize(cv2.imread(i, cv2.IMREAD_COLOR), (image_width,image_width), interpolation=cv2.INTER_CUBIC) 
            for i in X_valid_path]
    np.save(tmp_dir+str(image_width)+'_X_valid.npy',X_valid)
print('X_valid created')

X_test_path = ['../input/test/{}'.format(i) for i in os.listdir(test_dir)]
X_test_path.sort()
X_test = [cv2.resize(cv2.imread(i, cv2.IMREAD_COLOR), (image_width,image_width), interpolation=cv2.INTER_CUBIC) 
            for i in X_test_path]

elapsed = time.time() - t
print(str(elapsed)+' s')
    
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_train = X_train.astype('float32')
Y_train = Y_train.astype('float32')
X_train /= 255
Y_train = keras.utils.to_categorical(Y_train, len(characters_20))

if split_train:
    X_valid = np.array(X_valid)
    Y_valid = np.array(Y_valid)
    X_valid = X_valid.astype('float32')
    Y_valid = Y_valid.astype('float32')
    X_valid /= 255
    Y_valid = keras.utils.to_categorical(Y_valid, len(characters_20))


X_test = np.array(X_test)
#optimizer = optimizers.adadelta(lr=1.0, rho=0.95, epsilon=1e-24, decay=0.0)
#optimizer = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-24, decay=0, amsgrad=False)
optimizer = optimizers.rmsprop(lr=0.0001, decay=1e-6)
#optimizer = optimizers.sgd(lr=0.01, momentum=0, decay=0, nesterov=False)

#kernel_initializer = initializers.normal(mean=0, stddev=0.5, seed=None)
#kernel_initializer = initializers.uniform(minval=-0.05, maxval=0.05, seed=None)
#kernel_initializer = initializers.truncated_normal(mean=0.0, stddev=0.05, seed=None)
#kernel_initializer = initializers.orthogonal(gain=1, seed=None)
#kernel_initializer = initializers.identity(gain=1)
#kernel_initializer = initializers.he_uniform()
#kernel_initializer = initializers.glorot_uniform()

callbacks = []
#callbacks.append(keras.callbacks.EarlyStopping(monitor='loss', patience=50))
callbacks.append(keras.callbacks.EarlyStopping(monitor='val_acc', patience=20))
#callbacks.append(RestoreBestWeights(patience=1))
#callbacks.append(EarlyStoppingThreshold(monitor='loss', value=0.1892))
#callbacks.append(RestoreBestWeightsFinal())
#callbacks = None

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_width,image_width,3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
#model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(20, activation='softmax'))

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model.summary()

train_datagen = ImageDataGenerator( featurewise_center=False,  # set input mean to 0 over the dataset
                                    samplewise_center=False,  # set each sample mean to 0
                                    featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                    samplewise_std_normalization=False,  # divide each input by its std
                                    zca_whitening=False,  # apply ZCA whitening
                                    zca_epsilon=1e-06,  # epsilon for ZCA whitening
                                    rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
                                    # randomly shift images horizontally (fraction of total width)
                                    width_shift_range=0.1,
                                    # randomly shift images vertically (fraction of total height)
                                    height_shift_range=0.1,
                                    shear_range=0.2,  # set range for random shear
                                    zoom_range=0.2,  # set range for random zoom
                                    channel_shift_range=0.,  # set range for random channel shifts
                                    # set mode for filling points outside the input boundaries
                                    fill_mode='nearest',
                                    cval=0.,  # value used for fill_mode = "constant"
                                    horizontal_flip=True,  # randomly flip images
                                    vertical_flip=False,  # randomly flip images
                                    # set rescaling factor (applied before any other transformation)
                                    rescale=None,
                                    # set function that will be applied on each input
                                    preprocessing_function=None,
                                    # image data format, either "channels_first" or "channels_last"
                                    data_format=None)

t = time.time()

history_o = model.fit_generator(train_datagen.flow(X_train, Y_train, batch_size=batch_size),
                                steps_per_epoch=len(X_train)//batch_size,
                                epochs=epochs, validation_data=(X_valid, Y_valid) if split_train else None,
                                validation_steps=(len(X_valid) // batch_size) if split_train else None,
                                callbacks=callbacks,
                                workers=8,
                                )

elapsed = time.time() - t

history = pd.DataFrame(history_o.history)

Y_test = model.predict(X_test)

Y_test = np.argmax(Y_test, axis=1)

Y_test = np.array([characters_20[i] for i in Y_test])

history['epoch'] = history_o.epoch
    
final_loss = history['loss'].values[-1]
    
if split_train:
    final_val_loss = history['val_loss'].values[-1]

filename = '../result/'
filename += datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename += '_'+optimizer.__class__.__name__
filename += '_'+str(batch_size)
filename += '_'+str(history['epoch'].values[-1])
filename += '_split_train_'+str(split_train)
filename += '_' + 'normalY'

for i in range(len(metrics)):
    plt.plot(history[metrics[i]])
    if not split_train:
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
    plt.grid()
    plt.show()

f = plt.figure(figsize=(10,10));

plt.plot(history['loss'])
if not split_train:
    plt.title('Model loss='+str(final_loss))
else:
    plt.plot(history['val_loss'])
    plt.title('Model loss='+str(final_loss)+'\nval_loss='+str(final_val_loss))
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.ylim(0, history['loss'].mean()*2)
plt.xlim(left=0)
plt.grid()
if not split_train:
    plt.legend('Train', loc='upper left')
else:
    plt.legend(['Train', 'valid'], loc='upper right')
plt.show()

if output:
    Y_test_csv_format = pd.DataFrame(Y_test, index=np.linspace(1, Y_test.shape[0], Y_test.shape[0], dtype=int), columns=[prediction_target_name])
    Y_test_csv_format.to_csv(filename+'.csv', index_label='id')
    f.savefig(filename+'.pdf', bbox_inches='tight')
    model_json = model.to_json()
    with open(filename+'.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(filename+'.h5')
    
print(' elapsed : ' + str(elapsed))
for i in range(len(metrics)):
    print('    '+metrics[i]+' : ' + str(history[metrics[i]].values[-1]))
    if split_train:
        print('val_'+metrics[i]+' : ' + str(history['val_'+metrics[i]].values[-1]) + '    ' + str(history['val_'+metrics[i]].min()))
print('    loss : ' + str(final_loss))
if split_train:
    print('val_loss : ' + str(final_val_loss)+ '    ' + str(history['val_loss'].min()))
else:
    print('\n')
print('    batch_size : '+str(batch_size))
