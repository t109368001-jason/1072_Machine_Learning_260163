import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Input, Activation
from keras import initializers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import datetime
import time as time
import os
from keras import backend as K
import random
import re
import multiprocessing

K.clear_session()

#os.chdir(os.path.dirname(os.path.realpath(__file__)))

output = True
split_train = True
save_to_dir = False
pretrain_model = None
#pretrain_model = '20190503_041438_RMSprop_16_37_split_train_True.h5'

workers=multiprocessing.cpu_count()
verbose = 1
batch_size =8
epochs = 150
image_width = 224
validation_split = 0.1
color_mode='rgb'
#color_mode='grayscale'
prediction_target_name = 'character'
metrics = ['acc']
loss = 'categorical_crossentropy'

train_dir = '../input/train/characters-20/'
test_dir = '../input/test/'
tmp_dir = '../input/tmp/'

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, device_count={'GPU':1 , 'CPU':workers}))
keras.backend.set_session(sess)

#optimizer = optimizers.adadelta(lr=1.0, rho=0.95, epsilon=1e-24, decay=0.0)
#optimizer = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-24, decay=0, amsgrad=False)
optimizer = optimizers.rmsprop(lr=0.0001)
#optimizer = optimizers.sgd(lr=0.01, momentum=0, decay=0, nesterov=False)

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
#callbacks.append(keras.callbacks.EarlyStopping(monitor='loss', patience=50))
callbacks.append(keras.callbacks.EarlyStopping(monitor='val_acc', patience=10))
#callbacks.append(RestoreBestWeights(patience=1))
#callbacks.append(EarlyStoppingThreshold(monitor='loss', value=0.1892))
callbacks.append(RestoreBestWeightsFinal())
#callbacks = None

model = Sequential()
if pretrain_model is None:

    #'''224
    uints = 32
    model.add(Conv2D(uints*2, kernel_size=16, strides=1, padding='same', activation='relu', input_shape=(image_width,image_width,1 if color_mode=='grayscale' else 3)))
    model.add(BatchNormalization(epsilon=1e-12))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(Dropout(0.25))
    model.add(Conv2D(uints*1, kernel_size=8, strides=1, padding='same', activation='relu'))
    model.add(Conv2D(uints*1, kernel_size=4, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization(epsilon=1e-12))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(Dropout(0.25))
    model.add(Conv2D(uints*2, kernel_size=4, strides=1, padding='same', activation='relu'))
    model.add(Conv2D(uints*2, kernel_size=4, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization(epsilon=1e-12))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(Dropout(0.25))
    model.add(Conv2D(uints*4, kernel_size=4, strides=1, padding='same', activation='relu'))
    model.add(Conv2D(uints*4, kernel_size=4, strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization(epsilon=1e-12))
    model.add(MaxPooling2D((2, 2), strides=2))
    
    model.add(Flatten())
    model.add(Dense(512, use_bias=False))
    model.add(Activation('relu')) #14
    model.add(BatchNormalization(epsilon=1e-12))
    model.add(Dropout(0.5))
    model.add(Dense(20, activation='softmax'))
    #'''
    '''224
    uints = 32
    model.add(Conv2D(uints*2, kernel_size=16, strides=1, padding='same', activation='relu', input_shape=(image_width,image_width,1 if color_mode=='grayscale' else 3)))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(Dropout(0.25))
    model.add(Conv2D(uints*1, kernel_size=8, strides=1, padding='same', activation='relu'))
    model.add(Conv2D(uints*1, kernel_size=4, strides=2, padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(Dropout(0.25))
    model.add(Conv2D(uints*2, kernel_size=4, strides=1, padding='same', activation='relu'))
    model.add(Conv2D(uints*2, kernel_size=4, strides=2, padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(Dropout(0.25))
    model.add(Conv2D(uints*4, kernel_size=4, strides=1, padding='same', activation='relu'))
    model.add(Conv2D(uints*4, kernel_size=4, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=2))
    
    model.add(Flatten())
    model.add(Dense(512, use_bias=False))
    model.add(Activation('relu')) #14
    model.add(Dropout(0.5))
    model.add(Dense(20, activation='softmax'))
    #'''
    
else:
    f, e = os.path.splitext(pretrain_model)
    json_file = open(f+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(pretrain_model)
    
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model.summary()

train_datagen = ImageDataGenerator( rescale=1. / 255,
                                    validation_split=validation_split if split_train else None)

train_datagen2 = ImageDataGenerator( featurewise_center=False,  # set input mean to 0 over the dataset
                                    samplewise_center=False,  # set each sample mean to 0
                                    featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                    samplewise_std_normalization=False,  # divide each input by its std
                                    zca_whitening=False,  # apply ZCA whitening
                                    zca_epsilon=1e-06,  # epsilon for ZCA whitening
                                    rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
                                    # randomly shift images horizontally (fraction of total width)
                                    width_shift_range=0.2,
                                    # randomly shift images vertically (fraction of total height)
                                    height_shift_range=0.2,
                                    shear_range=0.,  # set range for random shear
                                    zoom_range=0.2,  # set range for random zoom
                                    channel_shift_range=0.,  # set range for random channel shifts
                                    # set mode for filling points outside the input boundaries
                                    fill_mode='nearest',
                                    cval=0.,  # value used for fill_mode = "constant"
                                    horizontal_flip=True,  # randomly flip images
                                    vertical_flip=False,  # randomly flip images
                                    # set rescaling factor (applied before any other transformation)
                                    rescale=1. / 255,
                                    # set function that will be applied on each input
                                    preprocessing_function=None,
                                    # image data format, either "channels_first" or "channels_last"
                                    data_format=None,
                                    validation_split=validation_split if split_train else None)

test_datagen = ImageDataGenerator( rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    color_mode=color_mode,
    target_size=(image_width,image_width),
    batch_size=batch_size,
    save_to_dir=(train_dir+'/train/') if save_to_dir else None,
    subset='training',
    shuffle=True)
#random.shuffle(train_generator.filenames)

train_generator.set_processing_attrs(train_datagen2,
                                     train_generator.target_size,
                                     train_generator.color_mode,
                                     train_generator.data_format,
                                     train_generator.save_to_dir,
                                     train_generator.save_prefix,
                                     train_generator.save_format,
                                     train_generator.subset,
                                     train_generator.interpolation)

valid_generator = train_datagen.flow_from_directory(
    train_dir, # same directory as training data
    color_mode=color_mode,
    target_size=(image_width, image_width),
    save_to_dir=(tmp_dir+'/valid/') if save_to_dir else None,
    batch_size=batch_size,
    subset='validation',
    shuffle=True) # set as validation data

test_generator = test_datagen.flow_from_directory(
    test_dir,
    color_mode=color_mode,
    target_size=(image_width,image_width),
    batch_size=1,
    class_mode=None,  # only data, no labels
    shuffle=False)

convert = lambda text: int(text) if text.isdigit() else text
alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
test_generator.filenames.sort(key=alphanum_key)

t = time.time()

model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = valid_generator if split_train else None, 
    validation_steps = (valid_generator.samples // batch_size) if split_train else None,
    epochs = epochs,
    callbacks=callbacks,
    verbose=verbose,
    workers=workers)
    
history_o = model.history

elapsed = time.time() - t

history = pd.DataFrame(history_o.history)

Y_test = model.predict_generator(test_generator, test_generator.samples,
    verbose=verbose,
    workers=workers)

Y_test = np.argmax(Y_test, axis=1)

characters_20 = {v: k for k, v in train_generator.class_indices.items()}

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

for i in range(len(metrics)):
    f = plt.figure(figsize=(10,10));
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
    if output:
        f.savefig(filename+'.pdf', bbox_inches='tight')

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
        print('val_'+metrics[i]+' : ' + str(history['val_'+metrics[i]].values[-1]) + '    ' + str(history['val_'+metrics[i]].max()))
print('    loss : ' + str(final_loss))
if split_train:
    print('val_loss : ' + str(final_val_loss)+ '    ' + str(history['val_loss'].min()))
else:
    print('\n')