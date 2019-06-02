import keras
import tensorflow as tf
import os
from os.path import join
import json
import random
import itertools
import re
import datetime
import cairocffi as cairo
import editdistance
import numpy as np
from scipy import ndimage
import pylab
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model, load_model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing import image
import keras.callbacks
import cv2
from collections import Counter

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, device_count={'GPU':1 , 'CPU':workers}))
keras.backend.set_session(sess)

train_data_path = '../input/train/'
test_data_path = '../input/test/'

def get_counter(dirpath):
    dirname = os.path.basename(dirpath)
    ann_dirpath = join(dirpath, 'ann')
    letters = ''
    lens = []
    for filename in os.listdir(ann_dirpath):
        json_filepath = join(ann_dirpath, filename)
        description = json.load(open(json_filepath, 'r'))['description']
        lens.append(len(description))
        letters += description
    print('Max plate length in "%s":' % dirname, max(Counter(lens).keys()))
    return Counter(letters)
c_val = get_counter(train_data_path)
c_train = get_counter(test_data_path)
letters_train = set(c_train.keys())
letters_val = set(c_val.keys())
if letters_train == letters_val:
    print('Letters in train and val do match')
else:
    raise Exception()
# print(len(letters_train), len(letters_val), len(letters_val | letters_train))
letters = sorted(list(letters_train))
print('Letters:', ' '.join(letters))

def labels_to_text(labels):
    return ''.join(list(map(lambda x: letters[int(x)], labels)))

def text_to_labels(text):
    return list(map(lambda x: letters.index(x), text))

def is_valid_str(s):
    for ch in s:
        if not ch in letters:
            return False
    return True

class TextImageGenerator:
    
    def __init__(self, 
                 dirpath, 
                 img_w, img_h, 
                 batch_size, 
                 downsample_factor,
                 max_text_len=8):
        
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.downsample_factor = downsample_factor
        
        img_dirpath = join(dirpath, 'img')
        ann_dirpath = join(dirpath, 'ann')
        self.samples = []
        for filename in os.listdir(img_dirpath):
            name, ext = os.path.splitext(filename)
            if ext == '.png':
                img_filepath = join(img_dirpath, filename)
                json_filepath = join(ann_dirpath, name + '.json')
                description = json.load(open(json_filepath, 'r'))['description']
                if is_valid_str(description):
                    self.samples.append([img_filepath, description])
        
        self.n = len(self.samples)
        self.indexes = list(range(self.n))
        self.cur_index = 0
        
    def build_data(self):
        self.imgs = np.zeros((self.n, self.img_h, self.img_w))
        self.texts = []
        for i, (img_filepath, text) in enumerate(self.samples):
            img = cv2.imread(img_filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (self.img_w, self.img_h))
            img = img.astype(np.float32)
            img /= 255
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            self.imgs[i, :, :] = img
            self.texts.append(text)
        
    def get_output_size(self):
        return len(letters) + 1
    
    def next_sample(self):
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]
    
    def next_batch(self):
        while True:
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            if K.image_data_format() == 'channels_first':
                X_data = np.ones([self.batch_size, 1, self.img_w, self.img_h])
            else:
                X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])
            Y_data = np.ones([self.batch_size, self.max_text_len])
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)
            label_length = np.zeros((self.batch_size, 1))
            source_str = []
                                   
            for i in range(self.batch_size):
                img, text = self.next_sample()
                img = img.T
                if K.image_data_format() == 'channels_first':
                    img = np.expand_dims(img, 0)
                else:
                    img = np.expand_dims(img, -1)
                X_data[i] = img
                Y_data[i] = text_to_labels(text)
                source_str.append(text)
                label_length[i] = len(text)
                
            inputs = {
                'the_input': X_data,
                'the_labels': Y_data,
                'input_length': input_length,
                'label_length': label_length,
                #'source_str': source_str
            }
            outputs = {'ctc': np.zeros([self.batch_size])}
            yield (inputs, outputs)
            
tiger = TextImageGenerator(train_data_path, 128, 64, 8, 4)
tiger.build_data()

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def train(img_w, load=False):
    # Input Parameters
    img_h = 64

    # Network parameters
    conv_filters = 16
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 32
    rnn_size = 512

    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)
        
    batch_size = 32
    downsample_factor = pool_size ** 2
    tiger_train = TextImageGenerator(train_data_path, img_w, img_h, batch_size, downsample_factor)
    tiger_train.build_data()
    tiger_val = TextImageGenerator(train_data_path, img_w, img_h, batch_size, downsample_factor)
    tiger_val.build_data()

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN:
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

    # Two layers of bidirecitonal GRUs
    # GRU seems to work as well, if not better than LSTM:
    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)

    # transforms RNN output to character activations:
    inner = Dense(tiger_train.get_output_size(), kernel_initializer='he_normal',
                  name='dense2')(concatenate([gru_2, gru_2b]))
    y_pred = Activation('softmax', name='softmax')(inner)
    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[tiger_train.max_text_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    if load:
        model = load_model('./tmp_model.h5', compile=False)
    else:
        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    
    if not load:
        # captures output of softmax so we can decode the output during visualization
        test_func = K.function([input_data], [y_pred])

        model.fit_generator(generator=tiger_train.next_batch(), 
                            steps_per_epoch=tiger_train.n,
                            epochs=1, 
                            validation_data=tiger_val.next_batch(), 
                            validation_steps=tiger_val.n)

    return model

model = train(128, load=False)

# For a real OCR application, this should be beam search with a dictionary
# and language model.  For this example, best path is sufficient.

def decode_batch(out):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(letters):
                outstr += letters[c]
        ret.append(outstr)
    return ret

tiger_test = TextImageGenerator('/home/ubuntu/PHDFITRI/codeLicencePlate/Licence-Plate-Recognition-Tensorflow-Keras/data/test/anpr_ocr/test/', 128, 64, 8, 4)
tiger_test.build_data()

net_inp = model.get_layer(name='the_input').input
net_out = model.get_layer(name='softmax').output

for inp_value, _ in tiger_test.next_batch():
    bs = inp_value['the_input'].shape[0]
    X_data = inp_value['the_input']
    net_out_value = sess.run(net_out, feed_dict={net_inp:X_data})
    pred_texts = decode_batch(net_out_value)
    labels = inp_value['the_labels']
    texts = []
    for label in labels:
        text = ''.join(list(map(lambda x: letters[int(x)], label)))
        texts.append(text)
    
    for i in range(bs):
        fig = plt.figure(figsize=(10, 10))
        outer = gridspec.GridSpec(2, 1, wspace=10, hspace=0.1)
        ax1 = plt.Subplot(fig, outer[0])
        fig.add_subplot(ax1)
        ax2 = plt.Subplot(fig, outer[1])
        fig.add_subplot(ax2)
        print('Predicted: %s\nTrue: %s' % (pred_texts[i], texts[i]))
        img = X_data[i][:, :, 0].T
        ax1.set_title('Input img')
        ax1.imshow(img, cmap='gray')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_title('Acrtivations')
        ax2.imshow(net_out_value[i].T, cmap='binary', interpolation='nearest')
        ax2.set_yticks(list(range(len(letters) + 1)))
        ax2.set_yticklabels(letters + ['blank'])
        ax2.grid(False)
        for h in np.arange(-0.5, len(letters) + 1 + 0.5, 1):
            ax2.axhline(h, linestyle='-', color='k', alpha=0.5, linewidth=1)
        
        #ax.axvline(x, linestyle='--', color='k')
        plt.show()
    break
'''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Activation
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import datetime
import time as time
from keras import backend as K
import re
import multiprocessing

K.clear_session()

output = True
save_to_dir = False

workers=multiprocessing.cpu_count()
verbose = 1
batch_size =16
epochs = 1000
image_width = 64
validation_split = 0.1

rotation_range=60
width_shift_range=0.1
height_shift_range=0.1
zoom_range=0.9

color_mode='rgb'
#color_mode='grayscale'
prediction_target_name = 'character'
metrics = ['acc']
loss = 'categorical_crossentropy'

tmp_data_path = '../input/tmp/'

#optimizer = optimizers.adadelta(lr=1.0, rho=0.95, epsilon=1e-24, decay=0.0)
#optimizer = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-24, decay=0, amsgrad=False)
optimizer = optimizers.rmsprop(lr=0.0001, decay=1e-6)
#optimizer = optimizers.sgd(lr=0.01, momentum=0, decay=0, nesterov=False)

filename = '../result/'
filename += datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename += '_'+optimizer.__class__.__name__
filename += '_'+str(batch_size)
filename += '_'+str(validation_split)
filename += '_'+color_mode
filename += '_'+str(rotation_range)
filename += '_'+str(width_shift_range)
filename += '_'+str(height_shift_range)
filename += '_'+str(zoom_range)

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, device_count={'GPU':1 , 'CPU':workers}))
keras.backend.set_session(sess)

class RestoreBestWeightsFinal(keras.callbacks.Callback):
    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 mode='auto',
                 baseline=None):
        super(RestoreBestWeightsFinal, self).__init__()
        
        self.monitor = monitor
        self.min_delta = min_delta
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
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = self.model.get_weights()
        
    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)

    def on_epoch_end(self, epoch, logs=None):
        val_current = logs.get(self.monitor)
        if val_current is None:
            return

        if self.monitor_op(val_current - self.min_delta, self.best):
            self.best = val_current
            self.best_weights = self.model.get_weights()
            print('best check point')
                
callbacks = []
callbacks.append(keras.callbacks.EarlyStopping(monitor='val_acc', patience=20))
callbacks.append(RestoreBestWeightsFinal(monitor='val_acc'))

filters = 32
model = Sequential()
model.add(Conv2D(filters=filters*1, kernel_size=3, padding='same', activation='relu',
                 input_shape=(image_width,image_width,1 if color_mode=='grayscale' else 3)))
model.add(Conv2D(filters=filters*1, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2), strides=2))
model.add(BatchNormalization(epsilon=1e-12))
model.add(Dropout(0.25))
model.add(Conv2D(filters=filters*2, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=filters*2, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2), strides=2))
model.add(BatchNormalization(epsilon=1e-12))
model.add(Dropout(0.25))
model.add(Conv2D(filters=filters*4, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=filters*4, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2), strides=2))
model.add(BatchNormalization(epsilon=1e-12))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, use_bias=False))
model.add(Activation('relu'))
model.add(BatchNormalization(epsilon=1e-12))
model.add(Dropout(0.5))
model.add(Dense(20, activation='softmax'))
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model.summary()

train_datagen = ImageDataGenerator( rescale=1. / 255,
                                    validation_split=validation_split)

train_datagen2 = ImageDataGenerator(rotation_range=rotation_range,
                                    width_shift_range=width_shift_range,
                                    height_shift_range=height_shift_range,
                                    zoom_range=(zoom_range, zoom_range),
                                    horizontal_flip=True,
                                    rescale=1. / 255,
                                    validation_split=validation_split)

train_generator = train_datagen.flow_from_directory(
                                    train_data_path,
                                    color_mode=color_mode,
                                    target_size=(image_width,image_width),
                                    batch_size=batch_size,
                                    save_to_dir=(tmp_data_path+'/train/') if save_to_dir else None,
                                    subset='training',
                                    shuffle=True)

train_generator.set_processing_attrs(
                                    train_datagen2,
                                    train_generator.target_size,
                                    train_generator.color_mode,
                                    train_generator.data_format,
                                    train_generator.save_to_dir,
                                    train_generator.save_prefix,
                                    train_generator.save_format,
                                    train_generator.subset,
                                    train_generator.interpolation)

valid_generator = train_datagen.flow_from_directory(
                                    train_data_path,
                                    color_mode=color_mode,
                                    target_size=(image_width, image_width),
                                    save_to_dir=(tmp_data_path+'/valid/') if save_to_dir else None,
                                    batch_size=batch_size,
                                    subset='validation',
                                    shuffle=True)

test_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
                                    test_data_path,
                                    color_mode=color_mode,
                                    target_size=(image_width,image_width),
                                    batch_size=1,
                                    class_mode=None,
                                    shuffle=False)

convert = lambda text: int(text) if text.isdigit() else text
alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
test_generator.filenames.sort(key=alphanum_key)

t = time.time()

model.fit_generator(train_generator,
                    steps_per_epoch = train_generator.samples // batch_size,
                    validation_data = valid_generator, 
                    validation_steps = (valid_generator.samples // batch_size),
                    epochs = epochs,
                    callbacks=callbacks,
                    verbose=verbose,
                    workers=workers)
    
Y_test = model.predict_generator(test_generator,
                                test_generator.samples,
                                verbose=verbose,
                                workers=workers)

elapsed = time.time() - t

try:
    history = history[0:restore_index-1]
    new_history = pd.DataFrame(model.history.history)
    new_history.index += restore_index
    history = history.append(new_history)
except NameError:
    history = pd.DataFrame(model.history.history)
    history.index += 1


Y_test = np.argmax(Y_test, axis=1)

characters_20 = {v: k for k, v in train_generator.class_indices.items()}

Y_test = pd.DataFrame([characters_20[i] for i in Y_test], columns=[prediction_target_name])
Y_test.index += 1

restore_index = np.argmax(history['val_acc'])

metrics.append('loss')

pdf = matplotlib.backends.backend_pdf.PdfPages(filename+'.pdf')
for metric in metrics:
    val_metric = 'val_'+metric
    title = '      ' + metric+'='+'%.6f' % history[metric].values[-1]
    title += '  ' + 'restored='+'%.6f' % history[metric].values[restore_index-1] + '(%s)' % restore_index
    title += '\n' + val_metric+'='+'%.6f' % history[val_metric].values[-1]
    title += '  ' + 'restored='+'%.6f' % history[val_metric].values[restore_index-1] + '(%s)' % restore_index
    ylim_top = max(history[metric].mean()+history[metric].std(), history[val_metric].mean()+history[val_metric].std())
    fig = plt.figure(figsize=(10,10));
    plt.plot(history[metric])
    plt.plot(history[val_metric])
    plt.title(title)
    plt.xlabel('Epoch '+'{:0>8}'.format(datetime.timedelta(seconds=int(elapsed))))
    plt.ylim(top=ylim_top)
    plt.xlim(left=1)
    plt.legend(['Train', 'valid'], loc='upper right')
    plt.grid()
    plt.show()
    if output:
        pdf.savefig(figure=fig)
pdf.close()

if output:
    Y_test.to_csv(filename+'.csv', index_label='id')
    history.to_csv(filename+'_history.csv', index_label='epoch')
    model_json = model.to_json()
    with open(filename+'.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(filename+'.h5')
    '''