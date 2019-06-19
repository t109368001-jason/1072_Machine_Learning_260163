#%% import
import os
from os.path import join
import sys
import re
import datetime
import random
import itertools
import numpy as np
import pandas as pd
import time as time
from ast import literal_eval
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.backends.backend_pdf
import multiprocessing
import tensorflow as tf
import keras.backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, Dropout
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, load_model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.optimizers import Adam, Adadelta
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback

sys.path.append(os.path.abspath('../input/library'))
from yolo_model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss, yolo_eval
from yolo_utils import get_random_data, letterbox_image

from PIL import Image, ImageDraw, ImageOps

#%% session config
K.clear_session()
workers=multiprocessing.cpu_count()
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, device_count={'GPU':1 , 'CPU':workers}))
K.set_session(sess)

#%% basic parameter
train_list = '../input/database/train.txt'
test_list = '../input/database/20190611_110849_test.txt'
output_prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_plate_")
output_model_path = output_prefix + 'trained_weights_final.h5'

class_names = ['plate']
num_classes = len(class_names)
anchors = [47, 20,  58, 26,  73, 26,  76, 33,  86, 41, 114, 53]
anchors = np.array(anchors).reshape(-1, 2)

input_shape = (128,64) # multiple of 32, hw
val_split = 0.1

epochs = 500
batch_size =512
verbose = 2
img_w = 128
img_h = 64

with open(test_list) as f:
    lines_test = f.readlines()
#lines_test = lines_test[:100]

with open(train_list) as f:
    lines = f.readlines()

np.random.shuffle(lines)
#lines = lines[:1000]
np.random.seed(None)
num_val = int(len(lines)*val_split)
num_train = len(lines) - num_val

#%% functions
max_text_len = max([(len(x.split()[0].split('/')[-1].split('_')[0])) for x in lines])

letters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z', ' ']

def labels_to_text(labels):
    return ''.join(list(map(lambda x: letters[int(x)], labels)))

def text_to_labels(text):
    return list(map(lambda x: letters.index(x), text))

def is_valid_str(s):
    for ch in s:
        if not ch in letters:
            return False
    return True

def text_to_img_number(text, input_shape, max_text_len, is_test_data=False):
    ss = text.split()
    img_path = ss[0]
    img = Image.open(img_path).convert('L')
    if len(ss) < 2:
        number = ' '*max_text_len
    else:
        box_class = ss[1].split(',')
        left, top, right, bottom = box_class[:4]
        number = img_path.split('/')[-1].split('_')[0]
        img = img.resize((320, 256), Image.ANTIALIAS)
        img = img.crop((int(left), int(top), int(right), int(bottom)))
    img = img.resize(input_shape, Image.ANTIALIAS)
    img = np.array(img, dtype='float32')
    img /= 255
    if len(number) < max_text_len:
        number += ' '*(max_text_len-len(number))
    if is_test_data:
        number = ' '*max_text_len
    return img, number

class TextImageGenerator:
    
    def __init__(self, 
                 lines, 
                 input_shape,
                 batch_size, 
                 downsample_factor,
                 max_text_len,
                 is_test_data=False):
        
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.is_test_data = is_test_data
        self.downsample_factor = downsample_factor
        self.samples = lines
        self.n = len(self.samples)
        self.indexes = list(range(self.n))
        self.cur_index = self.n
        
    def build_data(self):
        self.imgs = np.zeros((self.n, self.input_shape[1], self.input_shape[0]))
        self.texts = []
        for i, (text) in enumerate(self.samples):
            img, number = text_to_img_number(text, self.input_shape, self.max_text_len, self.is_test_data)
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            self.imgs[i, :, :] = img
            self.texts.append(number)
            #print('%d/%d %s'%(i, len(self.samples), number), end=' ', flush=True)
        
    def get_output_size(self):
        return len(letters) + 1
    
    def next_sample(self):
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            if not self.is_test_data:
                random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]
    
    def next_batch(self):
        while True:
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            if K.image_data_format() == 'channels_first':
                X_data = np.ones([self.batch_size, 1, self.input_shape[0], self.input_shape[1]])
            else:
                X_data = np.ones([self.batch_size, self.input_shape[0], self.input_shape[1], 1])
            Y_data = np.ones([self.batch_size, self.max_text_len])
            input_length = np.ones((self.batch_size, 1)) * (self.input_shape[0] // self.downsample_factor - 2)
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
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
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
class RestoreBestWeightsFinal(Callback):
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
                
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=verbose)
early_stop = EarlyStopping(monitor='loss', min_delta=0., patience=30, mode='min', verbose=verbose)
restore_best_weights = RestoreBestWeightsFinal(monitor='val_loss')

#%% load train data
tiger_train = TextImageGenerator(lines[:num_train], input_shape, batch_size, 4, max_text_len)
tiger_valid = TextImageGenerator(lines[num_train:], input_shape, min(batch_size, num_val), 4, max_text_len)
print('Building tiger_train')
tiger_train.build_data()
print('Building tiger_valid')
tiger_valid.build_data()
'''#check data
for inp, out in tiger_train.next_batch():
    print('Text generator output (data which will be fed into the neutral network):')
    print('1) the_input (image)')
    if K.image_data_format() == 'channels_first':
        img = inp['the_input'][0, 0, :, :]
    else:
        img = inp['the_input'][0, :, :, 0]
    
    plt.imshow(img.T, cmap='gray')
    plt.show()
    print('2) the_labels (plate number): %s is encoded as %s' % 
          (labels_to_text(inp['the_labels'][0]), list(map(int, inp['the_labels'][0]))))
    print('3) input_length (width of image that is fed to the loss function): %d == %d / 4 - 2' % 
          (inp['input_length'][0], tiger_train.input_shape[0]))
    print('4) label_length (length of plate number): %d' % inp['label_length'][0])
    break
'''

#%% create model
# Network parameters
def create_model():
    conv_filters = 128
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 1024
    rnn_size = 1024
    
    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)
        
    #act = 'relu'
    act = LeakyReLU(alpha=0.1)
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Dropout(0.75)(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)
    inner = Dropout(0.75)(inner)
    
    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)
    
    # cuts down input size going into RNN:
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)
    inner = Dropout(0.75)(inner)
    
    # Two layers of bidirecitonal GRUs
    # GRU seems to work as well, if not better than LSTM:
    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)
    gru1_merged = add([gru_1, gru_1b])
    gru1_merged = Dropout(0.75)(gru1_merged)
    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)
    
    # transforms RNN output to character activations:
    inner = Dense(len(letters) + 1, kernel_initializer='he_normal',
                  name='dense2')(concatenate([gru_2, gru_2b]))
    #inner = Dropout(0.5)(inner)
    y_pred = Activation('softmax', name='softmax')(inner)
    Model(inputs=input_data, outputs=y_pred).summary()
    
    labels = Input(name='the_labels', shape=[max_text_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=1)
    adam = Adam(lr=1e-4, epsilon=1e-12)
    adadelta = Adadelta(lr=1.0)
    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    
    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adadelta)
    return model

model = create_model()

#%% train
t = time.time()

print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
model.fit_generator(generator=tiger_train.next_batch(),
                    steps_per_epoch=int(tiger_train.n / batch_size),
                    epochs=epochs,
                    callbacks=[early_stop, restore_best_weights],
                    validation_data=tiger_valid.next_batch(),
                    validation_steps=int(tiger_valid.n / min(batch_size, num_val)),
                    verbose=verbose)
model.save_weights(output_model_path)
elapsed = time.time() - t


#%% predict
tiger_test = TextImageGenerator(lines_test, input_shape, 1, 4, max_text_len, True)
print('Building tiger_test')
tiger_test.build_data()

y_test = []
net_inp = model.get_layer(name='the_input').input
net_out = model.get_layer(name='softmax').output
i = 0
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
    number = pred_texts[0].replace(" ", "")
    y_test.append(number)
    '''
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
    '''
    i += 1
    if i == 10000:
        break

y_test = pd.DataFrame(y_test, columns=['Number'])
y_test.index += 1
y_test.to_csv(output_prefix+'.csv', index_label='ID')


#%% plot

pdf = matplotlib.backends.backend_pdf.PdfPages(output_prefix+'.pdf')

fig = plt.figure(figsize=(10,10));
plt.plot(model.history.epoch, model.history.history['loss'],
         model.history.epoch, model.history.history['val_loss'])
plt.title('Loss')
plt.xlabel('Epoch '+ str(datetime.timedelta(seconds=elapsed)))
#plt.ylim(top=ylim_top)
plt.legend(['Train', 'valid'], loc='upper right')
plt.grid()
plt.show()
pdf.savefig(figure=fig)
#==============================================================================
fig = plt.figure(figsize=(10,10));
plt.plot(model.history.epoch, model.history.history['loss'],
         model.history.epoch, model.history.history['val_loss'])
plt.title('Loss')
plt.xlabel('Epoch '+ str(datetime.timedelta(seconds=elapsed)))
i25 = int(len(model.history.history['loss'])*0.25)+1
a = model.history.history['loss']
b = model.history.history['val_loss']
a.sort()
b.sort()
top = max(a[-i25], b[-i25])
plt.ylim(0, top)
plt.legend(['Train', 'valid'], loc='upper right')
plt.grid()
plt.show()
pdf.savefig(figure=fig)

pdf.close()
