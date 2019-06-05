#%% import
import os
import sys
import re
import datetime
import random
import numpy as np
import pandas as pd
import time as time
from ast import literal_eval
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import multiprocessing
import tensorflow as tf
import keras.backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model, load_model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

sys.path.append(os.path.abspath('../input/library'))
from yolo_model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss, yolo_eval
from yolo_utils import get_random_data, letterbox_image

from PIL import Image, ImageDraw
import cv2

#%% session config
K.clear_session()
workers=multiprocessing.cpu_count()
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, device_count={'GPU':1 , 'CPU':workers}))
K.set_session(sess)
#%% basic parameter
annotation_path = '../input/database/train.txt'
test_path = '../input/ntut-ml-2019-computer-vision/test/data/test'
output_prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_plate_")
output_model_path = output_prefix + 'trained_weights_final.h5'

class_names = ['plate']
num_classes = len(class_names)
anchors = [47, 20,  58, 26,  73, 26,  76, 33,  86, 41, 114, 53]
anchors = np.array(anchors).reshape(-1, 2)

input_shape = (256,320) # multiple of 32, hw
val_split = 0.1

epochs = 5
batch_size = 8
verbose = 1
img_w = 128
img_h = 64
#%% functions
from collections import Counter
def get_counter(dirpath):
    dirname = os.path.basename(dirpath)
    ann_dirpath = os.path.join(dirpath, 'ann')
    letters = ''
    lens = []
    for filename in os.listdir(ann_dirpath):
        json_filepath = os.path.join(ann_dirpath, filename)
        description = os.path.join.load(open(json_filepath, 'r'))['description']
        lens.append(len(description))
        letters += description
    print('Max plate length in "%s":' % dirname, max(Counter(lens).keys()))
    return Counter(letters)
c_val = get_counter('/home/ubuntu/PHDFITRI/codeLicencePlate/Licence-Plate-Recognition-Tensorflow-Keras/data/val/anpr_ocr/train/')
c_train = get_counter('/home/ubuntu/PHDFITRI/codeLicencePlate/Licence-Plate-Recognition-Tensorflow-Keras/data/train/anpr_ocr/train/')
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
        
        img_dirpath = os.path.join(dirpath, 'img')
        ann_dirpath = os.path.join(dirpath, 'ann')
        self.samples = []
        for filename in os.listdir(img_dirpath):
            name, ext = os.path.splitext(filename)
            if ext == '.png':
                img_filepath = os.path.join(img_dirpath, filename)
                json_filepath = os.path.join(ann_dirpath, name + '.json')
                description = os.path.join.load(open(json_filepath, 'r'))['description']
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
tiger = TextImageGenerator('/home/ubuntu/PHDFITRI/codeLicencePlate/Licence-Plate-Recognition-Tensorflow-Keras/data/val/anpr_ocr/train/', 128, 64, 8, 4)
tiger.build_data()
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
#%% create model
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
tiger_train = TextImageGenerator('/home/ubuntu/PHDFITRI/codeLicencePlate/Licence-Plate-Recognition-Tensorflow-Keras/data/train/anpr_ocr/train/', img_w, img_h, batch_size, downsample_factor)
tiger_train.build_data()
tiger_val = TextImageGenerator('/home/ubuntu/PHDFITRI/codeLicencePlate/Licence-Plate-Recognition-Tensorflow-Keras/data/val/anpr_ocr/train/', img_w, img_h, batch_size, downsample_factor)
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

model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

#%% train
t = time.time()
# captures output of softmax so we can decode the output during visualization
test_func = K.function([input_data], [y_pred])

model.fit_generator(generator=tiger_train.next_batch(), 
                    steps_per_epoch=tiger_train.n,
                    epochs=1, 
                    validation_data=tiger_val.next_batch(), 
                    validation_steps=tiger_val.n)

model.save_weights(output_model_path)
elapsed = time.time() - t

#%% predict
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
y_test = pd.DataFrame(y_test, columns=['top', 'left', 'bottom', 'right', 'score', 'class'])
y_test.index += 1
y_test.to_csv(output_prefix+'.csv', index_label='ID')

a = pd.read_csv(output_prefix+'.csv')
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
plt.ylim(top=top)
plt.legend(['Train', 'valid'], loc='upper right')
plt.grid()
plt.show()
pdf.savefig(figure=fig)

pdf.close()