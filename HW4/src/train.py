#%% import
import os
import sys
import re
import datetime
import numpy as np
import pandas as pd
import time as time
from ast import literal_eval
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import multiprocessing
import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model, load_model, save_model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

sys.path.append(os.path.abspath('../input/library'))
from yolo_model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss, yolo_eval
from yolo_utils import get_random_data, letterbox_image

from PIL import Image, ImageDraw

#%% session config
K.clear_session()
workers=multiprocessing.cpu_count()
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, device_count={'GPU':1 , 'CPU':workers}))
K.set_session(sess)
#%% basic parameter
annotation_path = '../input/database/train.txt'
test_path = '../input/ntut-ml-2019-computer-vision/test/data/test'
output_prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")
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
#%% functions
def create_tiny_model(input_shape, anchors, num_classes):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def create_tiny_test_model(model_path):
    #model = load_model(model_path)
    image_input = Input(shape=(None, None, 3))
    model = tiny_yolo_body(image_input, len(anchors)//2, num_classes)
    model.load_weights(model_path) # make sure model, anchors and classes match
    return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=False)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)
#%% init model
model = create_tiny_model(input_shape, anchors, num_classes)
#logging = TensorBoard(log_dir=log_dir)
#checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5', monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

with open(annotation_path) as f:
    lines = f.readlines()
    
test_files = []
for filename in os.listdir(test_path):
    if '.xml' in filename:
        continue
    if '._' in filename:
        continue
    test_files.append(os.path.join(test_path, filename))

convert = lambda text: int(text) if text.isdigit() else text
alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
test_files.sort(key=alphanum_key)

np.random.shuffle(lines)
#lines = lines[:100]
#test_files = test_files[:10]
np.random.seed(None)
num_val = int(len(lines)*val_split)
num_train = len(lines) - num_val

# Unfreeze and continue training, to fine-tune.
# Train longer if the result is not good.
for i in range(len(model.layers)):
    model.layers[i].trainable = True
model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
print('Unfreeze all of the layers.')
# note that more GPU memory is required after unfreezing the body
print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
#%% train
t = time.time()
model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
    steps_per_epoch=max(1, num_train//batch_size),
    validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
    validation_steps=max(1, num_val//batch_size),
    epochs=epochs,
    initial_epoch=0,
    callbacks=[reduce_lr, early_stopping],
    verbose=verbose,
    workers=workers)

model.save_weights(output_model_path)
elapsed = time.time() - t

#%% predict
yolo_model = create_tiny_test_model(output_model_path)
input_image_shape = K.placeholder(shape=(2, ))
boxes, scores, classes = yolo_eval(yolo_model.output, anchors,
                len(class_names), input_image_shape,
                score_threshold=0.3, iou_threshold=0.45)
sess = K.get_session()

y_test = []
for test_file in test_files:
    image = Image.open(test_file)
    if input_shape != (None, None):
        assert input_shape[0]%32 == 0, 'Multiples of 32 required'
        assert input_shape[1]%32 == 0, 'Multiples of 32 required'
        boxed_image = letterbox_image(image, tuple(reversed(input_shape)))
    else:
        new_image_size = (image.width - (image.width % 32),
                          image.height - (image.height % 32))
        boxed_image = letterbox_image(image, new_image_size)
    image_data = np.array(boxed_image, dtype='float32')
    
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    
    out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict={
                    yolo_model.input: image_data,
                    input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0
                })
    if len(out_scores) == 0:
        y_test.append([0, 0, 256, 320, 0, 0])
    else:
        top, left, bottom, right = out_boxes[0]
        y_test.append([top, left, bottom, right, out_scores[0], out_classes[0]])
        '''draw = ImageDraw.Draw(image)
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        draw.rectangle([left, top, right, bottom],
                       outline=(255,255,255))'''

test_txt = open(output_prefix+'test.txt', 'w')
for i, (test_file) in enumerate(test_files):
    top, left, bottom, right, score, class_name = y_test[i]
    test_txt.write('%s %d,%d,%d,%d,%d,%d\n'%(test_file, left, top, right, bottom, score, class_name))
test_txt.close()

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