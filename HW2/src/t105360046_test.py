import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties
import scipy
import time
import pandas as pd
import random
np.random.seed(2)

from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.models import model_from_json

import warnings

warnings.filterwarnings('ignore')



dict_characters = {0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon', 2: 'bart_simpson', 
        3: 'charles_montgomery_burns', 4: 'chief_wiggum', 5: 'comic_book_guy', 6: 'edna_krabappel', 
        7: 'homer_simpson', 8: 'kent_brockman', 9: 'krusty_the_clown', 10: 'lenny_leonard', 11:'lisa_simpson',
        12: 'marge_simpson', 13: 'mayor_quimby',14:'milhouse_van_houten', 15: 'moe_szyslak', 
        16: 'ned_flanders', 17: 'nelson_muntz', 18: 'principal_skinner', 19: 'sideshow_bob'}

pic_size = 64
batch_size = 128
epochs = 100
num_classes = len(dict_characters)
pictures_per_class = 1000
test_size = 0.15

# Load the data
def load_train_set(dirname,dict_characters):
    X_train = []
    Y_train = []
    for label,character in dict_characters.items():
        list_images = os.listdir(dirname+'/'+character)
        for image_name in list_images:
            image = scipy.misc.imread(dirname+'/'+character+'/'+image_name)
            X_train.append(scipy.misc.imresize(image,(64,64),interp='lanczos'))
            Y_train.append(label)
    return np.array(X_train), np.array(Y_train)


# load the test data
def load_test_set(dirname,dict_characters):
    X_train = []
    Y_train = []
    list_images = os.listdir(dirname)
    list_images.sort(key=lambda x:int(x[:-4]))
    for image_name in list_images:
        image = scipy.misc.imread(dirname+'/'+image_name)
        X_train.append(scipy.misc.imresize(image,(64,64),interp='lanczos'))

    return np.array(X_train), np.array(Y_train)


X_train, Y_train = load_train_set("../input/train/characters-20/", dict_characters)       
X_test, Y_test = load_test_set("../input/test/", dict_characters) 





# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0


def display_samples(samples_index,imgs,obs, preds_classes=None,preds=None):
    """This function randomly displays 20 images with their observed labels 
    and their predicted ones(if preds_classes and preds are provided)"""
    n = 0
    nrows = 4
    ncols = 5
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize=(12,10))
    plt.subplots_adjust(wspace=0, hspace=0)
    for row in range(nrows):
        for col in range(ncols):
            index = samples_index[n]
            ax[row,col].imshow(imgs[index])
            
            actual_label = dict_characters[obs[index]].split("_")[0]
            actual_text = "Actual : {}".format(actual_label)
            
            ax[row,col].add_patch(patches.Rectangle((0, 53),64,25,color='white'))
            font0 = FontProperties()
            font = font0.copy()
            font.set_family("fantasy")
            
            ax[row,col].text(1, 54, actual_text , horizontalalignment='left', fontproperties=font,
                    verticalalignment='top',fontsize=10, color='black',fontweight='bold')
            
            if preds_classes != None and preds != None:
                predicted_label = dict_characters[preds_classes[index]].split('_')[0]
                predicted_proba = max(preds[index])*100
                predicted_text = "{} : {:.0f}%".format(predicted_label,predicted_proba)
            
                ax[row,col].text(1, 59, predicted_text , horizontalalignment='left', fontproperties=font,
                    verticalalignment='top',fontsize=10, color='black',fontweight='bold')
            n += 1


def pick_up_random_element(elem_type,array):
    """This function randomly picks up one element per type in the array"""
    return int(random.choice(np.argwhere(array == elem_type)))

samples = [pick_up_random_element(elem_type,Y_train) for elem_type in range(20)]

display_samples(samples,X_train,Y_train)




# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_train = to_categorical(Y_train, num_classes = 20)
Y_test = to_categorical(Y_test, num_classes = 20)



#X_valid=X_train[range(0,3000)]
#Y_valid=Y_train[range(0,3000)]


#for i in range(0,3000):
#    a=random.randrange(0,len(X_train)-1)
#    X_valid[i]=X_train[a]
#    Y_valid[i]=Y_train[a]
#    X_train=np.delete(X_train,a,axis = 0)
#    Y_train=np.delete(Y_train,a,axis = 0)
    
    
    

# Set the CNN model 
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

# Define the optimizer
optimizer = RMSprop(lr=0.001, decay=1e-6)
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu', input_shape = (64,64,3)))
model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(filters = 86, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 86, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Flatten())
#model.add(Dense(1024, activation = "relu"))
#model.add(Dropout(0.5))
model.add(Dense(512, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(20, activation = "softmax"))
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])



filepath="weights_8conv_%s.hdf5" % time.strftime("%Y%m%d") 
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')


history = model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1,
          verbose=1,
          shuffle=True,
          )

y_preds = model.predict(X_test)


df_sampleSubmission = pd.read_csv('/home/yan/learning/hw2/data/sampleSubmission.csv', index_col=0)


y_preds_classes = np.argmax(y_preds,axis = 1)
y_preds_true=df_sampleSubmission


for num in range(0,990):
    temp=y_preds_classes[num]
    y_preds_true.character[num+1]=dict_characters[temp]

    