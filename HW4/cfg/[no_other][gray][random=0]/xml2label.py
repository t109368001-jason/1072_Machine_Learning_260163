import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

py_path = os.path.abspath(__file__)[:-13]

cfg_name = py_path.split('/')[-1]
ann_path = os.path.abspath(os.path.join(py_path, '../../input/train/ann'))
img_path = os.path.abspath(os.path.join(py_path, '../../input/train/JPEGImages'))
labels_path = os.path.abspath(os.path.join(py_path, '../../input/train/labels'))
train_txt_path = os.path.abspath(os.path.join(py_path, 'train.txt'))
data_path = os.path.abspath(os.path.join(py_path, 'my.data'))
names_path = os.path.abspath(os.path.join(py_path, 'my.names'))

classes = []

with open(names_path) as names_file:
    lines = names_file.readlines()
    for line in lines:
        classes.append(line.strip())

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(image_id):
    in_file = open('%s/%s.xml'%(ann_path, image_id))
    out_file = open('%s/%s.txt'%(labels_path, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()

if not os.path.exists(labels_path):
    os.makedirs(labels_path)
#image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
image_ids = os.listdir(ann_path)
image_ids = [i.split('.')[0] for i in image_ids]
list_file = open(train_txt_path, 'w')
for image_id in image_ids:
    list_file.write('%s/%s.jpg\n'%(img_path, image_id))
    convert_annotation(image_id)
list_file.close()

data_file = open(data_path, 'w')

data_file.write('classes= %s\n'%(len(classes)))
data_file.write('train  = %s\n'%(train_txt_path))
data_file.write('names  = %s\n'%(names_path))
data_file.write('backup  = %s\n'%(py_path))
data_file.close()
