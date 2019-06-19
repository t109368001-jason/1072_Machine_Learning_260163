import xml.etree.ElementTree as ET
from os import getcwd
import os

classes = ['plate']


def convert_annotation(image_id, list_file):
    in_file = open('../input/ntut-ml-2019-computer-vision/train/data/train/%s.xml'%(image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd()

image_ids = []

for filename in os.listdir('../input/ntut-ml-2019-computer-vision/train/data/train/'):
    if not '.xml' in filename:
        continue
    if '._' in filename:
        continue
    fileName, fileExtension = os.path.splitext(filename)
    image_ids.append(fileName)

list_file = open('../input/database/train.txt', 'w')
for image_id in image_ids:
    list_file.write('../input/ntut-ml-2019-computer-vision/train/data/train/%s.jpg'%(image_id))
    convert_annotation(image_id, list_file)
    list_file.write('\n')
list_file.close()

