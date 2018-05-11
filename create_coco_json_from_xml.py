#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: rohit
Description: Convert XML annotations to COCO JSON format
"""

import json
from glob import glob
import xml.etree.ElementTree as ET
from pdb import set_trace as trace
import os

Classes = ['class1','class2']
Class_map = {k: v for v, k in enumerate(Classes)}

INPUT_DIRECTORY = "/home/rohit/annotation/"
IMAGE_DIRECTORY = "/home/rohit/Images/"
main_list = os.listdir(IMAGE_DIRECTORY)

# Read text file containing list of images to be considered
TXT_FILE = "/home/rohit/ImageSets/train.txt"
with open(TXT_FILE, 'r') as fr:
    image_list = fr.read().splitlines()
fr.close()
print "Number of images to be considered: ", len(image_list)

output_json = {}
output_json['type'] = 'instances'

annotations = []
images = []
categories = []
image_count = 201800000
annotation_id = 0
for filename in glob(INPUT_DIRECTORY + '*.xml'):
    if filename.split('/')[-1].split('.')[0] not in image_list:
        continue

    regex = filename.split('/')[-1].split('.')[0]
    imagename = filter(lambda x: regex in x, main_list)[0]
    tree = ET.parse(filename)
    root = tree.getroot()

    image_id = image_count

    """images dict"""
    image_dic = {}
    # If extension is missing from annotations file, get complete filename from 
    # ImageSets directory
    #imagename = root.find('filename').text
    for obj in root.findall('size'):
        height = obj.find('height').text
        width = obj.find('width').text
    image_dic['file_name'] = imagename
    image_dic['height'] = int(height)
    image_dic['width'] = int(width)
    image_dic['id'] = image_id
    images.append(image_dic)

    """annotations dict"""
    id_count = 1
    for obj in root.findall('object'):
        dic = {}
        name = obj.find('name').text
        bbox = []
        iscrowd = 0
        ignore = 0
        category_id = Class_map[name] + 1
        id = id_count
        id_count += 1
        for box in obj.findall('bndbox'):
            xmin = int(box.find('xmin').text)
            xmax = int(box.find('xmax').text)
            ymin = int(box.find('ymin').text)
            ymax = int(box.find('ymax').text)
            width = xmax-xmin
            height = ymax-ymin
            area = width * height
            bbox = [xmin, ymin, width, height]
            segmentation = [[xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]]
        #print name, bbox, area, iscrowd, ignore, category_id, id, image_id, segmentation
        dic['segmentation'] = segmentation
        dic['area'] = area
        dic['bbox'] = bbox
        dic['iscrowd'] = iscrowd
        dic['category_id'] = category_id
        annotation_id += 1
        dic['id'] = annotation_id#id
        dic['image_id'] = image_id
        dic['ignore'] = ignore
        annotations.append(dic)

    # Increment image count
    image_count += 1

""" Parse Categories """
for key, value in Class_map.iteritems():
    dic = {}
    dic['name'] = key
    dic['id'] = value + 1
    dic['supercategory'] = 'none'
    categories.append(dic)

output_json['categories'] = categories
output_json['annotations'] = annotations
output_json['images'] = images

with open('output.json', 'w') as fw:
    json.dump(output_json, fw)
fw.close()
