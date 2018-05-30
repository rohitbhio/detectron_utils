#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 12:21:13 2018

@author: rohit.bhiogade

Script to convert labelme annotations to deeplab format
"""

import numpy as np
import cv2
from glob import glob
import json
from PIL import Image, ImageDraw
import lxml.etree
import lxml.builder

#Global Variables
INPUT_DIRECTORY_IM = "/home/rohit/Images/"
INPUT_DIRECTORY_ANN = "/home/rohit/Annotations/" #Labelme format
OUTPUT_DIRECTORY_IM = "/home/rohit/Segmented_Image/" #Mask
OUTPUT_DIRECTORY_ANN = "/home/rohit/Annotation/" #In Pascal VOC format

#Iterate through all images
for filename in glob(INPUT_DIRECTORY_IM + '*.jpg'):
    print filename
    
    #Read Input Image
    img = cv2.imread(filename)
    width, height = img.shape[:2]
    
    #Read Annotation File
    with open(INPUT_DIRECTORY_ANN + filename.split('/')[-1].split('.')[0] + '.json', 'r') as f:
        ann = f.read()
    f.close()
    jason = json.loads(ann)
    
    #Define DOM
    E = lxml.builder.ElementMaker()
    ROOT = E.annotation
    size = E.size(E.width(str(width)), E.height(str(height)), E.depth('3'),)
    the_doc = ROOT(
            E.folder(''),
            E.filename(filename.split('/')[-1]),
            E.source(
                    E.database('The DB'),
                    E.annotation(''),
                    E.image(''),
                ),
            E.segmented('0'),
            size,
            )   
    
    #Iterate through Polygons
    blank_img = Image.new('L', (height, width), 0)
    for dots in jason['shapes']:
        poly = dots['points']
        label = dots['label']
        
        x1, x2, y1, y2 = None, None, None, None
        for p in poly:
            points = np.reshape(np.array(p), (int(len(p)/2), 2))
            
            if x1 is None: 
                x1, y1 = points.min(0)
                x2, y2 = points.max(0)
            else:
                if points.min(0)[0]<x1:
                    x1 = points.min(0)[0]
                if points.min(0)[1]<y1:
                    y1 = points.min(0)[1]
                if points.max(0)[0]>x2:
                    x2 = points.max(0)[0]
                if points.max(0)[1]>y2:
                    y2 = points.max(0)[1]
                    
        res = list(map(tuple, poly))
        ImageDraw.Draw(blank_img).polygon(res, outline=1, fill=1)
        mask = np.array(blank_img)
        
        bbox = E.bndbox(E.xmin(str(max(x1,0))), E.ymin(str(max(y1,0))), E.xmax(str(min(x2,width))), E.ymax(str(min(y2,height))))
        objec = E.object(E.name(label), E.pose('Unspecified'), E.truncated('0'), E.difficult('0'), bbox)
        the_doc.append(objec)
        
    mask[mask>0] = 255
    
    #Save Mask
    cv2.imwrite(OUTPUT_DIRECTORY_IM + filename.split('/')[-1], mask)
    
    #Save Annotation XML
    et = lxml.etree.ElementTree(the_doc)
    et.write(OUTPUT_DIRECTORY_ANN + filename.split('/')[-1].split('.')[0] + '.xml', pretty_print=True, xml_declaration=True, encoding="utf-8")
