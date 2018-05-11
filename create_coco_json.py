"""
Script to convert polygon annotations generated using Labelme to COCO format
"""

import glob
import os
import numpy as np
import json
import cv2

image_dir = "/home/rohit/Image_folder/"
file_path = "/home/rohit/output_polygon.json"
class_file = "/home/rohit/classes.txt"

if __name__=='__main__':

    with open(class_file, 'r') as f:
        classes = [x.strip() for x in f.readlines()]
           
    images, anns, categories = [], [], []
    id = 1
    class_dic = {}
    for label in classes:
        dic = {'name':label, 'id':id, "supercategory": "none"}
        class_dic[label] = id
        categories.append(dic)
        id += 1
    
    ann_index = 0
    for i, f in enumerate(sorted(glob.glob(os.path.join(os.path.abspath(image_dir), '*.json')))):
        
        with open(f, 'r') as g:
            s = g.read()
        jason = json.loads(s)
        
        ## Image Properties
        img = cv2.imread(f.split('.')[0] + '.jpg')
        width, height = img.shape[:2]
        dic = {'file_name': jason['imagePath'], 'id': i+2018000000, 'height': height, 'width': width}
        images.append(dic)
        
        for dots in jason['shapes']:
            poly = dots['points']
            cat_id = class_dic[dots['label']]
            
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
                        
            contour = np.around(np.array([[p] for p in poly])).astype(np.int32)
            area = cv2.contourArea(contour)
            
            result = [np.array(sum(poly, [])).clip(min = 0).tolist()]

            bbox = [max(x2,0), max(y2,0), max(x1,0), max(y1,0)]
            ann_index+=1
            dic2 = {'segmentation': result, 'area': np.abs(area), 'iscrowd':0, 'image_id':i+2018000000, 'bbox':bbox, 'category_id': cat_id, 'id': ann_index}
            anns.append(dic2)
              
    data = {'images':images, 'annotations':anns, 'categories':categories, 'classes': classes, 'type' : 'instances'}

    with open(file_path, 'w') as outfile:
        json.dump(data, outfile)
