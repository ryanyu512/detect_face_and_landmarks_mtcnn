import cv2
import json
from matplotlib import pyplot as plt

def visualise_labels(img_path):
    #load img
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) 
    #load label
    lab_path = (img_path.split('.'))[0] + '.json'

    #load label
    f = open(lab_path)
    lab = json.load(f)
    bbox = lab['bbox']
    cls = lab['class']
    
    #define bbox
    x_min = int(bbox[0]*img.shape[1])
    x_max = int(bbox[2]*img.shape[1])
    y_min = int(bbox[1]*img.shape[0])
    y_max = int(bbox[3]*img.shape[0])

    return cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=[0,0,255], thickness=1)

def visualise_boxes(img, kpoints, color = [0,0,255], thickness = 2, is_show_scores = False):
    for kpoint in kpoints:
        #define bbox
        x_min, y_min, x_max, y_max = kpoint[0:4]
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        cv2.rectangle(img, 
                      (x_min, y_min), 
                      (x_max, y_max),
                      color = color,
                      thickness = thickness)
        
        if len(kpoint) > 4 and is_show_scores:
            cv2.putText(img, 
                        str(round(kpoint[4], 2)), 
                        ((x_min + x_max)//2, 
                        y_max + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        color, 1) 
        
    return img