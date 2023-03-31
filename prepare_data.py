'''
UPDATE ON 2023/03/16

1. provide libraries to prepare FDDB data for training, validation and testing
'''

import os
import cv2
import math
import json
import random
import numpy as np

from marco import *
from train_model import *
from matplotlib import pyplot as plt

def gather_uid(dirs):
    '''
    args:
        dirs: directory containing target files
    
    returns:
        uid: unique identification
    '''
    
    data_list = os.listdir(dirs)
    
    uid = []
    for i in range(len(data_list)):
        file_type = data_list[i].split('.')[-1]
        t = data_list[i].split('.')[0]
        uid.append(t)

    uid = sorted(list(set(uid)))

    return uid

def split_ids(data, train=0.9, valid=0.1, test=0, seed = 0):
    """
    args:
       data : list of data paths
       train: train split size (between 0 - 1)
       valid: valid split size (between 0 - 1)
       test : test split size (between 0 - 1)
       seed : random seed
        
    returns:
        train_set: list of training data paths
        valid_set: list of validation data paths
         test_set: list of testing data paths
    """
    
    if train + valid + test != 1:
        return 
    
    list_copy = list(range(0, len(data)))
    random.Random(seed).shuffle(list_copy)
    
    #obtain the size of training, validation and testing data
    train_size = math.floor(len(list_copy) * train)
    valid_size = math.floor(len(list_copy) * valid)
    test_size  = len(list_copy) - train_size - valid_size
    
    train_set = [None]*train_size
    if train + valid == 1.0:
        valid_size += test_size 
        valid_set = [None]*valid_size
        test_set  = None
        test_size = 0
    else:
        valid_set = [None]*valid_size
        test_set  = [None]*test_size
    
    #split the data into training, validation and testing dataset
    idx = 0
    for i, rand_ind in enumerate(list_copy):
        
        if i == train_size or i == train_size + valid_size:
            idx = 0
            
        if i < train_size:
            train_set[idx]= data[rand_ind]
        elif i >= train_size and i < train_size + valid_size:
            valid_set[idx]= data[rand_ind]
        else:
            test_set[idx] = data[rand_ind]
        idx += 1
        
    return train_set, valid_set, test_set

def covert_ellipse_to_box(e_bb):
    '''
    args:
        e_bb : ellipse bounding box (major axis, minor axis, angle, center_x, center_y)
        img_w: image width
        img_h: image_height
        
    returns:
        x1, y1, x2, y2: top_x, top_y, bottom_x, bottom_y of rectangle bounding box
    '''

    a, b, ang, xc, yc = e_bb[0:5]
    
    asq = a**2
    bsq = b**2
    csq = np.cos(ang)**2
    ssq = np.sin(ang)**2

    x1 = -np.sqrt(asq*csq + bsq*ssq)
    y1 =  np.sqrt(asq*ssq + bsq*csq)
    x2 =  np.sqrt(asq*csq + bsq*ssq)
    y2 = -np.sqrt(asq*ssq + bsq*csq)
    
    x1, x2 =  x1 + xc,  x2 + xc
    y1, y2 = -y1 + yc, -y2 + yc
    
    return [x1, y1, x2, y2]

def get_FDDB_data_and_labels(IMG_DIR, LAB_DIR):
    '''
        args:
            dir: directory that contains target files
        
        returns:
            all_data: a list of data dict containing {data_path, data_num, bboxes}
                      data_path => relative path of file
                      data_num  => number of bounding boxes
                      boxes     => coordinates of bounding boxes
    '''
    
    all_data = []
    #original FDDB anntation is divided into 10 folds 
    for fold in range(1, 11):
        if fold < 10:
            fold = '0' + str(fold)
        lab_path = os.path.join(LAB_DIR, f'FDDB-fold-{fold}-ellipseList.txt')
        
        f = open(lab_path, 'r')
        lines = f.readlines()
        cnt = 0
        while True:
            data_dict = {}
            data_dict['data_path'] = os.path.join(IMG_DIR ,lines[cnt].replace('\n',''))
            data_dict['data_num']  = int(lines[cnt + 1].replace('\n',''))
            img = cv2.imread(data_dict['data_path'] + '.jpg')
            bboxes = []
            #extract box data
            for i in range(cnt + 2, cnt + 2 + data_dict['data_num']):
                #convert ellipse bbox into rectangle bbox
                e_bb = lines[i].replace('\n', '').split()
                e_bb = [float(_) for _ in e_bb]
                box = covert_ellipse_to_box(e_bb)

                if box[0] == box[1] or box[2] == box[3]:
                    continue
                
                box[0], box[1] = int(box[0]), int(box[1])
                box[2], box[3] = int(box[2]), int(box[3])
                                
                bboxes.append(box)
                
            data_dict['boxes'] = bboxes
            all_data.append(data_dict)
            
            cnt = cnt + 2 + data_dict['data_num']
            if cnt >= len(lines):
                break
            
    return all_data

def compute_IOUs(boxes, c_box):
    overlap = [None]*len(boxes)
    for i, box in enumerate(boxes):
        overlap[i] = compute_IOU(c_box, box)
    
    m_ind = np.argmax(overlap)
    return overlap[m_ind], m_ind
    
def get_pos_crop_img(img_load, lab_load, pos_num_per_box = 10, min_face_size = 12, IOU_t = 0.6):
    c_imgs  = []
    c_labs  = []
    img = cv2.imread(img_load)
    img_h, img_w, _ = img.shape
    lab = json.load(open(lab_load, 'r'))
    boxes = lab['boxes']
    max_trial = pos_num_per_box*10
    
    u_lab_num = 0
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        bw = x2 - x1 + 1
        bh = y2 - y1 + 1
        
        if max(bw, bh) < min_face_size:
            print("[NOTE] some bounding boxes are too small!!")
            continue
        
        # positive face data is extracted based on square crop image
        p_num = t_num = 0
        while p_num < pos_num_per_box and t_num < max_trial:
            t_num += 1
            
            c_size = np.random.randint(int(0.8*min(bw, bh)), 
                                       int(1.2*max(bw, bh)))
            
            dx = np.random.randint(-bw*0.1, bw*0.1)
            dy = np.random.randint(-bh*0.1, bh*0.1)

            cx1 = int(max(x1 + bw/2 + dx - c_size/2, 0))
            cy1 = int(max(y1 + bh/2 + dy - c_size/2, 0))
            cx2 = int(min(cx1 + c_size, img_w))
            cy2 = int(min(cy1 + c_size, img_h))
            c_box = [cx1, cy1, cx2, cy2]
        
            m_overlap = compute_IOU(box, c_box)
            if m_overlap >= IOU_t:
                c_img  = img[cy1:cy2, cx1:cx2,:].copy()
                c_lab  = {}
                c_lab['box'] = [(x1 - cx1)/float(c_size),
                                (y1 - cy1)/float(c_size),
                                (x2 - cx1)/float(c_size),
                                (y2 - cy1)/float(c_size)]
                c_lab['class'] = 1
                c_lab['o_box'] = [x1, y1, x2, y2]
                c_lab['data_path'] = img_load
                c_labs.append(c_lab)
                c_imgs.append(c_img)
                if p_num == 0:
                    u_lab_num += 1
                p_num += 1
        
        '''
        1. Due to aspect ratio of some boxes, it is difficult to find 
        positive face data based on square crop image in some cases
        2. To handle this, non-square crop image is used to find positive 
        face data firstly. Then, non-square crop image is converted to 
        square crop image with masking
        3. Not sure if it is a good solution. But, it is convenient.
        '''
        if p_num == 0:
            t_num = 0
            while p_num < pos_num_per_box and t_num < max_trial:
                t_num += 1
                dx = np.random.randint(-bw*0.15, bw*0.15)
                dy = np.random.randint(-bh*0.15, bh*0.15)

                cx1 = int(x1 + dx)
                cy1 = int(y1 + dy)
                cx2 = int(x1 + bw)
                cy2 = int(y1 + bh)
                c_box = [cx1, cy1, cx2, cy2]
            
                m_overlap = compute_IOU(box, c_box)

                if m_overlap >= IOU_t: 
                    cx1, cy1 = max(cx1, 0), max(cy1, 0)
                    cx2, cy2 = min(cx2, img_w), min(cy2, img_h)
                    cbw, cbh = cx2 - cx1 + 1, cy2 - cy1 + 1
                    c_size = max(cbw, cbh)

                    c_img = np.zeros((c_size, c_size, 3))
                    c_img[0:cbh - 1, 0:cbw - 1]  = img[cy1:cy2, cx1:cx2,:].copy()
                    c_lab  = {}
                    c_lab['box'] = [(x1 - cx1)/float(c_size),
                                    (y1 - cy1)/float(c_size),
                                    (x2 - cx1)/float(c_size),
                                    (y2 - cy1)/float(c_size)]
                    c_lab['class'] = 1
                    c_lab['o_box'] = [x1, y1, x2, y2]
                    c_lab['data_path'] = img_load
                    c_labs.append(c_lab)
                    c_imgs.append(c_img)
                    if p_num == 0:
                        u_lab_num += 1
                    p_num += 1
        
    return c_imgs, c_labs, u_lab_num

def get_neg_crop_img(img_load, lab_load, max_neg_num, min_face_size = 12, IOU_t = 0.3):
    c_imgs  = []
    c_labs  = []
    img = cv2.imread(img_load)
    img_h, img_w, _ = img.shape
    lab = json.load(open(lab_load, 'r'))
    boxes = lab['boxes']
    
    n_num = 0
    
    while n_num < max_neg_num:
        c_size = np.random.randint(min_face_size, 
                                   min(img_h, img_w)/2)

        cx1 = np.random.randint(0, img_w - c_size)
        cy1 = np.random.randint(0, img_h - c_size)
        cx2 = cx1 + c_size
        cy2 = cy1 + c_size
        c_box = [cx1, cy1, cx2, cy2]

        m_overlap, m_ind = compute_IOUs(boxes, c_box)
        if m_overlap < IOU_t:
            c_img  = img[cy1:cy2, cx1:cx2,:].copy()
            c_lab  = {}
            c_lab['box'] = [0, 0, 0, 0]
            c_lab['class'] = 0
            c_lab['o_box'] = [0, 0, 0, 0]
            c_lab['data_path'] = img_load
            c_labs.append(c_lab)
            c_imgs.append(c_img)
            n_num += 1
            
    return c_imgs, c_labs