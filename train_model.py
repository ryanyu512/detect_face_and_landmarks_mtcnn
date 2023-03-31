'''
UPDATE ON 2023/03/16

1. aims at providing libraries for training model

'''

import os
import cv2
import math
import json
import random 
import numpy as np
import tensorflow as tf

from marco import *
from network import *
from visualise import *
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model


def get_img_list(dir, file_type = 'jpg'):
    '''
        args:
            dir: directories that contains targeted images
      file_type: can be .jpg or .png...
        
        returns:
            f: image paths of face data
           nf: image paths of non_face data
    '''
    
    
    f  = os.listdir(os.path.join(dir, 'face'))
    f  = [os.path.join(dir, 'face', _) for _ in f if _.split('.')[-1] == file_type]

    nf = os.listdir(os.path.join(dir, 'non_face'))
    nf = [os.path.join(dir, 'non_face', _) for _ in nf if _.split('.')[-1] == file_type]
    
    return f, nf

def load_img(path):
    '''
        args:
            path: image paths
            
        returns:
            img: image
    '''
    
    
    byte_img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(byte_img)
    return img

def get_imgs(img_list, new_img_h = None, new_img_w = None):
    '''
        args:
            img_list: list of image paths
           new_img_h: scaled image height
           new_img_w: scaled image width
           
        returns:
                imgs: scaled and normalised images
    '''
    
    imgs = tf.data.Dataset.list_files(img_list, shuffle=False)
    imgs = imgs.map(load_img)
    if new_img_h is not None or new_img_w is not None:
        imgs = imgs.map(lambda x: tf.image.resize(x, (new_img_h, new_img_w)))
    imgs = imgs.map(lambda x: (x - 127.5)*0.0078125)

    return imgs

def get_label_list(img_list):
    '''
        args:
            img_list: list of image path
            
        returns:
            lab_list: list of label path
    '''
    
    lab_list = [None]*len(img_list)
    for i in range(len(lab_list)):
        uid  = (img_list[i].split('.')[0]).split('/')[-1]
        root = (img_list[i].split('.')[0]).split('/')
        root = '/'.join(root[0:len(root) - 1])
        lab_list[i] = os.path.join(root, uid + '.json')

    return lab_list

def load_labels(label_path):
    '''
        args:
            label_path: data path of label
        
        return:
            [label['class']]: face (1) or non_face (0)
            label['box']: coordindates of bounding box
            [label['is_have']]: contain face landmarks (1)  or no face landmarks(0)
            landmarks: coordinates of face landmarks
    '''
    
    with open(label_path.numpy(), 'r', encoding = "utf-8") as f:
        label = json.load(f)
        
    landmarks = label['right_eye'] + \
                label['left_eye'] + \
                label['nose'] + \
                label['mouth_left'] + \
                label['mouth_right']
    
    return [label['class']], label['box'], [label['is_have']], landmarks

def get_labels(lab_list):
    '''
        args:
            lab_list: list of label path
        returns:
            labs: labels
    '''
    
    labs = tf.data.Dataset.list_files(lab_list, shuffle=False)
    labs = labs.map(lambda x: tf.py_function(load_labels, 
                                             [x], 
                                             [tf.uint8, tf.float16, tf.uint8, tf.float16]
                                            )
                   )

    return labs

def combine_imgs_and_labels(imgs, labels, batch, pre_fetch):
    '''
        args:
            imgs: images
            labels: labels
            batch: number of data in one batch 
            per_fetch: number of data to feteh before processing
        
        returns:
            data: return one batch of images and cooresponding labels
    '''
    
    data = tf.data.Dataset.zip((imgs, labels))
    data = data.shuffle(5000)
    data = data.batch(batch)
    data = data.prefetch(pre_fetch)
    
    return data
    
def custom_loss(class_hat, class_true, bbox_hat, bbox_true, mrks_hat, mrks_true, have_mrks, stage):
    
    '''
        args:
            class_hat: class prediction (face or non_face)
            class_true: class ground truth (face or non_face)
            bbox_hat: bounding box coordindate prediction
            bbox_true: bounding box coordindate ground truth
            mrks_hat: face landmarks prediction
            mrks_true: face landmarks ground truth
            have_mrks: indicate if this label contain face landmarks
            stage: 'p' or 'r' or 'o'
        
        returns:
            class_loss: loss of classification
            regre_loss: loss of bounding box regression
            marks_loss: loss of landmarks regression
    '''
    
    if stage == 'p':
        class_loss =  (     class_true[:,0] *tf.math.log(class_hat[:,0,0,0] + 1e-10) + \
                       (1 - class_true[:,0])*tf.math.log(class_hat[:,0,0,1] + 1e-10)
                      )
        
        regre_loss =  tf.square(bbox_hat[:,0,0,0] - bbox_true[:, 0]) + \
                      tf.square(bbox_hat[:,0,0,1] - bbox_true[:, 1]) + \
                      tf.square(bbox_hat[:,0,0,2] - bbox_true[:, 2]) + \
                      tf.square(bbox_hat[:,0,0,3] - bbox_true[:, 3])
    else:
        class_loss =  (     class_true[:,0] *tf.math.log(class_hat[:,0] + 1e-10) + \
                       (1 - class_true[:,0])*tf.math.log(class_hat[:,1] + 1e-10)
                      )
        regre_loss =  tf.square(bbox_hat[:,0] - bbox_true[:, 0]) + \
                      tf.square(bbox_hat[:,1] - bbox_true[:, 1]) + \
                      tf.square(bbox_hat[:,2] - bbox_true[:, 2]) + \
                      tf.square(bbox_hat[:,3] - bbox_true[:, 3])
        
        if mrks_hat is not None:
            marks_loss = tf.square(mrks_hat[:,0] - mrks_true[:, 0]) + \
                         tf.square(mrks_hat[:,1] - mrks_true[:, 1]) + \
                         tf.square(mrks_hat[:,2] - mrks_true[:, 2]) + \
                         tf.square(mrks_hat[:,3] - mrks_true[:, 3]) + \
                         tf.square(mrks_hat[:,4] - mrks_true[:, 4]) + \
                         tf.square(mrks_hat[:,5] - mrks_true[:, 5]) + \
                         tf.square(mrks_hat[:,6] - mrks_true[:, 6]) + \
                         tf.square(mrks_hat[:,7] - mrks_true[:, 7]) + \
                         tf.square(mrks_hat[:,8] - mrks_true[:, 8]) + \
                         tf.square(mrks_hat[:,9] - mrks_true[:, 9])
            
    eff_num1 = tf.cast(tf.reduce_sum(class_true[:,0]), tf.float32)
    class_loss = -tf.reduce_mean(class_loss)
    regre_loss =  tf.reduce_sum(regre_loss*class_true[:,0])/(eff_num1 + 1e-10)
        
    if mrks_hat is not None:
        eff_num2   = tf.cast(tf.reduce_sum(have_mrks[:,0]), tf.float32)
        marks_loss = tf.reduce_sum(marks_loss*have_mrks[:,0])/(eff_num2 + 1e-10)
    else:
        marks_loss = 0
        
    return class_loss, regre_loss, marks_loss

def compute_IOU(reg_p, reg_t):
    
    '''
        args:
            reg_p, reg_t: coordinates of bounding box 1 and 2
        returns:
            overlap: overlap percentage of two bounding box
    '''
    
    area_p = (reg_p[2] - reg_p[0])*(reg_p[3] - reg_p[1])
    area_t = (reg_t[2] - reg_t[0])*(reg_t[3] - reg_t[1])
    
    inter_x1 = np.maximum(reg_p[0], reg_t[0])
    inter_y1 = np.maximum(reg_p[1], reg_t[1])
    inter_x2 = np.minimum(reg_p[2], reg_t[2])
    inter_y2 = np.minimum(reg_p[3], reg_t[3])
    
    inter_a = np.maximum(0, inter_x2 - inter_x1)* \
              np.maximum(0, inter_y2 - inter_y1)
    
    overlap = inter_a/(area_p + area_t - inter_a)

    return overlap
    
class TrainModel(Model): 
    def __init__(self, model,  **kwargs): 
        super().__init__(**kwargs)
        self.model = model
        self.test_result = {"batch_loss": 0, 
                            "cls_loss": 0, 
                            "reg_loss": 0,
                            "mrk_loss": 0,
                            "corr_cls": 0, 
                            "corr_box": 0,
                            "tp": 0, 
                            "tn": 0, 
                            "fp": 0, 
                            "fn": 0}
    def reset_result(self):
        self.test_result = {"batch_loss": 0, 
                            "cls_loss": 0, 
                            "reg_loss": 0,
                            "mrk_loss": 0,
                            "corr_cls": 0, 
                            "corr_box": 0,
                            "tp": 0, 
                            "tn": 0, 
                            "fp": 0, 
                            "fn": 0}
        
    def compile(self, opt, custom_loss, cls_w, reg_w, mrk_w, stage, **kwargs):
        super().compile(**kwargs)
        self.custom_loss = custom_loss
        self.opt = opt
        self.cls_w = cls_w
        self.reg_w = reg_w
        self.mrk_w = mrk_w
        self.stage = stage
        
    def train_step(self, batch, **kwargs): 
        
        x, y = batch
        
        with tf.GradientTape() as tape: 
            
            if self.stage == 'o':
                bbox_hat, marks_hat, class_hat  = self.model(x, training = True)
            else:
                bbox_hat, class_hat  = self.model(x, training = True)
                marks_hat = None
            
            cls_loss, reg_loss, mrk_loss = self.custom_loss(class_hat, 
                                                          tf.cast(y[0], tf.float32),
                                                          bbox_hat,
                                                          tf.cast(y[1], tf.float32),
                                                          marks_hat,
                                                          tf.cast(y[3], tf.float32),
                                                          tf.cast(y[2], tf.float32),  
                                                          self.stage)
            
            batch_loss = cls_loss*self.cls_w + \
                         reg_loss*self.reg_w + \
                         mrk_loss*self.mrk_w
            
            grad = tape.gradient(batch_loss, self.model.trainable_variables)
        
        self.opt.apply_gradients(zip(grad, self.model.trainable_variables))
        
        return {"batch_loss": batch_loss, 
                "cls_loss": cls_loss, 
                "reg_loss": reg_loss,
                "mrk_loss": mrk_loss}
    
    def eval_step(self, batch, **kwargs): 
        
        x, y = batch
        
        if self.stage == 'o':
            bbox_hat, marks_hat, class_hat  = self.model(x, training = False)
        else:
            bbox_hat, class_hat  = self.model(x, training = False)
            marks_hat = None

        #custom_loss(class_hat, class_true, bbox_hat, bbox_true, mrks_hat, mrks_true, have_mrks, stage):
        cls_loss, reg_loss, mrk_loss = self.custom_loss(class_hat, 
                                                      tf.cast(y[0], tf.float32),
                                                      bbox_hat,
                                                      tf.cast(y[1], tf.float32),
                                                      marks_hat,
                                                      tf.cast(y[3], tf.float32),
                                                      tf.cast(y[2], tf.float32),  
                                                      self.stage)

        batch_loss = cls_loss*self.cls_w + \
                     reg_loss*self.reg_w + \
                     mrk_loss*self.mrk_w
        
        return {"batch_loss": batch_loss, 
                "cls_loss": cls_loss, 
                "reg_loss": reg_loss,
                "mrk_loss": mrk_loss}
        
    def test_step(self, batch, **kwargs):
        
        x, y = batch     
        
        if self.stage == 'o':
            bbox_hat, marks_hat, class_hat  = self.model(x, training = False)
        else:
            bbox_hat, class_hat  = self.model(x, training = False)
            marks_hat = None

        #custom_loss(class_hat, class_true, bbox_hat, bbox_true, mrks_hat, mrks_true, have_mrks, stage):
        cls_loss, reg_loss, mrk_loss = self.custom_loss(class_hat, 
                                                      tf.cast(y[0], tf.float32),
                                                      bbox_hat,
                                                      tf.cast(y[1], tf.float32),
                                                      marks_hat,
                                                      tf.cast(y[3], tf.float32),
                                                      tf.cast(y[2], tf.float32),  
                                                      self.stage)

        batch_loss = cls_loss*self.cls_w + \
                     reg_loss*self.reg_w + \
                     mrk_loss*self.mrk_w
        
        tn = tp = fp = fn = 0
        corr_cls = corr_box = face_cnt = 0
        
        for i in range(class_hat.shape[0]):
            
            if self.stage == 'p': 
                est_cls = 1 if class_hat[i,0,0,0] > 0.5 else 0
            else:
                est_cls = 1 if class_hat[i,0] > 0.5 else 0
                
            est_box = bbox_hat[i,0,0,:] if self.stage == 'p' else bbox_hat[i,:]

            gt_cls = y[0][i, 0]            
            gt_box = y[1][i, :]

            if gt_cls == 1:
                
                overlap = compute_IOU(est_box, gt_box)

                face_cnt += 1

                if overlap >= 0.5:
                    corr_box += 1
                
            if est_cls == gt_cls:
                if est_cls == 0:
                    tn += 1
                else:
                    tp += 1
                corr_cls += 1
            else:
                if est_cls == 1:
                    fp += 1
                else:
                    fn += 1
        
        self.test_result["batch_loss"] += batch_loss
        self.test_result["cls_loss"] += cls_loss
        self.test_result["reg_loss"] += reg_loss
        self.test_result["mrk_loss"] += mrk_loss
        self.test_result["corr_cls"] += corr_cls
        self.test_result["corr_box"] += corr_box
        self.test_result["tp"] += tp
        self.test_result["tn"] += tn
        self.test_result["fp"] += fp
        self.test_result["fn"] += fn
        
    def call(self, X, **kwargs): 
        return self.model(X, **kwargs)
    
def train_and_test_stage(data_folders, flag):
    '''
        args:
            data_folders: it is designed to provide multiple sources for training, validation and testing. But, the sub-folders
            of data_folders need to be the same. For example, data_folders = ['a', 'b']. Then, subfolders of folder 'a' and 'b'
            need to have 'train' and 'valid' and optional 'test'
            flag: 
            1. Training needs training and validation data. Testing data is not necessary. 
            2. If you want to train the model, "IS_TRAIN" = True
            3. If you have pre-trained model parameters, "IS_LOAD_WEIGHTS" = True
            4. If you have saved previous training history, "IS_LOAD_HIST" = True
            5. If you want to save the current and best model parameters, "IS_SAVE" = True
            6. If you want to plot training and validation loss, "IS_PLOT" = True
            7. If you have testing data, choose flag['IS_TEST'] = True
            8. flag['STAGE'] could be ['p', 'r', 'o']
        returns:
            None
    '''
    if flag['STAGE'] == 'p':
        img_h = P_IMG_H
        img_w = P_IMG_W
        curr_model_name = 'cur_pnet.h5'
        best_model_name = 'best_pnet.h5'
        hist_file_name  = 'pnet_loss_history.npy'
        cls_w = P_WEIGHTS[0]
        reg_w = P_WEIGHTS[1]
        mrk_w = P_WEIGHTS[2]
        net_model = Network().pnet(input_shape = None)
    elif flag['STAGE'] == 'r':
        img_h = R_IMG_H
        img_w = R_IMG_W
        curr_model_name = 'cur_rnet.h5'
        best_model_name = 'best_rnet.h5'
        hist_file_name  = 'rnet_loss_history.npy'
        cls_w = R_WEIGHTS[0]
        reg_w = R_WEIGHTS[1]
        mrk_w = R_WEIGHTS[2]
        net_model = Network().rnet(input_shape = None)
    elif flag['STAGE'] == 'o':
        img_h = O_IMG_H
        img_w = O_IMG_W
        curr_model_name = 'cur_onet.h5'
        best_model_name = 'best_onet.h5'
        hist_file_name  = 'onet_loss_history.npy'
        cls_w = O_WEIGHTS[0]
        reg_w = O_WEIGHTS[1]
        mrk_w = O_WEIGHTS[2]
        net_model = Network().onet(input_shape = None)
        
    ####### Get face and non-face img path of based on uid_train, uid_val and uid_test #######

    train_img_list = []
    valid_img_list = []
    if flag['IS_TEST']:
        test_img_list  = []
    for i, data_folder in enumerate(data_folders):
        pf_train, nf_train = get_img_list(os.path.join(data_folder, 'train'))
        pf_valid, nf_valid = get_img_list(os.path.join(data_folder, 'valid'))
        if flag['IS_TEST']:
            pf_test , nf_test  = get_img_list(os.path.join(data_folder, 'test'))

        train_img_list = train_img_list + pf_train + nf_train
        valid_img_list = valid_img_list + pf_valid + nf_valid
        if flag['IS_TEST']:
            test_img_list  = test_img_list  + pf_test  + nf_test
        
    ####### Get training, validation and testing image data #######
        
    train_images = get_imgs(train_img_list, img_h, img_w)
    valid_images = get_imgs(valid_img_list, img_h, img_w)
    if flag['IS_TEST']:
        test_images  = get_imgs(test_img_list,  img_h, img_w)

    ####### Get training, validation and testing labels #######
    train_lab_list = get_label_list(train_img_list)
    valid_lab_list = get_label_list(valid_img_list)
    if flag['IS_TEST']:
        test_lab_list  = get_label_list(test_img_list)

    train_labels = get_labels(train_lab_list)
    valid_labels = get_labels(valid_lab_list)
    if flag['IS_TEST']:
        test_labels  = get_labels(test_lab_list)

    ####### Combine labels and images #######
    train = combine_imgs_and_labels(train_images, train_labels, BATCH, PRE_FETCH)
    val   = combine_imgs_and_labels(valid_images, valid_labels, BATCH, PRE_FETCH)
    if flag['IS_TEST']:
        test  = combine_imgs_and_labels(test_images, test_labels, BATCH, PRE_FETCH)

    batches_train = len(train)
    batches_valid = len(val)
    if flag['IS_TEST']:
        batches_test  = len(test)

    print('train batchs:', batches_train)
    print('valid batchs:', batches_valid)
    if flag['IS_TEST']:
        print('test batchs:', batches_test)
    
    ####### Define optimizer and loss #######

    LR_DECAY = (1./0.75 - 1)/batches_train
    opt = tf.keras.optimizers.Adam(learning_rate = flag['LR'], decay = LR_DECAY)

    ####### Training #######
    if flag['IS_TRAIN']:
        
        if flag['IS_LOAD_WEIGHTS']:
            net_model = load_model(curr_model_name)

        if flag['IS_LOAD_HIST']:
            hist = np.load(hist_file_name, allow_pickle=True).flat[0]
            best_valid_loss = hist['best_valid_loss']
        else:
            hist = {'batch_train_loss': [], 
                    'batch_valid_loss': [],
                    'best_valid_loss': None}
            best_valid_loss = None

        model = TrainModel(net_model)
        model.compile(opt = opt, 
                    custom_loss = custom_loss, 
                    cls_w = cls_w, 
                    reg_w = reg_w,
                    mrk_w = mrk_w,
                    stage = flag['STAGE'])

        for epoch in range(EPOCH):
            train_iter = train.as_numpy_iterator()
            val_iter   = val.as_numpy_iterator()

            total_train_loss = 0.0
            total_valid_loss = 0.0
            
            total_cls_loss_t = 0.0
            total_reg_loss_t = 0.0
            total_mrk_loss_t = 0.0
            
            total_cls_loss_v = 0.0
            total_reg_loss_v = 0.0
            total_mrk_loss_v = 0.0
            
            for i in range(batches_train):
                train_loss = model.train_step(train_iter.next())
                total_train_loss += train_loss['batch_loss']
                total_cls_loss_t += train_loss['cls_loss']
                total_reg_loss_t += train_loss['reg_loss']
                total_mrk_loss_t += train_loss['mrk_loss']
                
            for i in range(batches_valid):
                val_loss = model.eval_step(val_iter.next())
                total_valid_loss += val_loss['batch_loss']
                total_cls_loss_v += val_loss['cls_loss']
                total_reg_loss_v += val_loss['reg_loss']
                total_mrk_loss_v += val_loss['mrk_loss']

            total_train_loss /= batches_train
            total_cls_loss_t /= batches_train
            total_reg_loss_t /= batches_train
            total_mrk_loss_t /= batches_train
            
            total_valid_loss /= batches_valid
            total_cls_loss_v /= batches_valid
            total_reg_loss_v /= batches_valid
            total_mrk_loss_v /= batches_valid
            
            hist['batch_train_loss'].append(total_train_loss)
            hist['batch_valid_loss'].append(total_valid_loss)
            
            if best_valid_loss is None:
                best_valid_loss = total_valid_loss
            else:
                if best_valid_loss > total_valid_loss:
                    best_valid_loss = total_valid_loss
                    hist['best_valid_loss'] = best_valid_loss
                    if flag['IS_SAVE']:
                        net_model.save(best_model_name)
                        print("save the best model!")
                        
            if flag['IS_SAVE']:
                net_model.save(curr_model_name)
                np.save(hist_file_name, hist, allow_pickle=True)
            
            print(f'epoch {epoch + 1}')
            print(f'total_train_loss: {total_train_loss} total_cls_loss_t: {total_cls_loss_t} total_reg_loss_t: {total_reg_loss_t} total_mrk_loss_t: {total_mrk_loss_t}')
            print(f'total_valid_loss: {total_valid_loss} total_cls_loss_v: {total_cls_loss_v} total_reg_loss_v: {total_reg_loss_v} total_mrk_loss_v: {total_mrk_loss_v}')
            print(f'best_valid_loss:  {best_valid_loss}')
            
        if flag['IS_PLOT']:
            plt.plot(hist['batch_train_loss'], 
                    'r', 
                    label = 'training_loss')
            plt.plot(hist['batch_valid_loss'], 
                    'g', 
                    label = 'validation_loss')
            plt.ylabel('Loss')
            plt.xlabel('Number of Epoch')
            plt.legend()
            plt.show()
        
    ####### Testing #######

    if flag['IS_TEST']:
        net_model = load_model(best_model_name) 
        model = TrainModel(net_model)
        model.compile(opt = opt, 
                    custom_loss = custom_loss, 
                    cls_w = cls_w, 
                    reg_w = reg_w,
                    mrk_w = mrk_w,
                    stage = flag['STAGE'])
        
        model.reset_result()
        test_iter = test.as_numpy_iterator()
        for i in range(batches_test):
            model.test_step(test_iter.next())
                        
        res = model.test_result
        
        N = res['tp'] + res['tn'] + res['fn'] + res['fp']
        face_N = res['tp'] + res['fn']
        
        p_face = res['tp']/(res['tp'] +  res['fp'] + 1e-10)
        r_face = res['tp']/(res['tp'] +  res['fn'] + 1e-10)
        f1_face = 2*p_face*r_face/(p_face + r_face)
        
        p_nonface = res['tn']/(res['tn'] +  res['fn'] + 1e-10)
        r_nonface = res['tn']/(res['tn'] +  res['fp'] + 1e-10)
        f1_nonface = 2*p_nonface*r_nonface/(p_nonface + r_nonface)

        acc_cls = (res['tp'] + res['tn'])/N
        acc_box = res['corr_box']/face_N
        
        avg_tot_loss = res['batch_loss']/batches_test
        avg_cls_loss = res['cls_loss']/batches_test
        avg_reg_loss = res['reg_loss']/batches_test
        avg_mrk_loss = res['mrk_loss']/batches_test
        
        print("N:", N, "face_N:", face_N)
        print("[precision] ", 
            f"face: {p_face :.2f}, ",
            f"non_face: {p_nonface :.2f}")
        print("[recall] ", 
            f"face: {r_face :.2f}, ", 
            f"non_face: {r_nonface :.2f}")
        print("[f1-score] ",
            f"face: {f1_face :.2f}, ", 
            f"non_face: {f1_nonface :.2f}")
        print("[accuracy] ",
            f"cls: {acc_cls*100 :.2f}%, ",
            f"box: {acc_box*100 :.2f}%")
        print(f"avg_tot_loss: {avg_tot_loss :.2f}, \
                avg_cls_loss: {avg_cls_loss :.2f}, \
                avg_reg_loss: {avg_reg_loss :.2f}, \
                avg_mrk_loss: {avg_mrk_loss :.2f}")