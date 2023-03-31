import os 
import network 
from network import Network
import numpy as np
import cv2

class BBoxes_Status():
    def __init__(self, img_w, img_h, status = None):
        '''
            args:
                img_w, img_h: width and height of raw image
                status: bounding box status
        '''
        
        #original image width and height
        self.img_w = img_w
        self.img_h = img_h
        
        #pixel values relative to image origin
        self.img_x1 = []
        self.img_x2 = []
        self.img_y1 = []
        self.img_y2 = []
        
        #pixel values relative to sliding window origin
        self.bbox_x1 = []
        self.bbox_x2 = []
        self.bbox_y1 = []
        self.bbox_y2 = []
        
        #bounding box width and height
        self.boxw = []
        self.boxh = []
        
        if status is not None:
            self.update_status(status)
            
    def update_status(self, status):
        self.bbox_y1, self.bbox_y2, self.bbox_x1, self.bbox_x2, self.img_y1, self.img_y2, self.img_x1, self.img_x2, self.boxw, self.boxh = status

class MTCNN():
    def __init__(self, 
                 min_face_size = 20, 
                 face_t = None, 
                 scale_factor = 0.709):
        '''
            args:
                min_face_size: minimum face size
                face_t       : if confidence >= face_t, this sliding window contains a face
                scale_factor : the scale factor to build up image pyramid
        '''
        if face_t is None:
            face_t = [0.5, 0.7, 0.7]
        
        #sliding window for searching one face
        self.slide_win_size = 12   
        #stride step of sliding window
        self.stride = 2
        self.min_face_size  = min_face_size
        self.face_t = face_t
        self.scale_factor = scale_factor

        self.pnet = Network().pnet()
        self.rnet = Network().rnet()
        self.onet = Network().onet()
    
    def compute_scale_pyramid(self, win_face_ratio, scale_img_size):
        '''
        args:
            win_face_ratio: sliding_window_size/min_face_size
            scale_img_size: the min(width, height) of scale image size
        
        returns:
            scales: scales factor of original images
        '''
        
        scales = []
        cnt = 0
        
        #the whole idea is to keep scaling dowm the image until
        #the size of scaled image is < sliding windows size
        while scale_img_size >= self.slide_win_size:
            scales += [win_face_ratio*np.power(self.scale_factor, cnt)]
            scale_img_size = scale_img_size*self.scale_factor
            cnt += 1
            
        return scales
    
    def scale_image(self, img, scale):
        '''
        args: 
            img: input image
            scale: scale factor for resize
        
        returns:
            nor_img: normalized and rescaled image
        '''
        
        h, w, c = img.shape
        
        new_h = int(np.ceil(h*scale))
        new_w = int(np.ceil(w*scale))
        
        #check what is cv2.INTER_AREA
        #https://medium.com/@wenrudong/what-is-opencvs-inter-area-actually-doing-282a626a09b3        
        new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        #0.0078125 = 1/128
        new_img = (new_img - 127.5)*0.0078125

        return new_img
    
    def compute_bb(self, hmap, regs, scale, face_t):
        
        #transpose hmap and regs to obtain (x, y) instead of (y, x)
        hmap = np.transpose(hmap)
        
        #coordinates of sliding windows (x, y)
        #since hmap is transposed => row = x, cols = y
        x, y = r, c = np.where(hmap >= face_t)
        slide_windows = np.transpose(np.vstack([x, y]))

        #slide_windows_s = (x1, y1)
        slide_windows_s = np.fix((self.stride*slide_windows + 1)/scale)
        #slide_windows_e = (x2, y2)
        slide_windows_e = np.fix((self.stride*slide_windows + self.slide_win_size)/scale)

        #confidence of sliding windows
        conf = np.expand_dims(hmap[(x, y)], 1)
        #bboxes of sliding windows
        bbox_x1, bbox_y1 = np.transpose(regs[:,:,0]), np.transpose(regs[:,:,1])
        bbox_x2, bbox_y2 = np.transpose(regs[:,:,2]), np.transpose(regs[:,:,3])
        bbox_x1, bbox_x2 = bbox_x1[(x, y)], bbox_x2[(x, y)]
        bbox_y1, bbox_y2 = bbox_y1[(x, y)], bbox_y2[(x, y)]
        regs = np.transpose(np.vstack([bbox_x1, bbox_y1, bbox_x2, bbox_y2]))
        
        #pack together
        bboxes = np.hstack([slide_windows_s, slide_windows_e, conf, regs])
        return bboxes
        
    def compute_nms(self, bboxes, overlap_t, method = 'union'):
        """ 
        args:
            bboxes: list of bounding boxes 
            (win_x1, win_y1, win_x2, win_y2, confidence, 
             reg_x1, reg_y1, reg_x2, reg_y2)
            overlap_t: threshold for filtering overlapping boxes
        """
        
        if bboxes.size == 0:
            return np.empty((0, 3))
        
        #get the sliding window x1, y1, x2, y2
        win_x1 = bboxes[:, 0]
        win_y1 = bboxes[:, 1]
        win_x2 = bboxes[:, 2]
        win_y2 = bboxes[:, 3]
        
        conf  = bboxes[:,4]
        s_ind = np.argsort(conf)

        #compute windows area
        win_a = (win_y2 - win_y1 + 1)*(win_x2 - win_x1 + 1)       
        
        gbox_ind = np.zeros_like(s_ind)
        cnt = 0
        while s_ind.size > 0:
            i = s_ind[-1]
            gbox_ind[cnt] = i
            cnt += 1
            
            ind = s_ind[0:-1]
            inter_x1 = np.maximum(win_x1[i], win_x1[ind])
            inter_y1 = np.maximum(win_y1[i], win_y1[ind])
            inter_x2 = np.minimum(win_x2[i], win_x2[ind])
            inter_y2 = np.minimum(win_y2[i], win_y2[ind])
            
            inter_w = np.maximum(0.0, inter_x2 - inter_x1 + 1)
            inter_h = np.maximum(0.0, inter_y2 - inter_y1 + 1)
            inter_a = inter_w*inter_h
            
            if method == 'union':
                overlap = inter_a/(win_a[i] + win_a[ind] - inter_a)
            elif method == 'min':
                overlap = inter_a/np.minimum(win_a[i], win_a[ind])

            s_ind = s_ind[np.where(overlap <= overlap_t)]
        
        gbox_ind = gbox_ind[0:cnt]
        
        return gbox_ind
    
    def compute_sqbox(self, bboxes):
        h = bboxes[:,3] - bboxes[:,1]
        w = bboxes[:,2] - bboxes[:,0]
        
        max_L = np.maximum(h, w)
        bboxes[:,0] = bboxes[:,0] + 0.5*w - 0.5*max_L
        bboxes[:,1] = bboxes[:,1] + 0.5*h - 0.5*max_L
        bboxes[:,2] = bboxes[:,0] + max_L
        bboxes[:,3] = bboxes[:,1] + max_L
        
        return bboxes
    
    def compute_bboxes_status(self, bboxes, img_w, img_h):
        '''
        e.g. 
        10 - 0 = 10, but the actual width is 11
        so, 10 - 0 + 1 = 11
        '''

        bboxes_w = (bboxes[:,2] - bboxes[:,0] + 1).astype(np.int32)
        bboxes_h = (bboxes[:,3] - bboxes[:,1] + 1).astype(np.int32)
        bboxes_N = bboxes.shape[0]
        
        bboxes_x1 = np.zeros(bboxes_N, dtype = np.int32)
        bboxes_y1 = np.zeros(bboxes_N, dtype = np.int32)
        bboxes_x2 = (bboxes_w - 1).copy().astype(np.int32)
        bboxes_y2 = (bboxes_h - 1).copy().astype(np.int32)
        
        img_x1 = bboxes[:,0].copy().astype(np.int32)
        img_y1 = bboxes[:,1].copy().astype(np.int32)
        img_x2 = bboxes[:,2].copy().astype(np.int32)
        img_y2 = bboxes[:,3].copy().astype(np.int32)
        
        '''
        e.g.
        img_x1 = 50, img_x2 = 600, img_w = 400, bboxes_w = 650
        bboxes_x2 = -600 + 400 + 550 -1 = -600 + 1050 - 1 = 349
        bboxes_x1 = 0
        img_x1 = 50
        img_x2 = 399
        '''
        
        ind = np.where(img_x2 >= img_w)
        bboxes_x2.flat[ind] = np.expand_dims(-img_x2[ind] + img_w + bboxes_w[ind] - 2, 1)
        img_x2[ind] = img_w - 1
        
        ind = np.where(img_y2 >= img_h)
        bboxes_y2.flat[ind] = np.expand_dims(-img_y2[ind] + img_h + bboxes_h[ind] - 2, 1)
        img_y2[ind] = img_h - 1
        
        ind = np.where(img_x1 < 0)
        bboxes_x1.flat[ind] = np.expand_dims(-img_x1[ind], 1)
        img_x1[ind] = 0
                 
        ind = np.where(img_y1 < 0)
        bboxes_y1.flat[ind] = np.expand_dims(-img_y1[ind], 1)
        img_y1[ind] = 0
        
        return bboxes_y1, bboxes_y2, bboxes_x1, bboxes_x2, img_y1, img_y2, img_x1, img_x2, bboxes_w, bboxes_h
    
    def p_stage(self, img, scales, bboxes_status):
        '''
        args:
            img: raw img
            scales: scale factor for resizing image
            bboxes_status: sliding windows coordinates, bounding box coordinates and confidence
        
        returns:
            all_p_boxes: bounding boxes from p stage
            p_status: sliding windows coordinates, bounding box coordinates and related confidence
        '''
        all_p_boxes = np.empty((0, 9))
        p_status = bboxes_status
        
        for scale in scales:
            new_img = np.expand_dims(self.scale_image(img, scale), 0)

            p_out = self.pnet.predict(new_img)

            '''
            1. each pixel represents 12*12 sliding window 
            2. each pixel has the confidence
            3. confidence map of image
            '''
            hmap = p_out[1][0,:,:,0]
            #regression points of bounding box 
            regs = p_out[0][0,:,:,:] 

            p_boxes = self.compute_bb(hmap, regs, scale, self.face_t[0])
            good_boxes_ind = self.compute_nms(p_boxes.copy(), 0.5)
            
            if p_boxes.size > 0 and good_boxes_ind.size > 0:
                p_boxes = p_boxes[good_boxes_ind, :]
                all_p_boxes = np.append(all_p_boxes, p_boxes, axis = 0)

        if all_p_boxes.size > 0:
            good_boxes_ind = self.compute_nms(all_p_boxes.copy(), 0.7)
            all_p_boxes = all_p_boxes[good_boxes_ind, :]

            bbox_w = all_p_boxes[:,2] - all_p_boxes[:,0] + 1
            bbox_h = all_p_boxes[:,3] - all_p_boxes[:,1] + 1
            
            x1 = (all_p_boxes[:, 0] + all_p_boxes[:, 5] * bbox_w)
            y1 = (all_p_boxes[:, 1] + all_p_boxes[:, 6] * bbox_h)
            x2 = (all_p_boxes[:, 0] + all_p_boxes[:, 7] * bbox_w)
            y2 = (all_p_boxes[:, 1] + all_p_boxes[:, 8] * bbox_h)
            
            all_p_boxes = np.transpose(np.vstack([x1, y1, x2, y2, all_p_boxes[:, 4]]))
            all_p_boxes = self.compute_sqbox(all_p_boxes.copy())
            all_p_boxes[:, 0:4] = np.fix(all_p_boxes[:, 0:4]).astype(np.int32)
            
            bboxes_status = self.compute_bboxes_status(all_p_boxes.copy(), 
                                                       p_status.img_w, 
                                                       p_status.img_h)
            
            p_status = BBoxes_Status(p_status.img_w, p_status.img_h, 
                                     bboxes_status)
            
            return all_p_boxes, p_status
        else:
            return np.empty((0, 5)), np.empty((0, 10))
            
    
    def r_stage(self, img, p_bboxes, bboxes_status):
        
        '''
        args:
            img: raw img
            p_bboxes: bounding boxes from p stage
            bboxes_status: sliding windows coordinates, bounding box coordinates and confidence
        
        returns:
            r_bboxes: bounding boxes from r stage
            bboxes_status: sliding windows coordinates, bounding box coordinates and related confidence
        '''
        
        bboxes_num = p_bboxes.shape[0]
        if bboxes_num == 0:
            return np.empty((0, 5)), np.empty((0, 10))
        
        r_inputs = np.zeros(shape = (24,24,3,bboxes_num))
        
        for i in range(bboxes_num):
            sub_img = np.zeros((bboxes_status.boxh[i], bboxes_status.boxw[i], 3))

            sub_img[bboxes_status.bbox_y1[i]:bboxes_status.bbox_y2[i] + 1,
                    bboxes_status.bbox_x1[i]:bboxes_status.bbox_x2[i] + 1,:] = \
            img[bboxes_status.img_y1[i]:bboxes_status.img_y2[i] + 1,
                bboxes_status.img_x1[i]:bboxes_status.img_x2[i] + 1,:]
                
            if (sub_img.shape[0] > 0  and sub_img.shape[1] > 0) or \
               (sub_img.shape[0] == 0 and sub_img.shape[1] == 0):
                r_inputs[:,:,:,i] = cv2.resize(sub_img, (24, 24), 
                                               interpolation=cv2.INTER_AREA)
            else:
                return np.empty(shape=(0,)), bboxes_status
        
        
        r_inputs = (r_inputs - 127.5)*0.0078125
        t_r_inputs = np.transpose(r_inputs, (3, 0, 1, 2))
        r_outs = self.rnet.predict(t_r_inputs)
        
        r_conf = r_outs[1][:, 0]
        r_regs = r_outs[0]
        
        good_ind = np.where(r_conf > self.face_t[1])
        r_conf = r_conf[good_ind]
        r_regs = r_regs[good_ind[0], :]
        r_bboxes = np.hstack([p_bboxes[good_ind[0], 0:4].copy(), 
                              np.expand_dims(r_conf, 1)])
        
        if r_bboxes.shape[0] > 0:
            non_overlap_ind = self.compute_nms(r_bboxes.copy(), 0.7)
            r_bboxes = r_bboxes[non_overlap_ind, :]
            r_regs = r_regs[non_overlap_ind, :]
            
            reg_w = r_bboxes[:, 2] - r_bboxes[:, 0] + 1
            reg_h = r_bboxes[:, 3] - r_bboxes[:, 1] + 1

            x1 = (r_bboxes[:, 0] + r_regs[:, 0]*reg_w)
            y1 = (r_bboxes[:, 1] + r_regs[:, 1]*reg_h)
            x2 = (r_bboxes[:, 0] + r_regs[:, 2]*reg_w)
            y2 = (r_bboxes[:, 1] + r_regs[:, 3]*reg_h)
            
            r_bboxes[:, 0:4] = np.transpose(np.vstack([x1, y1, x2, y2]))
            r_bboxes = self.compute_sqbox(r_bboxes.copy())
                    
            return r_bboxes, bboxes_status
        else:
            return np.empty((0, 5)), np.empty((0, 10))
    
    def o_stage(self, img, r_bboxes, bboxes_status):
        
        '''
        args:
            img: raw img
            r_bboxes: bounding boxes from p stage
            bboxes_status: sliding windows coordinates, bounding box coordinates and confidence
        
        returns:
            o_bboxes: bounding boxes from o stage
        '''
        
        bboxes_num = r_bboxes.shape[0]
        if bboxes_num == 0:
            return np.empty((0, 5)), np.empty((0, 10))
        
        r_bboxes = np.fix(r_bboxes).astype(np.int32)
        bboxes_status = BBoxes_Status(bboxes_status.img_w, bboxes_status.img_h,
                                      self.compute_bboxes_status(r_bboxes.copy(), 
                                                                 bboxes_status.img_w,
                                                                 bboxes_status.img_h))
        
        o_inputs = np.zeros(shape = (48,48,3,bboxes_num))
        for i in range(bboxes_num):
            sub_img = np.zeros((bboxes_status.boxh[i], bboxes_status.boxw[i], 3))
 
            sub_img[bboxes_status.bbox_y1[i]:bboxes_status.bbox_y2[i] + 1,
                    bboxes_status.bbox_x1[i]:bboxes_status.bbox_x2[i] + 1,:] = \
            img[bboxes_status.img_y1[i]:bboxes_status.img_y2[i] + 1,
                bboxes_status.img_x1[i]:bboxes_status.img_x2[i] + 1,:]

            if (sub_img.shape[0] > 0  and sub_img.shape[1] > 0) or \
               (sub_img.shape[0] == 0 and sub_img.shape[1] == 0):
                o_inputs[:,:,:,i] = cv2.resize(sub_img, (48, 48), 
                                               interpolation=cv2.INTER_AREA)
            else:
                return np.empty(shape=(0,)), bboxes_status
            
        o_inputs = (o_inputs - 127.5)*0.0078125
        t_o_inputs = np.transpose(o_inputs, (3, 0, 1, 2))

        o_outs = self.onet.predict(t_o_inputs)
        o_conf = o_outs[2][:, 0]
        o_mrks = o_outs[1]
        o_regs = o_outs[0]

        good_ind = np.where(o_conf > self.face_t[2])
        o_conf = o_conf[good_ind]
        o_mrks = o_mrks[good_ind[0], :]
        o_regs = o_regs[good_ind[0], :]

        o_bboxes = np.hstack([r_bboxes[good_ind[0], 0:4].copy(), 
                                np.expand_dims(o_conf, 1)])
        
        landmarks = np.empty((0, 10))
        if o_bboxes.shape[0] > 0:
            reg_w = o_bboxes[:, 2] - o_bboxes[:, 0] + 1
            reg_h = o_bboxes[:, 3] - o_bboxes[:, 1] + 1
            
            x1 = o_bboxes[:, 0] + o_regs[:, 0]*reg_w
            y1 = o_bboxes[:, 1] + o_regs[:, 1]*reg_h
            x2 = o_bboxes[:, 0] + o_regs[:, 2]*reg_w
            y2 = o_bboxes[:, 1] + o_regs[:, 3]*reg_h
            
            landmarks = np.transpose(
                                    np.vstack([o_bboxes[:, 0] + o_mrks[:, 0]*reg_w,
                                               o_bboxes[:, 1] + o_mrks[:, 1]*reg_h,
                                               o_bboxes[:, 0] + o_mrks[:, 2]*reg_w,
                                               o_bboxes[:, 1] + o_mrks[:, 3]*reg_h,
                                               o_bboxes[:, 0] + o_mrks[:, 4]*reg_w,
                                               o_bboxes[:, 1] + o_mrks[:, 5]*reg_h,
                                               o_bboxes[:, 0] + o_mrks[:, 6]*reg_w,
                                               o_bboxes[:, 1] + o_mrks[:, 7]*reg_h,
                                               o_bboxes[:, 0] + o_mrks[:, 8]*reg_w,
                                               o_bboxes[:, 1] + o_mrks[:, 9]*reg_h])
                                    )
            
            o_bboxes[:, 0:4] = np.transpose(np.vstack([x1, y1, x2, y2]))
            non_overlap_ind = self.compute_nms(o_bboxes.copy(), 0.7, 'min')
            o_bboxes = o_bboxes[non_overlap_ind, :]
            landmarks = landmarks[non_overlap_ind, :]
            
            return o_bboxes, landmarks
        else:
            return np.empty((0, 5)), np.empty((0, 10))
        
    def detect_faces(self, img, is_r = True, is_o = True):
        
        '''
            args:
                img: raw image
                is_r: define if r - stage is used 
                is_o: define if o - stage is used
                
            returns:
                result: bounding boxes and or bounding box status
        '''
        
        if img is None or not hasattr(img, "shape"):
            return 
        
        h, w, c = img.shape
        
        bboxes_status = BBoxes_Status(w, h, None)
        '''
        e.g. 
        1. sliding winodow size is set as 12
        2. min_face_size > 12 => too large for sliding window
        3. the image needs to be scaled down to fit the sliding window
        '''
        win_face_ratio = self.slide_win_size/self.min_face_size
        '''
        ensure the scaled h and w are > sliding window, such that 
        faces can be searched based on sliding window
        '''
        min_L = np.amin([h, w])*win_face_ratio
        
        scales = self.compute_scale_pyramid(win_face_ratio, min_L)
        
        result = self.p_stage(img, scales, bboxes_status)
        
        empty_box = np.empty((0, 5))
        empty_mrk = np.empty((0, 10))
        if result[0].size > 0 and is_r:
            result = self.r_stage(img, result[0], result[1])
        else:
            if result[0].size > 0:
                return result[0], empty_mrk
            else:
                return empty_box, empty_mrk
        
        if result[0].size > 0 and is_o:
            regs, mrks = self.o_stage(img, result[0], result[1])
            return regs, mrks
        else:
            if result[0].size > 0:
                return result[0], empty_mrk
            else:
                return empty_box, empty_mrk
        
        
    
        
    

