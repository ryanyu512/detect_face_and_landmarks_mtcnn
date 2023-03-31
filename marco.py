'''
UPDATE ON 2023/03/16

1. aims to store less frequently changed global variables 
'''

#flag to define if it is in debug mode
IS_DEBUG = False
#input shape of p_net during training
P_IMG_H = P_IMG_W = 12
#input shape of r_net during training
R_IMG_H = R_IMG_W = 24
#input shape of o_net during training
O_IMG_H = O_IMG_W = 48
#training weights of loss
#WEIGHTS[0] => face classification
#WEIGHTS[1] => bounding box regression
#WEIGHTS[2] => landmarks detection
P_WEIGHTS = [1.0, 0.5, 0.0]
R_WEIGHTS = [1.0, 0.5, 0.0]
O_WEIGHTS = [1.0, 0.5, 1.0]
#number of data in a single batch
BATCH = 512
PRE_FETCH = BATCH//2
#learning rate
LR = 0.0005
#number of training epoch
EPOCH = 2000