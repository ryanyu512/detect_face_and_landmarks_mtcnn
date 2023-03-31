'''
    Updated on 2023/03/27
    
    1. aim to train a network for face detection
    2. p-net: propose network
    3. r-net: refine network
    4. o-net: output network
'''

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPooling2D, PReLU, Flatten, Softmax

class ConvBlock:
    
    def __init__(self,
                  channel_size,
                  kernel_size = (3, 3), 
                  c_strides = (1, 1), 
                  c_padding = 'valid', 
                  is_add_maxpool = True,
                  pool_size = (2, 2),
                  p_strides = (2, 2),
                  p_padding = 'same'):
        
        self.conv2d = Conv2D(channel_size, 
                             kernel_size=kernel_size, 
                             strides=c_strides, 
                             padding=c_padding)
        self.prelu = PReLU(shared_axes=[1, 2])
        self.maxpool2d = None
        if is_add_maxpool:
            self.maxpool2d = MaxPooling2D(pool_size=pool_size, 
                                          strides=p_strides, 
                                          padding=p_padding)
    def __call__(self, x):
        
        x = self.conv2d(x)
        x = self.prelu(x)
        if self.maxpool2d is not None:
            x = self.maxpool2d(x)
            
        return x
    
class Network:
    
    def __init__(self):
        self.p_conv1 = ConvBlock(channel_size = 10)
        self.p_conv2 = ConvBlock(channel_size = 16,
                                 is_add_maxpool = False)
        self.p_conv3 = ConvBlock(channel_size = 32,
                                 is_add_maxpool = False)
        self.r_conv1 = ConvBlock(channel_size = 28,
                                 pool_size = (3, 3))
        self.r_conv2 = ConvBlock(channel_size = 48,
                                 pool_size = (3, 3),
                                 p_padding = 'valid')
        self.r_conv3 = ConvBlock(channel_size = 64,
                                 kernel_size = (2, 2), 
                                 is_add_maxpool = False)
        self.o_conv1 = ConvBlock(channel_size = 32,
                                 pool_size = (3, 3))
        self.o_conv2 = ConvBlock(channel_size = 64,
                                 pool_size = (3, 3),
                                 p_padding = 'valid')
        self.o_conv3 = ConvBlock(channel_size = 64)
        self.o_conv4 = ConvBlock(channel_size = 128,
                                 kernel_size = (2, 2),
                                 is_add_maxpool = False)


        
    def pnet(self, input_shape = None):
        
        if input_shape is None:
            input_shape = (None, None, 3)
        
        #define input shape   
        #12, 12, 3
        feed = Input(input_shape)
        
        #convolution block
        #12, 12, 3 => 5, 5, 10
        x = self.p_conv1(feed)
        #5, 5, 10 => 3, 3, 16
        x = self.p_conv2(x)
        #3, 3, 16 => 1, 1, 32
        x = self.p_conv3(x)
        
        #out 1: confidence
        out1 = Conv2D(2,
                      kernel_size = (1, 1),
                      strides = (1, 1))(x)
        out1 = Softmax(axis = 3)(out1)
        
        #out 2: bounding box within 12*12 sub-image
        out2 = Conv2D(4, 
                      kernel_size = (1, 1),
                      strides = (1, 1))(x)
        
        model = Model(feed, [out2, out1])
        
        return model
    
    def rnet(self, input_shape = None):
        if input_shape is None:
            input_shape = (24, 24, 3)

        #24, 24, 3
        feed = Input(input_shape)

        #convolution block 
        #24, 24, 3 => 11, 11, 28
        x = self.r_conv1(feed)
        #11, 11, 28 => 4, 4, 8
        x = self.r_conv2(x)
        #4, 4, 48 => 3, 3, 64
        x = self.r_conv3(x)

        #FFC layer 1
        #3, 3, 64 => 576
        x = Flatten()(x)
        #576 => 128
        x = Dense(128)(x)
        #128 => 128
        x = PReLU()(x)

        #output 1
        #128 => 2
        out1 = Dense(2)(x)
        #2 => 2
        out1 = Softmax(axis=1)(out1)

        #output 2
        #128 => 4
        out2 = Dense(4)(x)

        r_net = Model(feed, [out2, out1])

        return r_net
    
    def onet(self, input_shape = None):

        if input_shape is None:
            input_shape = (48, 48, 3)
            
        feed = Input(input_shape)
        
        #convolution block 
        #48, 48, 3 => 23, 23, 32 
        x = self.o_conv1(feed)
        #23, 23, 32 => 10, 10, 64
        x = self.o_conv2(x)
        #10, 10, 64 => 4, 4, 64
        x = self.o_conv3(x)
        #4, 4, 64 => 3, 3, 128
        x = self.o_conv4(x)
        
        #FCC Layer 1
        #3, 3, 128 => 1152
        x = Flatten()(x)
        #1152 => 256
        x = Dense(256)(x)
        #256 => 256
        x = PReLU()(x)
        
        #ouput 1
        #256 => 2
        out1 = Dense(2)(x)
        out1 = Softmax(axis=1)(out1)
        
        #ouput 2
        #256 => 4
        out2 = Dense(4)(x)
        
        #output 3
        out3 = Dense(10)(x)
        
        o_net = Model(feed, [out2, out3, out1])
        
        return o_net