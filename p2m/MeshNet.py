# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.
import tensorflow as tf
import os
from tensorflow.keras.layers import Conv2D,Flatten,Dense
from p2m.losses import mesh_loss_2, laplace_loss_2
from p2m.layers import LocalGraphProjection, SampleHypothesis, DeformationReasoning




class MeshNet(tf.keras.Model):
    def __init__(self, sample_coord,sample_adj,args, **kwargs):
        super(MeshNet, self).__init__(**kwargs)
        self.sample_adj = sample_adj
        self.sample_coord = sample_coord
        self.args = args
        self.summary_loss = None
        self.merged_summary_op = None
        self.output1l = None
        self.output2l = None
        self.sample1 = None
        self.sample2 = None
        self.proj1 = None
        self.proj2 = None
        self.drb1 = None
        self.drb2 = None
        self.cnn_model()
        self.mesh_model()


    def mesh_model(self):
            # sample hypothesis points
            self.sample1 = SampleHypothesis(sample_coord=self.sample_coord, name='graph_sample_hypothesis_1_layer_0')
            # 1st projection block
            self.proj1 = LocalGraphProjection(name='graph_localproj_1_layer_1')
            # 1st DRB
            self.drb1 = DeformationReasoning(input_dim=self.args.stage2_feat_dim,
                                             sample_coord = self.sample_coord,
                                             sample_adj=self.sample_adj,
                                             args=self.args,
                                             name='graph_drb_blk1_layer_2')
            # sample hypothesis points
            self.sample2 = SampleHypothesis(sample_coord=self.sample_coord, name='graph_sample_hypothesis_2_layer_3')
            # 2nd projection block
            self.proj2 = LocalGraphProjection(name='graph_localproj_2_layer_4')
            # 2nd DRB
            self.drb2 = DeformationReasoning(input_dim=self.args.stage2_feat_dim,
                                             sample_coord = self.sample_coord,
                                             sample_adj=self.sample_adj,
                                             args=self.args,
                                             name='graph_drb_blk2_layer_5')
    def cnn_model(self): 
        #224 224
        #少了 （weight_decay=1e-5,regularizer='L2'） Conv2D里面没有
        self.conv1 = Conv2D(16,(3,3),strides=1,activation='relu',padding= "same",kernel_regularizer='l2')
        self.conv2 = Conv2D(16,(3,3),strides=1,activation='relu',padding= "same",kernel_regularizer='l2')
        self.conv3 = Conv2D(32,(3,3),strides=2,activation='relu',padding= "same",kernel_regularizer='l2')
        #112 112 
        self.conv4 = Conv2D(32,(3,3),strides=1,activation='relu',padding= "same",kernel_regularizer='l2')
        self.conv5 = Conv2D(32,(3,3),strides=1,activation='relu',padding= "same",kernel_regularizer='l2')
        self.conv6 = Conv2D(64,(3,3),strides=2,activation='relu',padding= "same",kernel_regularizer='l2')
        #56 56
        self.conv7 = Conv2D(64,(3,3),strides=1,activation='relu',padding= "same",kernel_regularizer='l2')
        self.conv8 = Conv2D(64,(3,3),strides=1,activation='relu',padding= "same",kernel_regularizer='l2')
        '''
        self.conv9 = Conv2D(128,(3,3),strides=2,activation='relu',padding= "same",kernel_regularizer='l2')
        #28 28 
        self.conv10 = Conv2D(128,(3,3),strides=1,activation='relu',padding= "same",kernel_regularizer='l2')
        self.conv11 = Conv2D(128,(3,3),strides=1,activation='relu',padding= "same",kernel_regularizer='l2')
        self.conv12 = Conv2D(256,(5,5),strides=2,activation='relu',padding= "same",kernel_regularizer='l2')
        #14 14
        self.conv13 = Conv2D(256,(3,3),strides=1,activation='relu',padding= "same",kernel_regularizer='l2')
        self.conv14 = Conv2D(256,(3,3),strides=1,activation='relu',padding= "same",kernel_regularizer='l2')
        self.conv15 = Conv2D(512,(5,5),strides=2,activation='relu',padding= "same",kernel_regularizer='l2')
        #7  7
        self.conv16 = Conv2D(512,(3,3),strides=1,activation='relu',padding= "same",kernel_regularizer='l2')
        self.conv17 = Conv2D(512,(3,3),strides=1,activation='relu',padding= "same",kernel_regularizer='l2')
        self.conv18 = Conv2D(512,(3,3),strides=1,activation='relu',padding= "same",kernel_regularizer='l2')
        '''
    def call_cnn(self,img_inp):
        """
        :param x
        :param 
        :return: x
        """
        x = img_inp
        
        x = tf.expand_dims(x,0)
        #224 224 224*224的RGB图片
        x = self.conv1(x)
        x = self.conv2(x)
        x0 = x
        x = self.conv3(x)
        #112 112 
        x = self.conv4(x)
        x = self.conv5(x)
        x1 = x
        x = self.conv6(x)
        #56 56  
        x = self.conv7(x)
        x = self.conv8(x)
        x2 = x
        '''
        x = self.conv9(x)
        #28 28
        x = self.conv10(x)
        x = self.conv11(x)
        x3 = x
        x = self.conv12(x)
        #14 14
        x = self.conv13(x)
        x = self.conv14(x)
        x4 = x
        x = self.conv15(x)
        #7 7
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        x5 = x
        '''
        #updata image feature  #tf.squeeze 把维度是1的值去掉
        #feat : feature
        self.img_feat = [tf.squeeze(x0), tf.squeeze(x1), tf.squeeze(x2)]
        #正则化损失函数无关
        #self.l2_loss += tf.nn.l2_loss(weights)*0.3
        #self.loss += self.l2_loss
        #self.loss += tf.add_n(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)) * 0.3

    def call(self,img_mesh_cameras):
        img_inp = img_mesh_cameras['img_all_view']
        self.call_cnn(img_inp)
        self.cameras = img_mesh_cameras['cameras']
        self.inputs = img_mesh_cameras['mesh']
        blk1_sample = self.sample1(self.inputs)
        blk1_proj_feat = self.proj1(blk1_sample,self.img_feat,self.cameras)
        blk1_out = self.drb1((blk1_proj_feat, self.inputs))

        blk2_sample = self.sample2(blk1_out)
        blk2_proj_feat = self.proj2(blk2_sample,self.img_feat,self.cameras)
        blk2_out = self.drb2((blk2_proj_feat, blk1_out))

        self.output1l = blk1_out
        self.output2l = blk2_out
        conv_layers = [self.drb1, self.drb2]
        trainable_variables = []
        for layer_param in conv_layers:
            trainable_variables.append(layer_param.trainable_variables)
        return self.output1l,self.output2l,trainable_variables
        # Build metrics