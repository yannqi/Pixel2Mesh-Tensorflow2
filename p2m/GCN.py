import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Flatten,Dense
from p2m.layers import *
from p2m.losses import *
class GCN_model(tf.keras.Model):
    def __init__(self,num_features_nonzero,placeholder_dropout,placeholders_features,
        pool_idx,support1,support2,support3,args,lape_idx,edges ,faces_triangle, **kwargs):
        super(GCN_model,self).__init__(**kwargs)
        self.args = args
        self.placeholders_features =placeholders_features
        self.pool_idx = pool_idx
        self.placeholder_dropout = placeholder_dropout
        self.support1 = support1
        self.support2 = support2
        self.support3 = support3
        #print('input dim:', input_dim)
        #print('output dim:', output_dim)
        self.num_features_nonzero = num_features_nonzero
        self.lape_idx = lape_idx
        self.edges = edges
        self.faces_triangle = faces_triangle
        # first project block
        self.layers_GCN = []
        self.cnn_model()
        self.GCNmodel()
        
    def call(self,img_with_cameras):
        img_inp = img_with_cameras['img_all_view']
        self.call_cnn(img_inp)
        self.cameras = img_with_cameras['cameras']
        # Build sequential resnet model
        self.inputs = self.placeholders_features
        eltwise = [3, 5, 7, 9, 11, 13,
                   19, 21, 23, 25, 27, 29,
                   35, 37, 39, 41, 43, 45]
        concat = [15, 31]
        # proj = [0, 15, 31]
        self.activations = []
        self.activations.append(self.inputs)
        for idx, layer in enumerate(self.layers_GCN[:48]):
            if layer.name[0:10]=='graph_proj':
                hidden = layer(self.activations[-1],self.img_feat,self.cameras)
            else:
                hidden = layer(self.activations[-1])
            if idx in eltwise:
                hidden = tf.add(hidden, self.activations[-2]) * 0.5
            if idx in concat:
                hidden = tf.concat([hidden, self.activations[-2]], 1)
            self.activations.append(hidden)
        self.output1 = self.activations[15]
        unpool_layer = GraphPooling(pool_idx=self.pool_idx, pool_id=1)
        self.output1_2 = unpool_layer(self.output1)

        self.output2 = self.activations[31]
        unpool_layer = GraphPooling(pool_idx=self.pool_idx, pool_id=2)
        self.output2_2 = unpool_layer(self.output2)
        self.output3 = self.activations[48]
        trainable_variables = []
        conv_layers = list(range(1, 15)) + list(range(17, 31)) + list(range(33, 48))
        for layer_id in conv_layers:
            trainable_variables.append(self.layers_GCN[layer_id].trainable_variables)
        return self.output1,self.output2,self.output3,self.output1_2,self.output2_2,trainable_variables

    def cnn_model(self):    
        #--------------------------------------------------------
        #----------------------CNN-------------------------------
        #--------------------------------------------------------
        # x = self.img_inp #输入的数据？[224,224,3]
        # x = tf.expand_dims(x, 0)#[1,224,224,3]  
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
    def GCNmodel(self):
        #--------------------------------------------------------
        #----------------------GCN-------------------------------
        #--------------------------------------------------------
        self.layers_GCN.append(GraphProjection( name='graph_proj_1_layer_0'))
        self.layers_GCN.append(GraphConvolution(input_dim=self.args.feat_dim,
                                        output_dim=self.args.hidden_dim,
                                        placeholder_dropout=self.placeholder_dropout,
                                        support1=self.support1,
                                        support2=self.support2,
                                        support3=self.support3, 
                                        gcn_block_id = 1 ,              
                                        num_features_nonzero= self.num_features_nonzero,                 
                                        name='graph_conv_blk1_1_layer_1', 
                                        ))
        for _ in range(12):
                self.layers_GCN.append(GraphConvolution(input_dim=self.args.hidden_dim,
                                                output_dim=self.args.hidden_dim,
                                                gcn_block_id=1,
                                                placeholder_dropout=self.placeholder_dropout,
                                                support1=self.support1,
                                                support2=self.support2,
                                                support3=self.support3,
                                                num_features_nonzero= self.num_features_nonzero,
                                                name='graph_conv_blk1_{}_layer_{}'.format(2 + _, 2 + _),
                                                ))
        # activation #15; layer #14; output 1
        self.layers_GCN.append(GraphConvolution(input_dim=self.args.hidden_dim,
                                        output_dim=self.args.coord_dim,
                                        activation=lambda x: x,
                                        gcn_block_id=1,
                                        placeholder_dropout=self.placeholder_dropout,
                                        support1=self.support1,
                                        support2=self.support2,
                                        support3=self.support3,
                                        num_features_nonzero= self.num_features_nonzero,
                                        name='graph_conv_blk1_14_layer_14'))
        # second project block
        self.layers_GCN.append(GraphProjection( name='graph_proj_2_layer_15'))
        self.layers_GCN.append(GraphPooling(pool_idx=self.pool_idx, pool_id=1, name='graph_pool_1to2_layer_16'))
        self.layers_GCN.append(GraphConvolution(input_dim=self.args.feat_dim + self.args.hidden_dim,
                                        output_dim=self.args.hidden_dim,
                                        gcn_block_id=2,
                                        placeholder_dropout=self.placeholder_dropout,
                                        support1=self.support1,
                                        support2=self.support2,
                                        support3=self.support3,       
                                        num_features_nonzero= self.num_features_nonzero,                                        
                                        name='graph_conv_blk2_1_layer_17'))
        for _ in range(12):
                self.layers_GCN.append(GraphConvolution(input_dim=self.args.hidden_dim,
                                                output_dim=self.args.hidden_dim,
                                                gcn_block_id=2,
                                                placeholder_dropout=self.placeholder_dropout,
                                                support1=self.support1,
                                                support2=self.support2,
                                                support3=self.support3,       
                                                num_features_nonzero= self.num_features_nonzero,                                             
                                                name='graph_conv_blk2_{}_layer_{}'.format(2 + _, 18 + _),
                                                ))
        self.layers_GCN.append(GraphConvolution(input_dim=self.args.hidden_dim,
                                        output_dim=self.args.coord_dim,
                                        activation=lambda x: x,
                                        gcn_block_id=2,
                                        placeholder_dropout=self.placeholder_dropout,
                                        support1=self.support1,
                                        support2=self.support2,
                                        support3=self.support3,   
                                        num_features_nonzero= self.num_features_nonzero,                                           
                                        name='graph_conv_blk2_14_layer_30'))
            # third project block
        self.layers_GCN.append(GraphProjection(name='graph_proj_3_layer_31'))
        self.layers_GCN.append(GraphPooling(pool_idx=self.pool_idx, pool_id=2, name='graph_pool_2to3_layer_32'))
        self.layers_GCN.append(GraphConvolution(input_dim=self.args.feat_dim + self.args.hidden_dim,
                                                output_dim=self.args.hidden_dim,
                                                gcn_block_id=3,
                                                placeholder_dropout=self.placeholder_dropout,
                                                support1=self.support1,
                                                support2=self.support2,
                                                support3=self.support3,
                                                num_features_nonzero= self.num_features_nonzero,
                                                name='graph_conv_blk3_1_layer_33'))
        for _ in range(13):
                self.layers_GCN.append(GraphConvolution(input_dim=self.args.hidden_dim,
                                                output_dim=self.args.hidden_dim,
                                                gcn_block_id=3,
                                                placeholder_dropout=self.placeholder_dropout,
                                                support1=self.support1,
                                                support2=self.support2,
                                                support3=self.support3,
                                                num_features_nonzero= self.num_features_nonzero,
                                                name='graph_conv_blk3_{}_layer_{}'.format(2 + _, 34 + _),
                                                ))
        self.layers_GCN.append(GraphConvolution(input_dim=self.args.hidden_dim,
                                                output_dim=self.args.coord_dim,
                                                activation=lambda x: x,
                                                gcn_block_id=3,
                                                placeholder_dropout=self.placeholder_dropout,
                                                support1=self.support1,
                                                support2=self.support2,
                                                support3=self.support3,
                                                num_features_nonzero= self.num_features_nonzero,
                                                name='graph_conv_blk3_15_layer_47'))
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
        #updata image feature  #tf.squeeze 把维度是1的值去掉
        #feat : feature
        self.img_feat = [tf.squeeze(x2), tf.squeeze(x3), tf.squeeze(x4), tf.squeeze(x5)]
        #正则化损失函数无关
        #self.l2_loss += tf.nn.l2_loss(weights)*0.3
        #self.loss += self.l2_loss
        #self.loss += tf.add_n(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)) * 0.3
        return x


        

