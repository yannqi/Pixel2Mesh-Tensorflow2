# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.
import tensorflow as tf
from p2m.inits import *
from utils.tools import camera_trans, camera_trans_inv, reduce_var, reduce_std

_LAYER_UIDS = {}

def project(img_feat, x, y, dim):
    x1 = tf.floor(x)
    x2 = tf.minimum(tf.math.ceil(x), tf.cast(tf.shape(img_feat)[0], tf.float32) - 1)
    y1 = tf.floor(y)
    y2 = tf.minimum(tf.math.ceil(y), tf.cast(tf.shape(img_feat)[1], tf.float32) - 1)
    Q11 = tf.gather_nd(img_feat, tf.stack([tf.cast(x1,tf.int32), tf.cast(y1,tf.int32)],1))
    Q12 = tf.gather_nd(img_feat, tf.stack([tf.cast(x1,tf.int32), tf.cast(y2,tf.int32)],1))
    Q21 = tf.gather_nd(img_feat, tf.stack([tf.cast(x2,tf.int32), tf.cast(y1,tf.int32)],1))
    Q22 = tf.gather_nd(img_feat, tf.stack([tf.cast(x2,tf.int32), tf.cast(y2,tf.int32)],1))

    weights = tf.multiply(tf.subtract(x2,x), tf.subtract(y2,y))
    Q11 = tf.multiply(tf.tile(tf.reshape(weights,[-1,1]),[1,dim]), Q11)

    weights = tf.multiply(tf.subtract(x,x1), tf.subtract(y2,y))
    Q21 = tf.multiply(tf.tile(tf.reshape(weights,[-1,1]),[1,dim]), Q21)

    weights = tf.multiply(tf.subtract(x2,x), tf.subtract(y,y1))
    Q12 = tf.multiply(tf.tile(tf.reshape(weights,[-1,1]),[1,dim]), Q12)

    weights = tf.multiply(tf.subtract(x,x1), tf.subtract(y,y1))
    Q22 = tf.multiply(tf.tile(tf.reshape(weights,[-1,1]),[1,dim]), Q22)

    outputs = tf.add_n([Q11, Q21, Q12, Q22])
    return outputs

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        #res = tf.sparse_tensor_dense_matmul(x, y)
        res = tf.sparse.sparse_dense_matmul(x,y)
    else:
        res = tf.matmul(x, y)
    return res







class GraphConvolution(tf.keras.layers.Layer):
    #注意，论文中bias是true
    def __init__(self,input_dim, output_dim,placeholder_dropout,support1,support2,support3,num_features_nonzero, dropout=False,
                 sparse_inputs=False, activation=tf.nn.relu, bias=True, gcn_block_id=1,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        if dropout:
            self.dropout = placeholder_dropout
        else:
            self.dropout = 0.

        self.activation = activation
        if gcn_block_id == 1:
            self.support = support1
        elif gcn_block_id == 2:
            self.support = support2
        elif gcn_block_id == 3:
            self.support = support3

        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.weights_var = []
        # helper variable for sparse dropout
        self.num_features_nonzero = num_features_nonzero

            
        
        for i in range(len(self.support)):
            self.w = self.add_weight('weight' + str(i), [self.input_dim, self.output_dim])
            
            self.weights_var.append(self.w)
        if self.bias:
            self.vars_bias = self.add_weight('bias', [self.output_dim])
        

        
    def call(self, inputs):
        x  = inputs
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x,  self.dropout)
        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.weights_var[i], sparse=self.sparse_inputs)
                
            else:
                pre_sup = self.weights_var[i]
            support_indices = self.support[i][0]
            support_values = self.support[i][1]
            support_shape = self.support[i][2]
            support_sparse = tf.SparseTensor(values=support_values,indices=support_indices,dense_shape=support_shape)
            #support = dot(self.support[i], pre_sup, sparse=True)
            support_sparse = tf.cast(support_sparse,dtype=tf.float32)
            support = dot(support_sparse, pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)
        # bias
        if self.bias:
            output += self.vars_bias
        return self.activation(output)


class GraphPooling(tf.keras.layers.Layer):
    def __init__(self, pool_idx, pool_id=1, **kwargs):
        super(GraphPooling, self).__init__(**kwargs)
        self.pool_idx = pool_idx[pool_id - 1]

    def call(self, inputs):
        X = inputs
        add_feat = (1 / 2.0) * tf.reduce_sum(tf.gather(X, self.pool_idx), 1)
        outputs = tf.concat([X, add_feat], 0)
        return outputs

class GraphProjection(tf.keras.layers.Layer):
    """Graph Pooling layer."""
 
    def __init__(self, **kwargs):
        super(GraphProjection, self).__init__(**kwargs)

        
    def call(self, inputs,img_feat):
        coord = inputs
        self.img_feat = img_feat
        X = inputs[:, 0]
        Y = inputs[:, 1]
        Z = inputs[:, 2]

        h = 250 * tf.divide(-Y, -Z) + 112
        w = 250 * tf.divide(X, -Z) + 112

        h = tf.minimum(tf.maximum(h, 0), 223)
        w = tf.minimum(tf.maximum(w, 0), 223)

        x = h/(224.0/56)  
        y = w/(224.0/56)   
        out1 = project(self.img_feat[0], x, y, 64)

        x = h/(224.0/28)
        y = w/(224.0/28)
        out2 = project(self.img_feat[1], x, y, 128)

        x = h/(224.0/14)
        y = w/(224.0/14)
        out3 = project(self.img_feat[2], x, y, 256)

        x = h/(224.0/7)
        y = w/(224.0/7)
        out4 = project(self.img_feat[3], x, y, 512)
        outputs = tf.concat([coord,out1,out2,out3,out4], 1)
        return outputs

class GraphProjection_plus(tf.keras.layers.Layer):
    def __init__(self,  **kwargs):
        super(GraphProjection_plus, self).__init__(**kwargs)

        
        self.view_number = 3

    def call(self, inputs,img_feat,cameras):
        self.img_feat = img_feat
        self.camera = cameras
        coord = inputs
        out1_list = []
        out2_list = []
        out3_list = []
        out4_list = []
        for i in range(self.view_number):
            point_origin = camera_trans_inv(self.camera[0], inputs)
            point_crrent = camera_trans(self.camera[i], point_origin)
            X = point_crrent[:, 0]
            Y = point_crrent[:, 1]
            Z = point_crrent[:, 2]
            h = 248.0 * tf.divide(-Y, -Z) + 112.0
            w = 248.0 * tf.divide(X, -Z) + 112.0

            h = tf.minimum(tf.maximum(h, 0), 223)
            w = tf.minimum(tf.maximum(w, 0), 223)
            n = tf.cast(tf.fill(tf.shape(h), i), tf.float32)

            indeces = tf.stack([n, h, w], 1)

            idx = tf.cast(indeces / (224.0 / 56.0), tf.int32)
            out1 = tf.gather_nd(self.img_feat[0], idx)
            out1_list.append(out1)
            idx = tf.cast(indeces / (224.0 / 28.0), tf.int32)
            out2 = tf.gather_nd(self.img_feat[1], idx)
            out2_list.append(out2)
            idx = tf.cast(indeces / (224.0 / 14.0), tf.int32)
            out3 = tf.gather_nd(self.img_feat[2], idx)
            out3_list.append(out3)
            idx = tf.cast(indeces / (224.0 / 7.00), tf.int32)
            out4 = tf.gather_nd(self.img_feat[3], idx)
            out4_list.append(out4)
        # ----
        all_out1 = tf.stack(out1_list, 0)
        all_out2 = tf.stack(out2_list, 0)
        all_out3 = tf.stack(out3_list, 0)
        all_out4 = tf.stack(out4_list, 0)

        # 3*N*[64+128+256+512] -> 3*N*F
        image_feature = tf.concat([all_out1, all_out2, all_out3, all_out4], 2)
        # 3*N*F -> N*F
        # image_feature = tf.reshape(tf.transpose(image_feature, [1, 0, 2]), [-1, FLAGS.feat_dim * 3])

        #image_feature = tf.reduce_max(image_feature, axis=0)
        image_feature_max = tf.reduce_max(image_feature, axis=0)
        image_feature_mean = tf.reduce_mean(image_feature, axis=0)
        image_feature_std = reduce_std(image_feature, axis=0)

        outputs = tf.concat([coord, image_feature_max, image_feature_mean, image_feature_std], 1)
        return outputs


class SampleHypothesis(tf.keras.layers.Layer):
    def __init__(self, sample_coord, **kwargs):
        super(SampleHypothesis, self).__init__(**kwargs)
        self.sample_delta = sample_coord

    def __call__(self, mesh_coords):
        """
        Local Grid Sample for fast matching init mesh
        :param mesh_coords:
        [N,S,3] ->[NS,3] for projection
        :return: sample_points_per_vertices: [NS, 3]
        """
        with tf.name_scope(self.name):
            center_points = tf.expand_dims(mesh_coords, axis=1)
            center_points = tf.tile(center_points, [1, 43, 1])

            delta = tf.expand_dims(self.sample_delta, 0)

            sample_points_per_vertices = tf.add(center_points, delta)

            outputs = tf.reshape(sample_points_per_vertices, [-1, 3])
        return outputs


class LocalGConv(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, dropout1,sample_adj,dropout=False, act=tf.nn.relu, bias=True, **kwargs):
        super(LocalGConv, self).__init__(**kwargs)

        if dropout:
            self.dropout = dropout1
        else:
            self.dropout = 0.

        self.act = act
        self.support = sample_adj

        self.bias = bias
        self.local_graph_vert = 43

        self.output_dim = output_dim
        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim], name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs  # N, S, VF
        # dropout
        x = tf.nn.dropout(x, 1 - self.dropout)
        # convolve
        supports = list()
        for i in range(len(self.support)):
            pre_sup = tf.einsum('ijk,kl->ijl', x, self.vars['weights_' + str(i)])
            support = tf.einsum('ij,kjl->kil', self.support[i], pre_sup)
            supports.append(support)
        output = tf.add_n(supports)
        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class DeformationReasoning(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, sample_coord, gcn_block=-1, args=None, **kwargs):
        super(DeformationReasoning, self).__init__(**kwargs)
        self.delta_coord = sample_coord
        self.s = 43
        self.f = args.stage2_feat_dim
        self.hidden_dim = 192
        with tf.variable_scope(self.name):
            self.local_conv1 = LocalGConv(input_dim=input_dim, output_dim=self.hidden_dim)
            self.local_conv2 = LocalGConv(input_dim=self.hidden_dim, output_dim=self.hidden_dim)
            self.local_conv3 = LocalGConv(input_dim=self.hidden_dim, output_dim=self.hidden_dim)
            self.local_conv4 = LocalGConv(input_dim=self.hidden_dim, output_dim=self.hidden_dim)
            self.local_conv5 = LocalGConv(input_dim=self.hidden_dim, output_dim=self.hidden_dim)
            self.local_conv6 = LocalGConv(input_dim=self.hidden_dim, output_dim=1)

    def _call(self, inputs):
        proj_feat, prev_coord = inputs[0], inputs[1]
        with tf.name_scope(self.name):
            x = proj_feat  # NS, F
            x = tf.reshape(x, [-1, self.s, self.f])  # N,S,F
            x1 = self.local_conv1(x)
            x2 = self.local_conv2(x1)
            x3 = tf.add(self.local_conv3(x2), x1)
            x4 = self.local_conv4(x3)
            x5 = tf.add(self.local_conv5(x4), x3)
            x6 = self.local_conv6(x5)  # N, S, 1
            score = tf.nn.softmax(x6, axis=1)  # N, S, 1
            tf.summary.histogram('score', score)
            delta_coord = score * self.delta_coord
            next_coord = tf.reduce_sum(delta_coord, axis=1)
            next_coord += prev_coord
            return next_coord


class LocalGraphProjection(tf.keras.layers.Layer):
    def __init__(self, img_feat,cameras, **kwargs):
        super(LocalGraphProjection, self).__init__(**kwargs)

        self.img_feat = img_feat
        self.camera = cameras
        self.view_number = 3

    def _call(self, inputs):
        coord = inputs
        out1_list = []
        out2_list = []
        out3_list = []
        # out4_list = []

        for i in range(self.view_number):
            point_origin = camera_trans_inv(self.camera[0], inputs)
            point_crrent = camera_trans(self.camera[i], point_origin)
            X = point_crrent[:, 0]
            Y = point_crrent[:, 1]
            Z = point_crrent[:, 2]
            h = 248.0 * tf.divide(-Y, -Z) + 112.0
            w = 248.0 * tf.divide(X, -Z) + 112.0

            h = tf.minimum(tf.maximum(h, 0), 223)
            w = tf.minimum(tf.maximum(w, 0), 223)
            n = tf.cast(tf.fill(tf.shape(h), i), tf.int32)

            x = h / (224.0 / 224)
            y = w / (224.0 / 224)
            out1 = self.bi_linear_sample(self.img_feat[0], n, x, y)
            out1_list.append(out1)
            x = h / (224.0 / 112)
            y = w / (224.0 / 112)
            out2 = self.bi_linear_sample(self.img_feat[1], n, x, y)
            out2_list.append(out2)
            x = h / (224.0 / 56)
            y = w / (224.0 / 56)
            out3 = self.bi_linear_sample(self.img_feat[2], n, x, y)
            out3_list.append(out3)
        # ----
        all_out1 = tf.stack(out1_list, 0)
        all_out2 = tf.stack(out2_list, 0)
        all_out3 = tf.stack(out3_list, 0)

        # 3*N*[16+32+64] -> 3*N*F
        image_feature = tf.concat([all_out1, all_out2, all_out3], 2)

        image_feature_max = tf.reduce_max(image_feature, axis=0)
        image_feature_mean = tf.reduce_mean(image_feature, axis=0)
        image_feature_std = reduce_std(image_feature, axis=0)

        outputs = tf.concat([coord, image_feature_max, image_feature_mean, image_feature_std], 1)
        return outputs

    def bi_linear_sample(self, img_feat, n, x, y):
        x1 = tf.floor(x)
        x2 = tf.ceil(x)
        y1 = tf.floor(y)
        y2 = tf.ceil(y)
        Q11 = tf.gather_nd(img_feat, tf.stack([n, tf.cast(x1, tf.int32), tf.cast(y1, tf.int32)], 1))
        Q12 = tf.gather_nd(img_feat, tf.stack([n, tf.cast(x1, tf.int32), tf.cast(y2, tf.int32)], 1))
        Q21 = tf.gather_nd(img_feat, tf.stack([n, tf.cast(x2, tf.int32), tf.cast(y1, tf.int32)], 1))
        Q22 = tf.gather_nd(img_feat, tf.stack([n, tf.cast(x2, tf.int32), tf.cast(y2, tf.int32)], 1))

        _weights = tf.multiply(tf.subtract(x2, x), tf.subtract(y2, y))
        Q11 = tf.multiply(tf.expand_dims(_weights, 1), Q11)
        _weights = tf.multiply(tf.subtract(x, x1), tf.subtract(y2, y))
        Q21 = tf.multiply(tf.expand_dims(_weights, 1), Q21)
        _weights = tf.multiply(tf.subtract(x2, x), tf.subtract(y, y1))
        Q12 = tf.multiply(tf.expand_dims(_weights, 1), Q12)
        _weights = tf.multiply(tf.subtract(x, x1), tf.subtract(y, y1))
        Q22 = tf.multiply(tf.expand_dims(_weights, 1), Q22)
        outputs = tf.add_n([Q11, Q21, Q12, Q22])
        return outputs