# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.
import numpy as np
import tensorflow as tf
import pickle
import cv2



def construct_feed_dict(pkl):
    coord = pkl['coord']
    pool_idx = pkl['pool_idx']
    faces = pkl['faces']
    lape_idx = pkl['lape_idx']
    features = coord
    edges = []
    support1 = pkl['stage1']
    support2 = pkl['stage2']
    support3 = pkl['stage3']
    faces_triangle = pkl['faces_triangle']
    sample_coord = pkl['sample_coord']
    sample_adj = pkl['sample_cheb_dense']
    
    for i in range(1, 4):
        adj = pkl['stage{}'.format(i)][1]
        edges.append(adj[0])
    

    
    return edges,faces,features,support1,support2,support3,pool_idx,lape_idx,faces_triangle,sample_adj,sample_coord


def reduce_var(x, axis=None, keepdims=False):
    """Variance of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the variance.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the variance of elements of `x`.
    """
    m = tf.reduce_mean(x, axis=axis, keepdims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)


def reduce_std(x, axis=None, keepdims=False):
    """Standard deviation of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the standard deviation.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the standard deviation of elements of `x`.
    """
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims) + 1e-6)

# -------------------------------------------------------------------
# cameras
# -------------------------------------------------------------------


def normal(v):
    #ensorflow中的norm函数作用是用来求L1_norm范数和Eukl_norm范数。
    #https://blog.csdn.net/a1920993165/article/details/105168218/
    #if 在TensorFlow中属于bool类型变量，不被允许使用。用tf.cond函数
    norm = tf.norm(v)
    v = tf.cond(norm == 0, lambda: v, lambda: tf.math.divide(v, norm))
    return v 
    '''
    if norm == 0:
        return v
    return tf.math.divide(v, norm)
    '''


def cameraMat(param):
    theta = param[0] * np.pi / 180.0 #角度 
    
    # tf.sin/cos (Input range is (-inf, inf)-度数 and output range is [-1,1]-幅值.)
    #camera y 
    camy = param[3] * tf.math.sin(param[1] * np.pi / 180.0)  
    lens = param[3] * tf.math.cos(param[1] * np.pi / 180.0)
    camx = lens * tf.math.cos(theta)
    camz = lens * tf.math.sin(theta)
    Z = tf.stack([camx, camy, camz])
    
    x = camy * tf.math.cos(theta + np.pi)
    z = camy * tf.math.sin(theta + np.pi)
    Y = tf.stack([x, lens, z])
    
    X = tf.experimental.numpy.cross(Y, Z)
    #TF2.0
    #X=tf.linalg.cross(Y,Z)
    X = tf.convert_to_tensor(X)
    # Stacks a list of rank-R tensors into one rank-(R+1) tensor.
    cm_mat = tf.stack([normal(X), normal(Y), normal(Z)])
    return cm_mat, Z


def camera_trans(camera_metadata, xyz):
    c, o = cameraMat(camera_metadata)
    points = xyz[:, :3]
    o = tf.cast(o,dtype= tf.float32)
    c = tf.cast(c,dtype= tf.float32)
    pt_trans = points - o
    pt_trans = tf.matmul(pt_trans, tf.transpose(c))
    return pt_trans


def camera_trans_inv(camera_metadata, xyz):
    c, o = cameraMat(camera_metadata)
    #tf.matmul  矩阵相乘  区分 tf.multiply 矩阵对应元素相乘   
    #tf.matrix_inverse 
    #inv_xyz = (tf.linalg.matmul(xyz, tf.matrix_inverse(tf.transpose(c)))) + o
    inv_c = tf.linalg.inv(tf.transpose(c))
    inv_c = tf.cast(inv_c,dtype=tf.float32)
    xyz = tf.cast(xyz,dtype= tf.float32)
    o = tf.cast(o,dtype= tf.float32)
    inv_xyz = (tf.linalg.matmul(xyz,inv_c)) + o
    return inv_xyz


def load_demo_image(demo_image_list):
    imgs = np.zeros((3, 224, 224, 3))
    for idx, demo_img_path in enumerate(demo_image_list):
        img = cv2.imread(demo_img_path, cv2.IMREAD_UNCHANGED)
        if img.shape[2] == 4:
            img[np.where(img[:, :, 3] == 0)] = 255
        img = cv2.resize(img, (224, 224))
        img_inp = img.astype('float32') / 255.0
        imgs[idx] = img_inp[:, :, :3]
    return imgs
