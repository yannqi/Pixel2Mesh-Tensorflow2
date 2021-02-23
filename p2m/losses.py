import tensorflow as tf
from tensorflow.python.keras.backend import dtype
from p2m.chamfer import nn_distance
import numpy as np
from scipy import stats

import tensorflow_probability as tfp
#loss这里 nn_distance没有好好思考，待考虑。

def laplace_coord(pred, lape_idx, block_id):
    vertex = tf.concat([pred, tf.zeros([1, 3])], 0)
    indices = lape_idx[block_id - 1][:, :8]
    weights = tf.cast(lape_idx[block_id - 1][:, -1], tf.float32)

    weights = tf.tile(tf.reshape(tf.math.reciprocal(weights), [-1, 1]), [1, 3])
    #因为有-1，对indice进行了+1处理
    indices = indices + 1
    laplace = tf.reduce_sum(tf.gather(vertex, indices), 1)
    laplace = tf.subtract(pred, tf.multiply(laplace, weights))
    return laplace


def laplace_loss(pred1, pred2, lape_idx, block_id):
    # laplace term
    lap1 = laplace_coord(pred1, lape_idx, block_id)
    lap2 = laplace_coord(pred2, lape_idx, block_id)
    laplace_loss = tf.math.reduce_mean(tf.reduce_sum(tf.math.square(tf.subtract(lap1, lap2)), 1)) * 1500
    move_loss = tf.math.reduce_mean(tf.reduce_sum(tf.math.square(tf.subtract(pred1, pred2)), 1)) * 50
    return laplace_loss + move_loss

def unit(tensor):
    return tf.math.l2_normalize(tensor,axis=1)
def mesh_loss(pred,labels,edges,faces_triangle,block_id):
    gt_pt = labels[:,:3]  # gt points   TODO what is gt?
    gt_nm = labels[:,3:]  # gt normals
    #----------------------------------
    #--------edge in graoh-------------
    #----------------------------------
    #tf.gather(params, indices, validate_indices=None, axis=None, batch_dims=0, name=None)
    #Gather slices from params axis  according to indices.
    nod1 = tf.gather(pred,edges[block_id-1][:,0])
    nod2 = tf.gather(pred,edges[block_id-1][:,1])
    edge = tf.math.subtract(nod1,nod2)  #Returns x - y element-wise.
    #----------------------------------
    #--------edge length loss ---------
    #----------------------------------
    #Computes the sum of elements across dimensions of a tensor.
    #tf.math.reduce_sum(input_tensor, axis=None, keepdims=False, name=None)
    edge_length = tf.math.reduce_sum(tf.math.square(edge),axis=1)#axis=1 按列相加
    #Computes the mean of elements across dimensions of a tensor.
    #tf.math.reduce_mean(input_tensor, axis=None, keepdims=False, name=None)
    edge_loss = tf.math.reduce_mean(edge_length)*350  #无axis,则返回一个数值，矩阵内所有值的平均
    #----------------------------------
    #--------chamer distance ----------
    #----------------------------------
    #nn_distance 这个有问题
    sample_pt = sample(pred, faces_triangle, block_id)
    sample_pred = tf.concat([pred, sample_pt], axis=0)
    dist1, idx1, dist2, idx2 = nn_distance(gt_pt, sample_pred)
    point_loss = (tf.math.reduce_mean(dist1) + 0.55 * tf.math.reduce_mean(dist2)) * 3000
    #----------------------------------
    #--------normal cosine distance----
    #----------------------------------
    # normal cosine loss
    normal = tf.gather(gt_nm, tf.squeeze(idx2, 0))
    normal = tf.gather(normal,edges[block_id - 1][:, 0])
    normal = tf.cast(normal,dtype=tf.float32)
    cosine = tf.math.abs(tf.math.reduce_sum(tf.math.multiply(unit(normal), unit(edge)), 1))
    normal_loss = tf.math.reduce_mean(cosine) * 0.5

    total_loss = point_loss + edge_loss + normal_loss
    return total_loss


def sample(pred, faces_triangle, block_id):
    uni = tfp.distributions.Uniform(low=0.0, high=1.0)
    #uni = tf.compat.v1.distributions.Uniform(low=0.0, high=1.0)
    faces = faces_triangle[block_id - 1]
    #tilefaces = choice_faces(verts=pred, faces=faces)
    #tilefaces =tf.cast(tilefaces,dtype = tf.int32)
    tilefaces = tf.numpy_function(choice_faces, [pred, faces], tf.int64)

    num_of_tile_faces = tf.shape(tilefaces)[0]

    xs = tf.gather(pred, tilefaces[:, 0])
    ys = tf.gather(pred, tilefaces[:, 1])
    zs = tf.gather(pred, tilefaces[:, 2])

    u = tf.sqrt(uni.sample([num_of_tile_faces, 1]))
    v = uni.sample([num_of_tile_faces, 1])
    points = (1 - u) * xs + (u * (1 - v)) * ys + u * v * zs
    return points

def choice_faces(verts, faces):
    num = 4000
    u1, u2, u3 = np.split(verts[faces[:, 0]] - verts[faces[:, 1]], 3, axis=1)
    v1, v2, v3 = np.split(verts[faces[:, 1]] - verts[faces[:, 2]], 3, axis=1)
    a = (u2 * v3 - u3 * v2) ** 2
    b = (u3 * v1 - u1 * v3) ** 2
    c = (u1 * v2 - u2 * v1) ** 2
    Areas = np.sqrt(a + b + c) / 2
    Areas = Areas / np.sum(Areas)
    choices = np.expand_dims(np.arange(Areas.shape[0]), 1)
    dist = stats.rv_discrete(name='custm', values=(choices, Areas))
    choices = dist.rvs(size=num)
    select_faces = faces[choices]
    return select_faces


    
def laplace_loss_2(pred1, pred2, lape_idx, block_id):
    # laplace term
    lap1 = laplace_coord(pred1, lape_idx, block_id)
    lap2 = laplace_coord(pred2, lape_idx, block_id)
    laplace_loss = tf.math.reduce_mean(tf.reduce_sum(tf.math.square(tf.subtract(lap1, lap2)), 1)) * 1500
    move_loss = tf.math.reduce_mean(tf.reduce_sum(tf.math.square(tf.subtract(pred1, pred2)), 1)) * 100
    return laplace_loss + move_loss


def mesh_loss_2(pred,labels,edges,faces_triangle,block_id):
    gt_pt = labels[:,:3]  # gt points   TODO what is gt?
    gt_nm = labels[:,3:]  # gt normals
    #----------------------------------
    #--------edge in graoh-------------
    #----------------------------------
    #tf.gather(params, indices, validate_indices=None, axis=None, batch_dims=0, name=None)
    #Gather slices from params axis  according to indices.
    nod1 = tf.gather(pred,edges[block_id-1][:,0])
    nod2 = tf.gather(pred,edges[block_id-1][:,1])
    edge = tf.math.subtract(nod1,nod2)  #Returns x - y element-wise.
    #----------------------------------
    #--------edge length loss ---------
    #----------------------------------
    #Computes the sum of elements across dimensions of a tensor.
    #tf.math.reduce_sum(input_tensor, axis=None, keepdims=False, name=None)
    edge_length = tf.math.reduce_sum(tf.math.square(edge),axis=1)#axis=1 按列相加
    #Computes the mean of elements across dimensions of a tensor.
    #tf.math.reduce_mean(input_tensor, axis=None, keepdims=False, name=None)
    edge_loss = tf.math.reduce_mean(edge_length)*500  #无axis,则返回一个数值，矩阵内所有值的平均
    #----------------------------------
    #--------chamer distance ----------
    #----------------------------------
    #nn_distance 这个有问题
    sample_pt = sample(pred, faces_triangle, block_id)
    sample_pred = tf.concat([pred, sample_pt], axis=0)
    dist1, idx1, dist2, idx2 = nn_distance(gt_pt, sample_pred)
    point_loss = (tf.math.reduce_mean(dist1) + 0.55 * tf.math.reduce_mean(dist2)) * 3000
    #----------------------------------
    #--------normal cosine distance----
    #----------------------------------
    # normal cosine loss
    normal = tf.gather(gt_nm, tf.squeeze(idx2, 0))
    normal = tf.gather(normal,edges[block_id - 1][:, 0])
    normal = tf.cast(normal,dtype=tf.float32)
    cosine = tf.math.abs(tf.math.reduce_sum(tf.math.multiply(unit(normal), unit(edge)), 1))
    normal_loss = tf.math.reduce_mean(cosine) * 0.5

    total_loss = point_loss + edge_loss + normal_loss
    return total_loss