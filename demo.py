import tensorflow as tf
import numpy as np
import os
from p2m.config import execute
import pickle
# from utils.dataloader import DataFetcher
import utils.tools as tools
# from utils.visualize import plot_scatter
import p2m.GCN as GCN
from p2m.config import *
from utils.dataloader import *
from p2m.losses import *
def main(cfg):
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
    # Invalid device or cannot modify virtual devices once initialized.
        pass
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_id)
    # ---------------------------------------------------------------
    # Load init ellipsoid and info about vertices and edges
    
    # Construct Feed dict
    print('=> pre-processing:参数初始化ing')
    # ---------------------------------------------------------------
    #num_blocks = 3
    #num_supports = 2
    #name: 'coarse_mvp2m'
    #save_path: 'results'
    root_dir = os.path.join(cfg.save_path, cfg.name)
    print(cfg.save_path)
    model_dir = os.path.join(cfg.save_path, cfg.name, 'models')
    log_dir = os.path.join(cfg.save_path, cfg.name, 'logs')
    plt_dir = os.path.join(cfg.save_path, cfg.name, 'plt')
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
        print('==> make root dir {}'.format(root_dir))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print('==> make model dir {}'.format(model_dir))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print('==> make log dir {}'.format(log_dir))
    if not os.path.exists(plt_dir):
        os.makedirs(plt_dir)
        print('==> make plt dir {}'.format(plt_dir))
    train_loss = open('{}/train_loss_record.txt'.format(log_dir), 'a')
    train_loss.write('Net {} | Start training | lr =  {}\n'.format(cfg.name, cfg.lr))


    #data_loading
    print("=> data loading:数据加载ing") 
    # ---------------------------------------------------------------   
    # Load init ellipsoid and info about vertices and edges
    pkl = pickle.load(open('data/iccv_p2mpp.dat', 'rb'))
    # Construct Feed dict
    edges,faces,features,support1,support2,support3,pool_idx,lape_idx,faces_triangle,sample_adj,sample_coord = tools.construct_feed_dict(pkl)
    num_features_nonzero = None
    dropout = 0
    # ---------------------------------------------------------------   
    #Define model
    print("=> model loading:模型加载ing") 
    #数据类型转换为Tensor类型!   
    features = tf.convert_to_tensor(features,dtype=tf.float32)
    pre_train = False
    if pre_train == True :
        model=tf.keras.models.load_model('/workspace/3D/tf2_gcn-main/results/coarse_mvp2m/models/20200222model')
    else:
        model = GCN.GCN_model(placeholders_features=features,num_features_nonzero=num_features_nonzero,placeholder_dropout=dropout,
            pool_idx=pool_idx,args=cfg,support1=support1,support2=support2,support3=support3,lape_idx=lape_idx,edges = edges,faces_triangle=faces_triangle)
    model.load_weights('/workspace/3D/tf2_gcn-main/results/coarse_mvp2m/models/20200223model_weights_epoch/1')
    print('=> build model complete')
    print('=> start demo ')
    # -------------------------------------------------------------------
    # ---------------------------------------------------------------
    
    print('=> load data')
    demo_img_list = ['data/demo/plane1.png',
                     'data/demo/plane2.png',
                    'data/demo/plane3.png']
    
    img_all_view = tools.load_demo_image(demo_img_list)
    cameras = np.loadtxt('data/demo/cameras.txt')
    

    img_with_cameras = {}
    img_with_cameras.update({'img_all_view':img_all_view}) 
    img_with_cameras.update({'cameras':cameras}) 
    output1,output2,output3,output1_2,output2_2,trainable_variables= model(img_with_cameras)
    #loss = get_loss(output1,output2,output3,output1_2,output2_2,features,trainable_variables,labels,edges,faces_triangle,lape_idx)
    vert = output3
    vert = np.hstack((np.full([vert.shape[0],1], 'v'), vert))
    face = np.loadtxt('data/face3.obj', dtype='|S32')
    mesh = np.vstack((vert, face))

    pred_path = 'data/demo/predict.obj'
    np.savetxt(pred_path, mesh, fmt='%s', delimiter=' ')

    print('=> save to {}'.format(pred_path))

if __name__ == '__main__':
    print('=> set config')
    yaml_path = 'cfgs/mvp2m.yaml'
    args=execute(yaml_path)
    # pprint.pprint(vars(args))
    main(args)
