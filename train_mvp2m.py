import tensorflow as tf
import numpy as np
import pprint
import pickle
import os
from tensorflow.python.autograph.core.converter import Feature
from tensorflow.python.keras.backend import dropout
from p2m.config import *
from utils.dataloader import *
import utils.tools as tools
import p2m.GCN as GCN
from p2m.losses import *
from utils.visualize import plot_scatter
def main(cfg):
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
    # Invalid device or cannot modify virtual devices once initialized.
        pass
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_id)
    # pre-processing
    print('=> pre-processing:参数初始化ing')
    seed = 123
    np.random.seed(seed)
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
    data = DataFetcher(file_list=cfg.train_file_path, data_root=cfg.train_data_path, image_root=cfg.train_image_path, is_val=False)
    data.setDaemon(True)
    data.start()
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
    features = tf.convert_to_tensor(features,dtype=tf.float32)
    #-------------------------------
    #是否加载预训练模型  Default：False
    #-------------------------------
    pre_train = False
    if pre_train == True :
        model=tf.keras.models.load_model('/workspace/3D/tf2_gcn-main/results/coarse_mvp2m/models/20200222model')
    else:
        model = GCN.GCN_model(placeholders_features=features,num_features_nonzero=num_features_nonzero,placeholder_dropout=dropout,
            pool_idx=pool_idx,args=cfg,support1=support1,support2=support2,support3=support3,lape_idx=lape_idx,edges = edges,faces_triangle=faces_triangle)
    print('模型加载完成')
    #-------------------------------
    #是否加载预训练权重  Default：False
    #-------------------------------
    pre_weight = True
    if pre_weight == True:
        model.load_weights('/workspace/3D/tf2_gcn-main/results/coarse_mvp2m/models/20200223model_weights/epoch1')
    #权重保存位置
    model_weights_save_path = '/workspace/3D/tf2_gcn-main/results/coarse_mvp2m/models/20200223model_weights/epoch'
    
    
    print('=> start train ')
    def get_loss(output1,output2,output3,output1_2,output2_2,features,trainable_variables,labels,edges,faces_triangle,lape_idx):
        '''损失函数'''
        # # Weight decay loss
        loss = tf.zeros([])
        # Cross entropy error
         # Pixel2mesh loss
        loss += mesh_loss(pred =output1, labels = labels,edges=edges,faces_triangle=faces_triangle, block_id = 1)
        loss += mesh_loss(pred =output2, labels = labels,edges=edges,faces_triangle=faces_triangle, block_id = 2)
        loss += mesh_loss(pred =output3, labels = labels,edges=edges,faces_triangle=faces_triangle,block_id = 3)
        loss += laplace_loss(pred1 = features, pred2 = output1, lape_idx=lape_idx,block_id = 1)
        loss += laplace_loss(pred1 = output1_2, pred2= output2, lape_idx=lape_idx, block_id = 2)
        loss += laplace_loss(pred1 = output2_2, pred2 = output3, lape_idx=lape_idx, block_id = 3)
        # Weight decay loss
        #conv_layers = list(range(1, 15)) + list(range(17, 31)) + list(range(33, 48))
        #for layer_id in conv_layers:
        for var1 in trainable_variables:
            for var in var1:
                loss += 5e-6 * tf.nn.l2_loss(var)
        return loss
    optimizer = tf.keras.optimizers.Adam(lr=cfg.lr)
    @tf.function(experimental_relax_shapes=True)
    def train_step(img_with_cameras,labels):
        with tf.GradientTape() as tape:
            output1,output2,output3,output1_2,output2_2,trainable_variables= model(img_with_cameras)
            loss = get_loss(output1,output2,output3,output1_2,output2_2,features,trainable_variables,labels,edges,faces_triangle,lape_idx)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss,output3 
    
    train_number = data.number
    step = 0
    for epoch in range(cfg.epochs):
        # 在下一个epoch开始时，重置评估指标
        current_epoch = epoch + 1 + cfg.init_epoch
        epoch_plt_dir = os.path.join(plt_dir, str(current_epoch))
        if not os.path.exists(epoch_plt_dir):
            os.mkdir(epoch_plt_dir)
        mean_loss = 0
        all_loss = np.zeros(train_number, dtype='float32')
        #for iters in range(train_number):
        for iters in range(50):
            #train_number  35010
            step += 1 
            img_all_view, labels, cameras, data_id ,mesh= data.fetch()
            img_with_cameras = {}
            img_with_cameras.update({'img_all_view':img_all_view}) 
            img_with_cameras.update({'cameras':cameras}) 
            #cameras : [3,5] 
            loss,output3 = train_step(img_with_cameras,labels)
            all_loss[iters] = loss
            mean_loss = np.mean(all_loss[np.where(all_loss)])
            if iters % 100 == 0:
                print('Epoch {}, Iteration {}, Mean loss = {}, iter loss = {}, {}, data id {}'.format(current_epoch, iters + 1, mean_loss, loss, data.queue.qsize(), data_id))
            if iters+1 % 1000 == 0:
                train_loss.write('Epoch {}, Iteration {}, Mean loss = {}, iter loss = {}, {}, data id {}\n'.format(current_epoch, iters + 1, mean_loss, loss, data.queue.qsize(), data_id))
                plot_scatter(pt=output3, data_name=data_id, plt_path=epoch_plt_dir)
            train_loss.flush()
        model.save_weights(model_weights_save_path+str(current_epoch))
    print('模型保存完成，保存路径：',model_weights_save_path)   
    # ---------------------------------------------------------------
    data.shutdown()
    #model save

    print('CNN-GCN Optimization Finished!')
        # ---------------------------------------------------------------


if __name__ == '__main__':
    print('=> set config')
    yaml_path = '/workspace/3D/tf2_gcn-main/cfgs/mvp2m.yaml'
    args = execute(yaml_path)
    pprint.pprint(vars(args))
    main(args)

