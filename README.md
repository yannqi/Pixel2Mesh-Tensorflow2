# Pixel2Mesh-Tensorflow2
语言选择/Language: [中文](https://github.com/yannqi/Pixel2Mesh-Tensorflow2/blob/main/README.md)  // English（To be completed）


此代码主要是笔者为了本科毕设需要（将Pixel2Mesh迁移到移动设备上）而完成。因为Tensorflow 1.x版本受到限制，因此将**Pixel2Mesh**代码进一步修改成Tensorflow 2.x版本。

代码构成主要参考了如下链接。[Pixel2Mesh](https://github.com/nywang16/Pixel2Mesh)//[Pixel2Mesh++](https://github.com/walsvid/Pixel2MeshPlusPlus)
非常感谢师兄师姐在此领域作出的贡献。(此前已联系Nanyang Wang师兄，本代码的Blog和Github项目可以开源)

代码主要基于CNN+[GCN](https://github.com/tkipf/gcn)框架。


# Dependencies

Requirements：

- Python 3.6+
- Tensorflow(version 2.0+) 
- numpy
- ...


本代码在Tensorflow 2.4,CUDA 11.0，Ubuntu 20.04 ，硬件设备：GeForce GTX 1050 Ti上测试过。**建议懒得配环境的同学用TF官网Docker，本代码全程在Docker中调试运行**（备注：由于过年在家期间用自己电脑跑的，未在实验室，因此预训练权重和Model暂且不放上来了、后续会补充。一个Epoch数据有35010个，跑完一个Epoch大概需要1.5h。loss在跑5-10个epoch后可收敛到4~6左右，三维重建后的模型较为精细。loss最低预估可收敛到4.5附近。）

# Dataset

采用[ShapeNet](https://shapenet.org/)数据集，视角的Rendered方法来自[3D-R2N2](https://github.com/chrischoy/3D-R2N2)
训练和测试的数据集可以在***data/train_list.txt***和***data/test_list.txt***中找到。具体数据集的下载可以从[Pixel2Mesh++](https://github.com/walsvid/Pixel2MeshPlusPlus)里下载。

# Pre-trained Model
待完成

# Quick demo

首先从下载Pre-trained Model，（链接待完成）。将下载好的demo放到指定位置，我的路径是***results/coarse_mvp2m/models/20200222model***，你可以在代码中更改自己预训练模型的路径。
之后在终端执行

`python demo.py`

输入图片及三维Mesh效果图如下所示：（待完成）

# Training
在配置好环境且数据集下载完成后，可以开始模型的训练
## Step1
首先在`cfgs/mvp2m.yaml`中将对应路径设置成自己数据集所在的位置
- train_file_path: the path of your own train split file which contains training data name for each instance
- train_image_path: input image path
- train_data_path: ground-truth model path
- coarse_result_*: the configuration items related to the coarse intermediate mesh should be same as the training data

## Step2
配置完对应的cfg文件，且在***train_mvp2m.py***中修改预训练模型路径后(记着将***pre_train***的选项由False改为True  Default：False)。
`python train_mvp2m.py`

# Test
暂且没写，目前用不上。后续会跟进。

# Statement
This software is for research purpose only.
Please contact me for the licence of commercial purposes. All rights are preserved.

# Contact
Yann Qi (email: yannqi@qq.com)

# License
Apache License version 2.0
