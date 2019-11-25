This repo is based on [DeepLabv3.pytorch](https://github.com/chenxi116/DeepLabv3.pytorch).

# DeepLabv3.pytorch

This is a PyTorch implementation of [DeepLabv3](https://arxiv.org/abs/1706.05587) that aims to reuse the [resnet implementation in torchvision](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py) as much as possible. This means we use the [PyTorch model checkpoint](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L13) when finetuning from ImageNet, instead of [the one provided in TensorFlow](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md).

We try to match every detail in DeepLabv3, except that Multi-Grid other than (1, 1, 1) is not yet supported. On PASCAL VOC 2012 validation set, using the same hyperparameters, we reproduce the performance reported in the paper (GPU with 16GB memory is required). We also support the combination of Group Normalization + Weight Standardization:

Implementation | Normalization | Multi-Grid | ASPP | Image Pooling | mIOU
--- | --- | --- | --- | --- | ---
Paper | BN | (1, 2, 4) | (6, 12, 18) | Yes | 77.21
Ours | BN | (1, 1, 1) | (6, 12, 18) | Yes | 76.49
Ours | GN+WS | (1, 1, 1) | (6, 12, 18) | Yes | 77.20
Ours | BCN+WS | (1, 1, 1) | (6, 12, 18) | Yes | 78.10

To run the experiment, begin by downloading the BCN+WS ResNet101 trained on ImageNet:
```bash
wget https://cs.jhu.edu/~syqiao/BatchChannelNormalization/rn101-bcn-ws.pth.tar -P data/
python main.py --train --exp gn_ws_lr7e-3 --epochs 50 --base_lr 0.007 --groups 32 --weight_std
```
To test the trained model, use the same command except delete `--train`. To use our trained model (78.10):
```bash
wget https://cs.jhu.edu/~syqiao/BatchChannelNormalization/deeplab_resnet101_pascal_v3_gn_ws_lr7e-3_epoch50.pth -P data/
```


## Prepare PASCAL VOC 2012 Dataset
```bash
mkdir data
cd data
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_11-May-2012.tar
cd VOCdevkit/VOC2012/
wget http://cs.jhu.edu/~cxliu/data/SegmentationClassAug.zip
wget http://cs.jhu.edu/~cxliu/data/SegmentationClassAug_Visualization.zip
wget http://cs.jhu.edu/~cxliu/data/list.zip
unzip SegmentationClassAug.zip
unzip SegmentationClassAug_Visualization.zip
unzip list.zip
```

## Prepare Cityscapes Dataset
```bash
unzip leftImg8bit_trainvaltest.zip
unzip gtFine_trainvaltest.zip
git clone https://github.com/mcordts/cityscapesScripts.git
mv cityscapesScripts/cityscapesscripts ./
rm -rf cityscapesScripts
python cityscapesscripts/preparation/createTrainIdLabelImgs.py
```
