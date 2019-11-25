This repo is based on [Link](https://github.com/joe-siyuan-qiao/pytorch-classification).

## Training

ResNet-50:
```
python -W ignore imagenet.py -a l_resnet50 --data ~/dataset/ILSVRC2012/ --epochs 90 --schedule 30 60 --gamma 0.1 -c checkpoints/imagenet/resnet50 --gpu-id 0,1,2,3
```

ResNet-101:
```
python -W ignore imagenet.py -a l_resnet101 --data ~/dataset/ILSVRC2012/ --epochs 100 --schedule 30 60 90 --gamma 0.1 -c checkpoints/imagenet/resnet101 --gpu-id 0,1,2,3 --train-batch 128 --test-batch 128
```

## Pretrained Models
| Architecture | Pretrained |
|--------------|:----------:|
| ResNet-50    | [Link](https://cs.jhu.edu/~syqiao/BatchChannelNormalization/rn50_bcn_ws.pth)  |
| ResNet-101   | [Link](https://cs.jhu.edu/~syqiao/BatchChannelNormalization/rn101_bcn_ws.pth) |
| ResNeXt-50   | [Link](https://cs.jhu.edu/~syqiao/BatchChannelNormalization/rx50_bcn_ws.pth)  |
