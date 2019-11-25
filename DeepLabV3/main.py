import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
from PIL import Image
from scipy.io import loadmat
from torch.autograd import Variable
from torchvision import transforms

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import deeplab
from pascal import VOCSegmentation
from cityscapes import Cityscapes
from utils import AverageMeter, inter_and_union

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', default=False,
                    help='training mode')
parser.add_argument('--exp', type=str, required=True,
                    help='name of experiment')
parser.add_argument('--gpu', type=int, default=0,
                    help='test time gpu device id')
parser.add_argument('--backbone', type=str, default='resnet101',
                    help='resnet101')
parser.add_argument('--dataset', type=str, default='pascal',
                    help='pascal or cityscapes')
parser.add_argument('--groups', type=int, default=None,
                    help='num of groups for group normalization')
parser.add_argument('--epochs', type=int, default=30,
                    help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size')
parser.add_argument('--base_lr', type=float, default=0.00025,
                    help='base learning rate')
parser.add_argument('--last_mult', type=float, default=1.0,
                    help='learning rate multiplier for last layers')
parser.add_argument('--scratch', action='store_true', default=False,
                    help='train from scratch')
parser.add_argument('--freeze_bn', action='store_true', default=False,
                    help='freeze batch normalization parameters')
parser.add_argument('--weight_std', action='store_true', default=False,
                    help='weight standardization')
parser.add_argument('--beta', action='store_true', default=False,
                    help='resnet101 beta')
parser.add_argument('--crop_size', type=int, default=513,
                    help='image crop size')
parser.add_argument('--resume', type=str, default=None,
                    help='path to checkpoint to resume from')
parser.add_argument('--workers', type=int, default=4,
                    help='number of data loading workers')


def start():
  args = parser.parse_args()
  if not args.train:
    main(0, args)
  else:
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node
    mp.spawn(main, nprocs=ngpus_per_node, args=(args, ngpus_per_node))


def main(gpu, args, ngpus_per_node=None):
  assert torch.cuda.is_available()
  torch.backends.cudnn.benchmark = True
  model_fname = 'data/deeplab_{0}_{1}_v3_{2}_epoch%d.pth'.format(
      args.backbone, args.dataset, args.exp)
  if args.dataset == 'pascal':
    dataset = VOCSegmentation('data/VOCdevkit',
        train=args.train, crop_size=args.crop_size)
  elif args.dataset == 'cityscapes':
    dataset = Cityscapes('data/cityscapes',
        train=args.train, crop_size=args.crop_size)
  else:
    raise ValueError('Unknown dataset: {}'.format(args.dataset))
  if args.backbone == 'resnet101':
    model = getattr(deeplab, 'resnet101')(
        pretrained=(not args.scratch),
        num_classes=len(dataset.CLASSES),
        num_groups=args.groups,
        weight_std=args.weight_std,
        beta=args.beta)
  else:
    raise ValueError('Unknown backbone: {}'.format(args.backbone))

  if args.train:
    args.gpu = gpu
    args.rank = 0
    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group('nccl', init_method='tcp://127.0.0.1:12345',
                world_size=args.world_size, rank=args.rank)

    criterion = nn.CrossEntropyLoss(ignore_index=255).cuda(args.gpu)
    # model = nn.DataParallel(model).cuda()
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    args.batch_size = int(args.batch_size / ngpus_per_node)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model.train()
    if args.freeze_bn:
      for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
          m.eval()
          m.weight.requires_grad = False
          m.bias.requires_grad = False
    backbone_params = (
        list(model.module.conv1.parameters()) +
        list(model.module.bn1.parameters()) +
        list(model.module.layer1.parameters()) +
        list(model.module.layer2.parameters()) +
        list(model.module.layer3.parameters()) +
        list(model.module.layer4.parameters()))
    last_params = list(model.module.aspp.parameters())
    optimizer = optim.SGD([
      {'params': filter(lambda p: p.requires_grad, backbone_params)},
      {'params': filter(lambda p: p.requires_grad, last_params)}],
      lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        pin_memory=True, num_workers=args.workers, sampler=train_sampler)
    max_iter = args.epochs * len(dataset_loader)
    losses = AverageMeter()
    start_epoch = 0

    if args.resume:
      if os.path.isfile(args.resume):
        print('=> loading checkpoint {0}'.format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('=> loaded checkpoint {0} (epoch {1})'.format(
          args.resume, checkpoint['epoch']))
      else:
        print('=> no checkpoint found at {0}'.format(args.resume))

    for epoch in range(start_epoch, args.epochs):
      train_sampler.set_epoch(epoch)
      for i, (inputs, target) in enumerate(dataset_loader):
        cur_iter = epoch * len(dataset_loader) + i
        lr = args.base_lr * (1 - float(cur_iter) / max_iter) ** 0.9
        optimizer.param_groups[0]['lr'] = lr
        optimizer.param_groups[1]['lr'] = lr * args.last_mult

        inputs = Variable(inputs.cuda(args.gpu))
        target = Variable(target.cuda(args.gpu))
        outputs = model(inputs)
        loss = criterion(outputs, target)
        if np.isnan(loss.item()) or np.isinf(loss.item()):
          pdb.set_trace()
        losses.update(loss.item(), args.batch_size)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if args.rank % ngpus_per_node == 0:
          print('epoch: {0}\t'
                'iter: {1}/{2}\t'
                'lr: {3:.6f}\t'
                'loss: {loss.val:.4f} ({loss.ema:.4f})'.format(
                epoch + 1, i + 1, len(dataset_loader), lr, loss=losses))

      if epoch % 10 == 9 and args.rank % ngpus_per_node == 0:
        torch.save({
          'epoch': epoch + 1,
          'state_dict': model.state_dict(),
          'optimizer': optimizer.state_dict(),
          }, model_fname % (epoch + 1))

  else:
    torch.cuda.set_device(args.gpu)
    model = model.cuda()
    model.eval()
    checkpoint = torch.load(model_fname % args.epochs)
    state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
    model.load_state_dict(state_dict)
    cmap = loadmat('data/pascal_seg_colormap.mat')['colormap']
    cmap = (cmap * 255).astype(np.uint8).flatten().tolist()

    inter_meter = AverageMeter()
    union_meter = AverageMeter()
    for i in range(len(dataset)):
      inputs, target = dataset[i]
      inputs = Variable(inputs.cuda())
      outputs = model(inputs.unsqueeze(0))
      _, pred = torch.max(outputs, 1)
      pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)
      mask = target.numpy().astype(np.uint8)
      imname = dataset.masks[i].split('/')[-1]
      mask_pred = Image.fromarray(pred)
      mask_pred.putpalette(cmap)
      mask_pred.save(os.path.join('data/val', imname))
      print('eval: {0}/{1}'.format(i + 1, len(dataset)))

      inter, union = inter_and_union(pred, mask, len(dataset.CLASSES))
      inter_meter.update(inter)
      union_meter.update(union)

    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    for i, val in enumerate(iou):
      print('IoU {0}: {1:.2f}'.format(dataset.CLASSES[i], val * 100))
    print('Mean IoU: {0:.2f}'.format(iou.mean() * 100))


if __name__ == "__main__":
  start()
