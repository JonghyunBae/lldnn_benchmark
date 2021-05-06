from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
#import matplotlib
#import matplotlib.pyplot as plt


try:
    from nvidia.dali.plugin.pytorch import LastBatchPolicy, DALIGenericIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
    from os.path import join
    from random import shuffle
    from collections import namedtuple
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

import torch.cuda.nvtx as nvtx

def create_dali_pipeline(batch_size, num_threads, device_id, data_dir):
    files = []
    with open(join(data_dir, "file_list.txt"), "r") as f:
        files = [line.rstrip() for line in f if line is not '']

    shuffle(files)

    img_files = []
    seg_files = []

    for prefix in files:
        img_files.append(join(data_dir, "leftImg8bit/train", prefix + "_leftImg8bit.png"))
        seg_files.append(join(data_dir, "gtFine/train", prefix + "_gtFine_labelIds.png"))

    pipeline = Pipeline(batch_size, num_threads, device_id, seed=12 + device_id)
    with pipeline:
        imgs, _ = fn.file_reader(files=img_files,
                                 shard_id=0, num_shards=1,
                                 random_shuffle=False, pad_last_batch=True)

        segs, _ = fn.file_reader(files=seg_files,
                                 shard_id=0, num_shards=1,
                                 random_shuffle=False, pad_last_batch=True)

        dali_device = 'gpu'
        decoder_device = 'mixed'
        # device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        # host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        device_memory_padding = 0
        host_memory_padding = 0

        imgs = fn.image_decoder(imgs, device=decoder_device, output_type=types.RGB,
                                device_memory_padding=device_memory_padding,
                                host_memory_padding=host_memory_padding,
                                hybrid_huffman_threshold=250000)

        segs = fn.image_decoder(segs, device=decoder_device, output_type=types.GRAY,
                                 device_memory_padding=device_memory_padding,
                                 host_memory_padding=host_memory_padding,
                                 hybrid_huffman_threshold=250000)

        imgs = fn.crop_mirror_normalize(imgs, device=dali_device,
                                        crop=(512, 512),
                                        dtype=types.FLOAT,
                                        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                                        output_layout="CHW")

        segs = fn.crop(segs, device=dali_device,
                       dtype=types.UINT8, crop=(512, 512))

        pipeline.set_outputs(imgs, segs)

    return pipeline


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)
    
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=1,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            #et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=opts.download, transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', download=False, transform=val_transform)

    if opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            #et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter( brightness=0.5, contrast=0.5, saturation=0.5 ),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            #et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)
    return train_dst, val_dst


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):
            
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

#            if opts.save_val_results:
#                for i in range(len(images)):
#                    image = images[i].detach().cpu().numpy()
#                    target = targets[i]
#                    pred = preds[i]
#
#                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
#                    target = loader.dataset.decode_target(target).astype(np.uint8)
#                    pred = loader.dataset.decode_target(pred).astype(np.uint8)
#
#                    Image.fromarray(image).save('results/%d_image.png' % img_id)
#                    Image.fromarray(target).save('results/%d_target.png' % img_id)
#                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)
#
#                    fig = plt.figure()
#                    plt.imshow(image)
#                    plt.axis('off')
#                    plt.imshow(pred, alpha=0.7)
#                    ax = plt.gca()
#                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
#                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
#                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
#                    plt.close()
#                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples


def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19

    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset=='voc' and not opts.crop_val:
        opts.val_batch_size = 1
    

    pipe = create_dali_pipeline(batch_size=opts.batch_size, num_threads=8,
                                device_id=0, data_dir="/home/ubuntu/cityscapes")
    pipe.build()
    train_loader = DALIGenericIterator(pipe, output_map=['image', 'label'], last_batch_policy=LastBatchPolicy.PARTIAL)

    # Set up model
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }

    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1*opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    #optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    #torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy=='poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy=='step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    #criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)
    
    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    #==========   Train Loop   ==========#
    interval_loss = 0

    class_conv = [255, 255, 255, 255, 255,
                  255, 255, 255, 0, 1,
                  255, 255, 2, 3, 4,
                  255, 255, 255, 5, 255,
                  6, 7, 8, 9, 10,
                  11, 12, 13, 14, 15,
                  255, 255, 16, 17, 18]

    while True: #cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        #model = model.half()
        cur_epochs += 1
        while True:
            train_iter = iter(train_loader)
            try:
                nvtx.range_push("Batch " + str(cur_itrs))

                nvtx.range_push("Data loading")
                data = next(train_iter)
                cur_itrs += 1

                images = data[0]['image'].to(dtype=torch.float32)
                labels = data[0]['label'][:, :, :, 0].to(dtype=torch.long)
                labels = torch.zeros(data[0]['label'][:, :, :, 0].shape).to(device, dtype=torch.long)
                nvtx.range_pop()

                nvtx.range_push("Forward pass")
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                nvtx.range_pop()

                nvtx.range_push("Backward pass")
                loss.backward()
                optimizer.step()
                nvtx.range_pop()

                np_loss = loss.detach().cpu().numpy()
                interval_loss += np_loss

                nvtx.range_pop()

                if cur_itrs == 10:
                    break

                if vis is not None:
                    vis.vis_scalar('Loss', cur_itrs, np_loss)

                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0

                scheduler.step()  

                if cur_itrs >=  opts.total_itrs:
                    return
            except StopIteration:
                break

        break
        
if __name__ == '__main__':
    main()
