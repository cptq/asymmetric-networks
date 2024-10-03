"""
Fast training script for CIFAR-10 using FFCV.
For tutorial, see https://docs.ffcv.io/ffcv_examples/cifar10.html.
"""
import pickle
import os
import copy
from argparse import ArgumentParser
from typing import List
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss, Conv2d, BatchNorm2d
from torch.optim import SGD, lr_scheduler, AdamW
import torchvision

from fastargs import get_current_config, Param, Section
from fastargs.decorators import param
from fastargs.validation import And, OneOf

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

from models import resnet8, sparse_resnet8, sigma_asym_resnet8
from interp import interpolate_test

Section('training', 'Hyperparameters').params(
    lr=Param(float, 'The learning rate to use', required=True),
    epochs=Param(int, 'Number of epochs to run for', required=True),
    lr_peak_epoch=Param(int, 'Peak epoch for cyclic lr', required=True),
    batch_size=Param(int, 'Batch size', default=512),
    momentum=Param(float, 'Momentum for SGD', default=0.9),
    weight_decay=Param(float, 'l2 weight decay', default=5e-4),
    label_smoothing=Param(float, 'Value of label smoothing', default=0.1),
    num_workers=Param(int, 'The number of workers', default=8),
    lr_tta=Param(bool, 'Test time augmentation by averaging with horizontally flipped version', default=True),
    optimizer_type=Param(str, 'Type of optimizer', default='sgd')
)

Section('data', 'data related stuff').params(
    train_dataset=Param(str, '.dat file to use for training', required=True),
    val_dataset=Param(str, '.dat file to use for validation', required=True),
    gpu=Param(int, 'gpu num', required=True),
)

Section('model', 'model related stuff').params(
    model_type=Param(str, 'model type'),
    w_multiplier=Param(float, 'width multiplier'),
    mask_constant=Param(float, 'mask constant for sparse net'),
    mask_num_fixed=Param(int, 'num fixed entries for sparse net'),
    saved_mask=Param(bool, 'whether to use saved masks'),
)

Section('misc', 'misc').params(
    interp=Param(bool, 'whether to run interpolation test'),
    mli_interp=Param(bool, 'whether to run mli test'),
    root=Param(str, 'root for saving trained networks'),
    save_path=Param(str, 'path to save trained network'),
    save_name=Param(str, 'filename for trained network'),
)

@param('data.train_dataset')
@param('data.val_dataset')
@param('training.batch_size')
@param('training.num_workers')
@param('data.gpu')
def make_dataloaders(train_dataset=None, val_dataset=None, batch_size=None, num_workers=None, gpu=None):
    paths = {
        'train': train_dataset,
        'test': val_dataset

    }

    start_time = time.time()
    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]
    loaders = {}

    for name in ['train', 'test']:
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice(torch.device(f'cuda:{gpu}')), Squeeze()]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
        if name == 'train':
            image_pipeline.extend([
                RandomHorizontalFlip(),
                RandomTranslate(padding=2, fill=tuple(map(int, CIFAR_MEAN))),
                Cutout(4, tuple(map(int, CIFAR_MEAN))),
            ])
        image_pipeline.extend([
            ToTensor(),
            ToDevice(torch.device(f'cuda:{gpu}'), non_blocking=True),
            ToTorchImage(),
            Convert(torch.float16),
            torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        
        ordering = OrderOption.RANDOM if name == 'train' else OrderOption.SEQUENTIAL

        loaders[name] = Loader(paths[name], batch_size=batch_size, num_workers=num_workers,
                               order=ordering, drop_last=(name == 'train'),
                               pipelines={'image': image_pipeline, 'label': label_pipeline})

    return loaders, start_time

class ReshapeFlatten(torch.nn.Module):
    def forward(self, x): return x.reshape(x.size(0), -1)

@param('model.model_type')
@param('model.w_multiplier')
@param('model.mask_constant')
@param('model.mask_num_fixed')
@param('model.saved_mask')
@param('data.gpu')
def construct_model(model_type=None, w_multiplier=None, gpu=None, mask_constant=None, mask_num_fixed=None, saved_mask=False):
    num_class = 10
    if model_type == 'resnet8':
        model = resnet8(w=w_multiplier)
    elif model_type == 'smaller_resnet8':
        model = resnet8(w=.875*w_multiplier)
    elif model_type == 'sigma_asym_resnet8':
        model = sigma_asym_resnet8(w=w_multiplier)
    elif model_type == 'sparse_resnet8':
        if not saved_mask:
            mask_type = 'filter_random_subsets'
            #mask_type = 'random_subsets'
            linear_mask_params = {'mask_constant' : mask_constant, 'mask_type': 'random_subsets', 'do_normal_mask' : True, 'num_fixed': mask_num_fixed}
            conv_mask_params_f = {'mask_constant' : mask_constant, 'mask_type': mask_type, 'do_normal_mask' : True, 'num_fixed': mask_num_fixed}
            
            conv_mask_params_1_conv = {'mask_constant': mask_constant, 'mask_type' : mask_type, 'do_normal_mask' : True, 'num_fixed': mask_num_fixed}
            conv_mask_params_1_skip = {'mask_constant': mask_constant, 'mask_type' : mask_type, 'do_normal_mask' : True, 'num_fixed': mask_num_fixed}
            conv_mask_params_1 = {'conv1' : conv_mask_params_1_conv, 'conv2' : conv_mask_params_1_conv, 'skip':conv_mask_params_1_skip}
            
            conv_mask_params_2_conv = {'mask_constant': mask_constant, 'mask_type' : mask_type, 'do_normal_mask' : True, 'num_fixed': mask_num_fixed}
            conv_mask_params_2_skip = {'mask_constant': mask_constant, 'mask_type' : mask_type, 'do_normal_mask' : True, 'num_fixed': mask_num_fixed}
            conv_mask_params_2 = {'conv1': conv_mask_params_2_conv, 'conv2': conv_mask_params_2_conv, 'skip':conv_mask_params_2_skip}
            
            conv_mask_params_3_conv = {'mask_constant': mask_constant, 'mask_type' : mask_type, 'do_normal_mask' : True, 'num_fixed': mask_num_fixed}
            conv_mask_params_3_skip = {'mask_constant': mask_constant, 'mask_type' : mask_type, 'do_normal_mask' : True, 'num_fixed': mask_num_fixed}
            conv_mask_params_3 = {'conv1': conv_mask_params_3_conv, 'conv2': conv_mask_params_3_conv, 'skip': conv_mask_params_3_skip}
            mask_params = {
                   'linear' : linear_mask_params, 
                   'conv_f' : conv_mask_params_f, 
                   'conv_1': conv_mask_params_1,
                   'conv_2': conv_mask_params_2,
                   'conv_3': conv_mask_params_3
                  }
            model = sparse_resnet8(mask_params, w=w_multiplier)
        else:
            mask_params_saved = {
               'linear' : {"mask_path": "masks/sparse_resnet8_linear.pt", "mask_constant": mask_constant}, 
               'conv_f' : {"mask_path": "masks/sparse_resnet8_conv_f.pt", "mask_constant": mask_constant},
               'conv_1': {"conv1": {"mask_path": "masks/sparse_resnet8_block1_conv1.pt", "mask_constant": mask_constant}, "conv2": {"mask_path": "masks/sparse_resnet8_block1_conv2.pt", "mask_constant": mask_constant}, "skip": {"mask_path": "masks/sparse_resnet8_block1_skip.pt", "mask_constant": mask_constant}},
               'conv_2': {"conv1": {"mask_path": "masks/sparse_resnet8_block2_conv1.pt", "mask_constant": mask_constant}, "conv2": {"mask_path": "masks/sparse_resnet8_block2_conv2.pt", "mask_constant": mask_constant}, "skip": {"mask_path": "masks/sparse_resnet8_block2_skip.pt", "mask_constant": mask_constant}},
               'conv_3': {"conv1": {"mask_path": "masks/sparse_resnet8_block3_conv1.pt", "mask_constant": mask_constant}, "conv2": {"mask_path": "masks/sparse_resnet8_block3_conv2.pt", "mask_constant": mask_constant}, "skip": {"mask_path": "masks/sparse_resnet8_block3_skip.pt", "mask_constant": mask_constant}},
              }
            print("Using saved masks")
            model = sparse_resnet8(mask_params_saved, w=w_multiplier)
    elif model_type == 'mlp':
        in_dim = 32 * 32 * 3
        hidden_dim = 128
        model = nn.Sequential(ReshapeFlatten(),
                              nn.Linear(in_dim, hidden_dim),
                              nn.LayerNorm(hidden_dim),
                              nn.GELU(),
                              nn.Linear(hidden_dim, hidden_dim),
                              nn.LayerNorm(hidden_dim),
                              nn.GELU(),
                              nn.Linear(hidden_dim, 10))
                              
    else:
        raise ValueError('Invalid model')
    model = model.to(memory_format=torch.channels_last).to(f'cuda:{gpu}')
    print("Num params:", sum(p.numel() for p in model.parameters()))
    return model

@param('training.lr')
@param('training.epochs')
@param('training.momentum')
@param('training.weight_decay')
@param('training.label_smoothing')
@param('training.lr_peak_epoch')
@param('training.optimizer_type')
def train(model, loaders, lr=None, epochs=None, label_smoothing=None,
          momentum=None, weight_decay=None, lr_peak_epoch=None, optimizer_type=None):
    if optimizer_type == 'sgd':
        opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError('Invalid optimizer type')
    iters_per_epoch = len(loaders['train'])
    # Cyclic LR with single triangle
    lr_schedule = np.interp(np.arange((epochs+1) * iters_per_epoch),
                            [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
                            [0, 1, 0])
    scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    scaler = GradScaler()
    loss_fn = CrossEntropyLoss(label_smoothing=label_smoothing)

    for _ in range(epochs):
        for ims, labs in tqdm(loaders['train']):
            opt.zero_grad(set_to_none=True)
            with autocast():
                out = model(ims)
                loss = loss_fn(out, labs)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

@param('training.lr_tta')
@param('training.label_smoothing')
def evaluate(model, loaders, lr_tta=False, verbose=True, label_smoothing=None):
    model.eval()
    loss_fn = CrossEntropyLoss(label_smoothing=label_smoothing)
    metrics = {}

    with torch.no_grad():
        for name in ['train', 'test']:
            total_correct, total_num = 0., 0.
            total_loss = 0.
            for ims, labs in tqdm(loaders[name], disable=not verbose):
                with autocast():
                    out = model(ims)
                    if lr_tta:
                        out += model(ims.flip(-1))
                    batch_loss = loss_fn(out, labs)
                    total_loss += (batch_loss * ims.shape[0]).item()

                    total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                    total_num += ims.shape[0]
            acc = total_correct / total_num * 100
            loss = total_loss / total_num
            if verbose:
                print(f'{name} accuracy: {acc:.1f}%')
                print(f'{name} loss: {loss:.6f}')
            metrics[name + '_acc'] = acc
            metrics[name + '_loss'] = loss
    return metrics # returns test accuracy

if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser(description='Fast CIFAR-10 training')
    config.augment_argparse(parser)
    # Also loads from args.config_path if provided
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()

    loaders, start_time = make_dataloaders()
    model = construct_model()
    print(model)
    init_model = copy.deepcopy(model)
    train(model, loaders)
    print(f'Total time: {time.time() - start_time:.5f}')
    test_acc = evaluate(model, loaders)['test_acc']

    interp = config['misc.interp']
    if interp:
        model2 = construct_model()
        train(model2, loaders)
        evaluate(model2, loaders)
        print('Interpolating ...')
        results = interpolate_test(model, model2, lambda m: evaluate(m, loaders, verbose=False, label_smoothing=config['training.label_smoothing'])['test_acc'])
        print('Results:', results)
        plt.plot(results)
        plt.savefig('interp_plot.png')

    if config['misc.mli_interp']:
        print('Interpolating ...')
        results = interpolate_test(init_model, model, lambda m: evaluate(m, loaders, verbose=False, label_smoothing=config['training.label_smoothing'])['train_loss'])
        print('Results:', results)

        def mli_stats(results):
            """
                results is a 1d array of some train/test metric across interpolation values
            """
            results = np.array(results)
            diffs = results[1:] - results[:-1]
            max_diff = np.max(diffs)
            
            # convexity measures
            h = 1 / (results.shape[0]-1)
            second_derivatives = []
            for i in range(1, results.shape[0]-1):
                finite_diff = results[i+1] - 2*results[i] + results[i-1]
                finite_diff = finite_diff / h**2
                second_derivatives.append(finite_diff)
            second_derivatives = np.array(second_derivatives)
            convexity1 = np.min(second_derivatives)
            convexity2 = np.mean(second_derivatives >= 0)
            # proportion of points lying below the line segment between 0 and 1
            xs = np.linspace(0, 1, results.shape[0])
            line = results[0] + (results[1] - results[0]) * xs
            below_line = results <= line
            convexity3 = np.mean(below_line[1:-1])
            
            return max_diff, convexity1, convexity2, convexity3
        max_diff, convexity1, convexity2, convexity3 = mli_stats(results)
        is_monotonic = max_diff <= 0
        print("Max diff:", max_diff)
        print("Is monotonic:", is_monotonic)
        print("Convexity 1:", convexity1)
        print("Convexity 2:", convexity2)
        print("Convexity 3:", convexity3)
        mli_save_file = f'results_mli/{config["model.model_type"]}/{config["misc.save_name"]}.pkl'
        print(f"Saving MLI results to {mli_save_file}")
        with open(mli_save_file, 'wb') as f:
            pickle.dump(results, f)

        plt.plot(results)
        plt.savefig('mli_interp_plot.png')

    root = config['misc.root']
    save_path = root + config['misc.save_path']
    save_name = config['misc.save_name']
    if not os.path.exists(save_path):
        print('Making directory:', save_path)
        os.makedirs(save_path)
    filename = save_path + save_name + '.pt'
    print('Saving to', filename)
    torch.save((model.cpu().state_dict(), test_acc), filename)

