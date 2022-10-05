'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.blip_imagecode import blip_nlvr

import wandb
import utils
from utils import cosine_lr_schedule, warmup_lr_schedule
from data import create_dataset, create_sampler, create_loader

wandb.init(project='BLIP-imagecode', settings=wandb.Settings(start_method='fork'))

def train(model, data_loader, optimizer, epoch, device, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 10

    for i,(image0, image1, text, targets, is_video) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
  
        images = torch.cat([image0, image1], dim=0)
        images, targets = images.to(device), targets.to(device)   

        loss = model(images, text, targets=targets, train=True)    
        
        loss.backward()  
        if i%config['grad_accumulation'] == 0:
            optimizer.step()
            optimizer.zero_grad()
       
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.item())
        wandb.log({'Loss': loss.item()})
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    


@torch.no_grad()
def evaluate(model, data_loader, device, config):
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50

    for image0, image1, text, targets, is_video in metric_logger.log_every(data_loader, print_freq, header):
        images = torch.cat([image0, image1], dim=0)
        images, targets = images.to(device), targets.to(device)   
        
        prediction = model(images, text, targets=targets, train=False)  
 
        _, pred_class = prediction.max(1)
        accuracy = (targets==pred_class).sum() / targets.size(0)
        video_accuracy = ((pred_class.cuda() == targets.cuda()) * is_video.cuda()).sum() / is_video.sum()

        metric_logger.meters['acc'].update(accuracy.item(), n=image0.size(0))
        metric_logger.meters['video_acc'].update(accuracy.item(), n=image0.size(0))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print("Averaged stats:", metric_logger.global_avg())   
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

        
def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating dataset")
    datasets = create_dataset('imagecode', config) 
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True,False], num_tasks, global_rank)
    else:
        samplers = [None, None, None]
    
    batch_size=[config['batch_size_train'],config['batch_size_test'],config['batch_size_test']]
    train_loader, val_loader = create_loader(datasets,samplers,batch_size=batch_size,
                                                          num_workers=[4,4],is_trains=[True,False], 
                                                          collate_fns=[None,None])

    #### Model #### 
    print("Creating model")
    model = blip_nlvr(pretrained=config['pretrained'], image_size=config['image_size'], 
                         vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])

    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed: 
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
            
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    print("Start training")
    start_time = time.time()
    best = 0
    best_epoch = 0

    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            
            train_stats = train(model, train_loader, optimizer, epoch,  device, config) 
            
        val_stats = evaluate(model, val_loader, device, config)
        # test_stats = evaluate(model, test_loader, device, config)  
        wandb.log({'Val Accuracy': val_stats['acc']})
        if utils.is_main_process():  
            if args.evaluate:                
                log_stats = {**{f'val_{k}': v for k, v in val_stats.items()}
                            #  **{f'test_{k}': v for k, v in test_stats.items()},
                            }
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")   
                
            else:       
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_stats.items()},
                            #  **{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch,
                            }

                if float(val_stats['acc'])>best:
                    # save_obj = {
                    #     'model': model_without_ddp.state_dict(),
                    #     'optimizer': optimizer.state_dict(),
                    #     'config': config,
                    #     'epoch': epoch,
                    # }
                    # torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth')) 
                    best = float(val_stats['acc'])
                    wandb.log({'Best Val Accuracy': best})
                    best_epoch = epoch

                # with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    # f.write(json.dumps(log_stats) + "\n")
        if args.evaluate:             
            break            
         
        # dist.barrier()   
    
    # if utils.is_main_process():   
        # with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
        #     f.write("best epoch: %d"%best_epoch)      
            
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/imagecode.yaml')
    parser.add_argument('--output_dir', default='output/imagecode')
    parser.add_argument('--evaluate', action='store_true')      
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--batchsize', default=16, type=int)
    parser.add_argument('--grad_accumulation', default=1, type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--decay', type=float)
    parser.add_argument('--max_epochs', type=float)
    parser.add_argument('--video_only', type=str)
    parser.add_argument('--random_pair_sampling', type=str)
    parser.add_argument('--max_words', type=int)
    parser.add_argument('--aug_prob', type=float)
    parser.add_argument('--concat_layer1to6', type=str)
    parser.add_argument('--job_id', type=str)
    args = parser.parse_args()

    args.video_only = args.video_only == 'True'
    args.random_pair_sampling = args.random_pair_sampling == 'True'
    args.concat_layer1to6 = args.concat_layer1to6 == 'True'

    wandb.config.update(args)
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config['grad_accumulation'] = args.grad_accumulation
    config['init_lr'] = args.lr
    config['weight_decay'] = args.decay
    config['max_epochs'] = args.max_epochs
    config['video_only'] = args.video_only
    config['random_pair_sampling'] = args.random_pair_sampling
    config['max_words'] = args.max_words
    config['aug_prob'] = args.aug_prob
    config['concat_layer1to6'] = args.concat_layer1to6
    
    # Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    # yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)