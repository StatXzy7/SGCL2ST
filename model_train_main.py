import os
import numpy as np
from tqdm import tqdm
import scipy.io as sio

import torch
from torch import nn
import torch.distributed as dist
import torch.utils.data.distributed
from utils import get_lr, AvgMeter

import config as CFG
from dataset import *
from models import *
from torch.utils.data import DataLoader

import scanpy as sc
import argparse

parser = argparse.ArgumentParser(description='DDP for CLIP')

parser.add_argument('--exp_name', type=str, default='clip', help='')
parser.add_argument('--batch_size', type=int, default=256, help='')
parser.add_argument('--max_epochs', type=int, default=200, help='')
parser.add_argument('--num_workers', type=int, default=4, help='')

parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
# parser.add_argument('--dist-backend', default='gloo', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')

parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')
parser.add_argument('--model', type=str, default='auto', help='')

# Hi
parser.add_argument('--name', type=str, default='hist2ST', help='prefix name.')
parser.add_argument('--data', type=str, default='cscc', help='dataset name:{"her2st","cscc"}.')
parser.add_argument('--logger', type=str, default='../logs/my_logs', help='logger path.')
parser.add_argument('--fold', type=int, default=5, help='dataset fold.')
parser.add_argument('--prune', type=str, default='Grid', help='how to prune the edge:{"Grid","NA"}')
parser.add_argument('--policy', type=str, default='mean', help='the aggregation way in the GNN .')
parser.add_argument('--neighbor', type=int, default=8, help='the number of neighbors in the GNN.') # Hist2ST = 4

def pk_load(fold,mode='train',flatten=False,dataset='her2st',r=4,ori=True,adj=True,prune='Grid',neighs=8): #r=4 Hist2ST, r=2 224
    assert dataset in ['her2st','cscc']
    if dataset=='her2st':
        dataset = CLIP_HER2ST(
            train=(mode=='train'),fold=fold,flatten=flatten,
            ori=ori,neighs=neighs,adj=adj,prune=prune,r=r
        )
    elif dataset=='cscc':
        dataset = CLIP_SKIN(
            train=(mode=='train'),fold=fold,flatten=flatten,
            ori=ori,neighs=neighs,adj=adj,prune=prune,r=r
        )
    return dataset

def build_loaders(args):
    trainset = pk_load(args.fold,'train',False,args.data,neighs=args.neighbor, prune=args.prune)
    train_loader = DataLoader(trainset, batch_size=1, num_workers=4, shuffle=True)
    testset = pk_load(args.fold,'test',False,args.data,neighs=args.neighbor, prune=args.prune)
    test_loader = DataLoader(testset, batch_size=1, num_workers=4, shuffle=False)
    return train_loader, test_loader

def cleanup():
    dist.destroy_process_group()

def train_epoch(model, train_loader, optimizer, args, lr_scheduler=None):
    loss_meter = AvgMeter()
    contrastive_loss_meter = AvgMeter()
    rmse_loss_meter = AvgMeter()
    zinb_loss_meter = AvgMeter()
    
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    
    for batch in tqdm_object:
        # Get data from batch (from your original code)
        ID, patch, center, exp, adj, oris, sfs, *_ = batch
        B,N,C,H,W = patch.shape
        patch = patch.reshape(B*N,C,H,W)  # (N,3,224,224)
        adj = adj.squeeze(0)
        exp = exp.squeeze(0)
        # loss = model(patch, center, exp, adj, oris, sfs)
        loss, contrastive_loss, rmse_loss, zinb_loss = model(patch, center, exp, adj, oris, sfs)
        
        optimizer.zero_grad()
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is None:
                print(f"{name} grad is None")
        
        for param in model.parameters():
            torch.distributed.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
            param.grad.data /= args.world_size

        optimizer.step()
        
        count = patch.size(0)
        loss_meter.update(loss.item(), count)
        contrastive_loss_meter.update(contrastive_loss.item(), count)
        rmse_loss_meter.update(rmse_loss.item(), count)
        zinb_loss_meter.update(zinb_loss.item(), count)

        # tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
        tqdm_object.set_postfix(train_loss=loss_meter.avg, contra_loss=contrastive_loss_meter.avg, 
                                rmse_loss=rmse_loss_meter.avg, zinb_loss=zinb_loss_meter.avg , lr=get_lr(optimizer))

    return loss_meter

# The train_epoch function now uses your original data extraction method and the loss computation from the new code.


def test_epoch(model, test_loader):
    loss_meter = AvgMeter()
    contrastive_loss_meter = AvgMeter()
    rmse_loss_meter = AvgMeter()
    zinb_loss_meter = AvgMeter()

    tqdm_object = tqdm(test_loader, total=len(test_loader))
    for batch in tqdm_object:
        # batch = {k: v.cuda() for k, v in batch.items() if k == "image" or k == "reduced_expression"}
        # loss = model(batch)
        ID, patch, center, exp, adj, oris, sfs, *_ = batch
        B,N,C,H,W = patch.shape
        patch = patch.reshape(B*N,C,H,W)  # (N,3,112,112)
        adj = adj.squeeze(0)
        exp = exp.squeeze(0)
        # loss = model(patch, center, exp, adj, oris, sfs)
        loss, contrastive_loss, rmse_loss, zinb_loss = model(patch, center, exp, adj, oris, sfs)

        count = patch.size(0)
        loss_meter.update(loss.item(), count)
        contrastive_loss_meter.update(contrastive_loss.item(), count)
        rmse_loss_meter.update(rmse_loss.item(), count)
        zinb_loss_meter.update(zinb_loss.item(), count)

        # tqdm_object.set_postfix(valid_loss=loss_meter.avg)
        tqdm_object.set_postfix(valid_loss=loss_meter.avg, contra_loss=contrastive_loss_meter.avg, 
                                rmse_loss=rmse_loss_meter.avg, zinb_loss=zinb_loss_meter.avg)
    return loss_meter


def main():
    print("Starting...")
    args = parser.parse_args()
    ngpus_per_node = torch.cuda.device_count()
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    rank = int(os.environ.get("SLURM_LOCALID", 0))*ngpus_per_node + local_rank
    
    current_device = local_rank
    torch.cuda.set_device(current_device)

    """ this block initializes a process group and initiate communications
		between all processes running on all nodes """
    
    print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
    #init the process group
    # dist.init_process_group(backend=args.dist_backend, init_method=args.init_method, world_size=args.world_size, rank=rank)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.init_method, world_size=args.world_size, rank=rank)

    print("process group ready!")


    #make the model
    print('From Rank: {}, ==> Making model..'.format(rank))
    if args.model == "resnet50_baseline":
        model = CLIPModel_resnet50().cuda(current_device)
        print("Image encoder is resnet50_baseline")
    elif args.model == "auto":
        model = myModel().cuda(current_device)
        print("Image encoder is ResNet50, Expression encoder is autoencoder")
    model = nn.parallel.DistributedDataParallel(model, device_ids=[current_device])

    #load the data
    print('From Rank: {}, ==> Preparing data..'.format(rank))
    train_loader, test_loader = build_loaders(args)
    
    # Initialize optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=CFG.step, gamma=CFG.factor)

    
    
    # Train the model for a fixed number of epochs
    best_loss = float('inf')
    best_epoch = 0
    for epoch in range(args.max_epochs):
        print(f"Epoch: {epoch + 1}")
        # step = "epoch"

        # Train the model
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, args)


        # Evaluate the model
        model.eval()
        with torch.no_grad():
            test_loss = test_epoch(model, test_loader)
        
        # Update learning rate
        # lr_scheduler.step()
        lr_scheduler.step(test_loss.avg)
        
        if test_loss.avg < best_loss and rank == 0:
            if not os.path.exists(str(args.exp_name)):
                os.mkdir(str(args.exp_name))
            best_loss = test_loss.avg
            best_epoch = epoch

            torch.save(model.state_dict(), str(args.exp_name) + "/best.pt")
            print("Saved Best Model! Loss: {}".format(best_loss))

    print("Done!, final loss: {}".format(best_loss))
    print("Best epoch: {}".format(best_epoch))
    cleanup()

if __name__ == "__main__":
    main()

