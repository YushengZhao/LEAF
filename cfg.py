import argparse
import random
import numpy as np
import torch
import os
import time
from datetime import datetime, timedelta

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='PEMS08', choices=['PEMS03', 'PEMS04', 'PEMS08'], help='dataset')
parser.add_argument('--seq_in_len', type=int, default=12, help='input sequence length')
parser.add_argument('--seq_out_len', type=int, default=12, help='output sequence length')
parser.add_argument('--method', type=str, default='basic', choices=['basic', 'select'])
parser.add_argument('--model_name', type=str, default='STGCN', help='model name')
parser.add_argument('--model_list', type=str, nargs='+', default=['GraphBranch', 'HypergraphBranch'], help='model list')
parser.add_argument('--adapter_type', type=str, default='basic')
parser.add_argument('--selector_type', type=str, default='optimal')
parser.add_argument('--update_selection', default=False, action='store_true', help='update selection')
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--stgcn_num_layers', type=int, default=7)
parser.add_argument('--hgnn_num_backbone_layers', type=int, default=6)
parser.add_argument('--hgnn_num_head_layers', type=int, default=1)
parser.add_argument('--hgnn_num_hyper_edge', type=int, default=32)
parser.add_argument('--hgnn_num_scales', type=int, default=6, choices=[1, 2, 6])
parser.add_argument('--train_ratio', type=float, default=0.1, help='train ratio')
parser.add_argument('--val_ratio', type=float, default=0.1, help='validation ratio')
parser.add_argument('--sample_test', default=False, action='store_true', help='sample test dataset')
parser.add_argument('--num_test_segments', type=int, default=100, help='number of test segments')
parser.add_argument('--test_indices', type=str, default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=int, default=0, help='')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--test_batch_size', type=int, default=32, help='test batch size')
parser.add_argument('--learning_rate', '--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', '--wd', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--test_lr', type=float, default=0.001, help='test learning rate')
parser.add_argument('--test_wd', type=float, default=0.000, help='test weight decay rate')
parser.add_argument('--clip', type=int, default=5, help='clip')

parser.add_argument('--pretrain_epochs', type=int, default=300)
parser.add_argument('--finetune_epochs', type=int, default=1)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--update_iters', type=int, default=1)

parser.add_argument('--print_every', type=int, default=1, help='')
parser.add_argument('--save', type=str, default='./ckpts/', help='save path')
parser.add_argument('--expid', type=str, default='debug', help='experiment id')
parser.add_argument('--csv', type=str, default='./outputs/', help='csv path')
parser.add_argument('--dump_dir', type=str, default='./outputs/dump', help='dump path')
parser.add_argument('--dump', default=False, action='store_true', help='dump results')

parser.add_argument('--ckpt_paths', type=str, nargs='+', default=[], help='pretrained model paths')
parser.add_argument('--disable_aug', default=False, action='store_true', help='disable augmentation')
parser.add_argument('--disable_graph', default=False, action='store_true', help='disable graph')
parser.add_argument('--disable_hypergraph', default=False, action='store_true', help='disable hypergraph')
parser.add_argument('--verbose_metric', default=False, action='store_true', help='verbose metric')

args = parser.parse_args()

args.device = torch.device('cuda:%d' % args.device)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(os.path.join(args.save, args.expid), exist_ok=True)
os.makedirs(os.path.join(args.csv, args.expid), exist_ok=True)

args.current_time_string = time.strftime('%m-%d-%H-%M-%S', time.localtime())
args.csv_path = os.path.join(args.csv, args.expid, f'{args.current_time_string}.csv')

if args.dataset == 'PEMS08':
    args.num_nodes = 170
    args.in_dim = 3
    args.out_dim = 1
    args.start_time = datetime(2016, 7, 1, 0, 0, 0)
    args.time_delta = timedelta(minutes=5)
elif args.dataset == 'PEMS04':
    args.num_nodes = 307
    args.in_dim = 3
    args.out_dim = 1
    args.start_time = datetime(2018, 1, 1, 0, 0, 0)
    args.time_delta = timedelta(minutes=5)
elif args.dataset == 'PEMS03':
    args.num_nodes = 358
    args.in_dim = 1
    args.out_dim = 1
    args.start_time = datetime(2018, 9, 1, 0, 0, 0)
    args.time_delta = timedelta(minutes=5)
else:
    raise ValueError('Unknown dataset')


if args.disable_aug:
    args.postfix = '_noaug'
elif args.disable_graph:
    args.postfix = '_nograph'
elif args.disable_hypergraph:
    args.postfix = '_nohypergraph'
elif args.update_iters not in [0, 5]:
    args.postfix = f'_update{args.update_iters}'
else:
    args.postfix = ''