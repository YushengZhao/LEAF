import torch
import torch.nn as nn
from cfg import args
from .basic_trainer import build_basic_trainer
from .select_trainer import SelectTrainer
from model import get_model_class


def build_trainer_from_cfg(train_loader, val_loader, test_loader):
    if args.method == 'basic':
        net = get_model_class(args.model_name)()
        solver = build_basic_trainer(net, train_loader, val_loader, test_loader, args.epochs)
    elif args.method == 'select':
        nets = [get_model_class(name)() for name in args.model_list]
        solver = SelectTrainer(nets, train_loader, val_loader, test_loader)
    else:
        raise ValueError('Method not supported')
    return solver