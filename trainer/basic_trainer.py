from cfg import args
from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from utils import *


class BasicTrainer:
    def __init__(self, model, optimizer, scheduler, criterion, train_loader, val_loader, test_loader, num_epochs):
        self.model = model
        self.model_name = model.__class__.__name__
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.num_epochs = num_epochs

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.ckpt_path = os.path.join(args.save, args.expid, f'{self.model_name}.pth')

    def train(self):
        logger.info('Start training')
        self.model = self.model.to(args.device)
        best_val_mae = float('inf')
        best_val_rmse = float('inf')
        best_val_mape = float('inf')
        best_epoch = 0
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            losses = []
            for i, data in enumerate(self.train_loader):
                move_data(data, args.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, data['y'])
                loss.backward()
                if args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                self.optimizer.step()
                losses.append(loss.item())
            train_loss = sum(losses) / len(losses)
            val_mae, val_rmse, val_mape = self.validate(self.val_loader)
            test_mae, test_rmse, test_mape = self.validate(self.test_loader)
            self.scheduler.step(val_mae)
            logger.info(f'Epoch {epoch}, train loss: {train_loss:.4f}, val mae: {val_mae:3f}, val rmse: {val_rmse:.3f}, val mape: {val_mape:.3f}, test mae: {test_mae:.3f}, test rmse: {test_rmse:.3f}, test mape: {test_mape:.3f}')

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_val_rmse = val_rmse
                best_val_mape = val_mape
                best_epoch = epoch
                torch.save(self.model.state_dict(), self.ckpt_path)
        logger.info(f'Best epoch: {best_epoch}, val mae: {best_val_mae}, val rmse: {best_val_rmse}, val mape: {best_val_mape}')

    def validate(self, loader, verbose=False):
        outputs, targets = [], []
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(loader):
                move_data(data, args.device)
                output = self.model(data)
                output = denormalize_output(output)
                target = data['y']
                target = denormalize_output(target)
                outputs.append(output)
                targets.append(target)
        outputs = torch.cat(outputs, dim=0)
        targets = torch.cat(targets, dim=0)
        if verbose:
            return metric(outputs, targets, verbose=True)
        else:
            return metric(outputs, targets)

    def test(self):
        # load best model
        self.model.load_state_dict(torch.load(self.ckpt_path))
        test_mae, test_rmse, test_mape = self.validate(self.test_loader)
        return test_mae, test_rmse, test_mape

    def run(self):
        self.train()
        test_mae, test_rmse, test_mape = self.test()
        logger.info(f'Test mae: {test_mae}, test rmse: {test_rmse}, test mape: {test_mape}')


def build_basic_trainer(model, train_loader, val_loader, test_loader, num_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.HuberLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=False)
    return BasicTrainer(model, optimizer, scheduler, criterion, train_loader, val_loader, test_loader, num_epochs)