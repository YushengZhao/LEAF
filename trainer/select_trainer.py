from cfg import args
from loguru import logger
import torch
import os
from utils import *
from .basic_trainer import build_basic_trainer
from adapter import build_adapter_from_cfg
from tqdm import tqdm

class SelectTrainer:
    def __init__(self, models, train_loader, val_loader, test_loader) -> None:
        self.models = models
        self.optimizers = [torch.optim.Adam(model.get_finetune_params(), lr=args.test_lr, weight_decay=args.test_wd) for model in models]
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.pretrain_trainers = [build_basic_trainer(model, train_loader, val_loader, test_loader, args.pretrain_epochs) for model in models]
        self.adapter = build_adapter_from_cfg()
        self.inter_paths = [os.path.join(args.dump_dir, f'{args.dataset.lower()}_{self.models[0].__class__.__name__}_{args.selector_type}{args.postfix}.pth'), os.path.join(args.dump_dir, f'{args.dataset.lower()}_{self.models[1].__class__.__name__}_{args.selector_type}{args.postfix}.pth')]

    def get_previous_paths(self):
        new_name = {
            'llm_r2': 'llm',
        }[args.selector_type]
        return [os.path.join(args.dump_dir, f'{args.dataset.lower()}_{self.models[0].__class__.__name__}_{new_name}{args.postfix}.pth'), os.path.join(args.dump_dir, f'{args.dataset.lower()}_{self.models[1].__class__.__name__}_{new_name}{args.postfix}.pth')]
    
    def pretrain(self):
        ckpt_paths = []
        if len(args.ckpt_paths) != len(self.models):
            logger.info('Checkpoint paths not used, pretraining from scratch')
            for trainer in self.pretrain_trainers:
                trainer.train()
                ckpt_paths.append(trainer.ckpt_path)
        else:
            ckpt_paths = args.ckpt_paths
        
        if args.selector_type.startswith('llm_r'):
            ckpt_paths = self.get_previous_paths()
        
        logger.info('Loading checkpoint paths')
        for i, path in enumerate(ckpt_paths):
            self.models[i].load_state_dict(torch.load(path))
            self.models[i].to(args.device)
            if args.verbose_metric:
                test_mae, test_rmse, test_mape, mae_list, rmse_list, mape_list = self.pretrain_trainers[i].validate(self.test_loader, verbose=True)
                logger.info(f'MAE list: {mae_list}')
                logger.info(f'RMSE list: {rmse_list}')
                logger.info(f'MAPE list: {mape_list}')
            else:
                test_mae, test_rmse, test_mape = self.pretrain_trainers[i].validate(self.test_loader)
            logger.info(f'Model {self.models[i].__class__.__name__}, test mae: {test_mae:.3f}, test rmse: {test_rmse:.3f}, test mape: {test_mape:.3f}')
        
    def run(self):
        self.pretrain()
        logger.info('Pretraining finished')

        outputs, targets = [], []
        pbar = tqdm(enumerate(self.test_loader), total=len(self.test_loader))
        interval = 1
        for iter, data in pbar:
            move_data(data, args.device)
            target = data['y']
            target = denormalize_output(target)
            result = self.adapter.run(data, self.models, self.optimizers)
            outputs.append(denormalize_output(result['output']))
            targets.append(target)

            if (iter + 1) % interval == 0:
                temp_outputs = torch.cat(outputs, dim=0)
                temp_targets = torch.cat(targets, dim=0)
                test_mae, test_rmse, test_mape = metric(temp_outputs, temp_targets)
                pbar.set_postfix({'MAE': test_mae, 'RMSE': test_rmse, 'MAPE': test_mape})
        outputs = torch.cat(outputs, dim=0)
        targets = torch.cat(targets, dim=0)
        if args.verbose_metric:
            test_mae, test_rmse, test_mape, mae_list, rmse_list, mape_list = metric(outputs, targets, verbose=True, time_dim=1)
            logger.info(f'Test mae: {test_mae:.3f}, test rmse: {test_rmse:.3f}, test mape: {test_mape:.3f}')
            logger.info(f'MAE list: {mae_list}')
            logger.info(f'RMSE list: {rmse_list}')
            logger.info(f'MAPE list: {mape_list}')
        else:
            test_mae, test_rmse, test_mape = metric(outputs, targets)
            logger.info(f'Test mae: {test_mae:.3f}, test rmse: {test_rmse:.3f}, test mape: {test_mape:.3f}')
        if args.selector_type.startswith('llm') and args.dump:
            torch.save(self.models[0].state_dict(), self.inter_paths[0])
            torch.save(self.models[1].state_dict(), self.inter_paths[1])
        