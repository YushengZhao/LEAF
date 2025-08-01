from cfg import args
from trainer import build_trainer_from_cfg
from dataset import get_dataloader

def main():
    train_loader, val_loader, test_loader = get_dataloader(args.train_ratio, args.val_ratio)
    solver = build_trainer_from_cfg(train_loader, val_loader, test_loader)
    solver.run()

if __name__ == '__main__':
    main()