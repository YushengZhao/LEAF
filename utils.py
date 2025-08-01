import torch
import numpy as np
from cfg import args


def move_data(data, device):
    for key in data.keys():
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].to(device)
    return data

def time_idx_to_datetime(time_idx, offset=0):
    time = args.start_time + args.time_delta * (time_idx + offset)
    return time

def RMSE(y_true, y_pred):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        rmse = np.square(np.abs(y_pred - y_true))
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        return rmse


def MAE(y_true, y_pred):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(y_pred - y_true)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        return mae


def MAPE(y_true, y_pred, null_val=0):
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype("float32")
        mask /= np.mean(mask)
        mape = np.abs(np.divide((y_pred - y_true).astype("float32"), y_true))
        mape = np.nan_to_num(mask * mape, nan=0)
        return np.mean(mape) * 100


def metric(y_pred, y_true, verbose=False, time_dim=1):
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    mae = MAE(y_true, y_pred)
    rmse = RMSE(y_true, y_pred)
    mape = MAPE(y_true, y_pred)
    if verbose:
        mae_list, rmse_list, mape_list = [], [], []
        for idx in range(y_pred.shape[time_dim]):
            y_pred_slice = np.take(y_pred, indices=idx, axis=time_dim)
            y_true_slice = np.take(y_true, indices=idx, axis=time_dim)
            mae_list.append(MAE(y_true_slice, y_pred_slice))
            rmse_list.append(RMSE(y_true_slice, y_pred_slice))
            mape_list.append(MAPE(y_true_slice, y_pred_slice))
        return mae, rmse, mape, mae_list, rmse_list, mape_list
    return mae, rmse, mape


def _normalize(x, mean_value, std_value):
    if isinstance(x, torch.Tensor):
        mean = torch.tensor(mean_value, device=x.device)
        std = torch.tensor(std_value, device=x.device)
        while len(mean.shape) < len(x.shape):
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return (x - mean) / std
    elif isinstance(x, np.ndarray):
        mean = mean_value
        std = std_value
        while len(mean.shape) < len(x.shape):
            mean = mean[np.newaxis, ...]
            std = std[np.newaxis, ...]
        return (x - mean) / std
    else:
        raise ValueError('x should be torch.Tensor or np.ndarray')


def normalize_feat(feat):
    return _normalize(feat, args.feature_mean, args.feature_std)

def normalize_output(output):
    return _normalize(output, args.output_mean, args.output_std)

def _denormalize(x, mean_value, std_value):
    if isinstance(x, torch.Tensor):
        mean = torch.tensor(mean_value, device=x.device)
        std = torch.tensor(std_value, device=x.device)
        while len(mean.shape) < len(x.shape):
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return x * std + mean
    elif isinstance(x, np.ndarray):
        mean = mean_value
        std = std_value
        while len(mean.shape) < len(x.shape):
            mean = mean[np.newaxis, ...]
            std = std[np.newaxis, ...]
        return x * std + mean
    else:
        raise ValueError('x should be torch.Tensor or np.ndarray')
    
def denormalize_feat(feat):
    return _denormalize(feat, args.feature_mean, args.feature_std)

def denormalize_output(output):
    return _denormalize(output, args.output_mean, args.output_std)

