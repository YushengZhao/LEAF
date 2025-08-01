from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from cfg import args
from utils import *
import random

class TrafficDataset(Dataset):
    def __init__(self, signal, time_offset=0) -> None:
        super().__init__()
        self.signal = signal
        self.offset = time_offset
        
    def __len__(self):
        return len(self.signal) - args.seq_in_len - args.seq_out_len + 1
    
    def __getitem__(self, idx):
        data = self.get_segment(idx)
        return data
    
    def get_segment(self, idx, bias=0):
        data = {}
        x_data = self.signal[idx:idx+args.seq_in_len, :, :]
        x_signal = x_data[:, :, :args.in_dim]
        y_signal = self.signal[idx+args.seq_in_len:idx+args.seq_in_len+args.seq_out_len, :, :args.out_dim]

        x_signal[:, :, :args.out_dim] = x_signal[:, :, :args.out_dim] + np.arange(x_signal.shape[0])[:, np.newaxis, np.newaxis] * bias
        y_signal = y_signal + np.arange(x_signal.shape[0], x_signal.shape[0] + y_signal.shape[0])[:, np.newaxis, np.newaxis] * bias
        x_signal = x_signal.astype(np.float32)
        y_signal = y_signal.astype(np.float32)

        x_signal_norm = normalize_feat(x_signal)
        y_signal_norm = normalize_output(y_signal)
        data['time_index'] = idx + self.offset
        data['x'] = x_signal_norm  # shape: (seq_in_len, num_nodes, args.in_dim)
        data['time_of_day'] = x_data[:, :, args.in_dim]  # shape: (seq_in_len, num_nodes)
        data['day_of_week'] = x_data[:, :, args.in_dim+1]  # shape: (seq_in_len, num_nodes)
        data['y'] = y_signal_norm  # shape: (seq_out_len, num_nodes, args.out_dim)
        return data


def sample_segments(sequence_len, segment_len, n):
    if segment_len * n > sequence_len:
        raise ValueError("The total length of all segments exceeds the sequence length.")
    possible_starts = list(range(sequence_len - segment_len + 1))
    sampled_segments = []
    while len(sampled_segments) < n:
        start = random.choice(possible_starts)
        sampled_segments.append((start, start + segment_len))
        possible_starts = [pos for pos in possible_starts if pos >= start + segment_len or pos < start]
    return sampled_segments


class TrafficSampledDataset(Dataset):
    def __init__(self, signal, time_offset=0, num_segments=100) -> None:
        super().__init__()
        self.signal = signal
        self.offset = time_offset
        self.num_segments = num_segments
        self.segment_len = args.seq_in_len + args.seq_out_len
        self.full_dataset = TrafficDataset(signal, time_offset)
        if args.test_indices is not None:
            import json
            with open(args.test_indices, 'r') as f:
                starts = json.load(f)['indices']
            self.segments = [(start, start + self.segment_len) for start in starts]
        else:
            self.segments = sample_segments(len(signal), self.segment_len, num_segments)
        
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        data = self.full_dataset.get_segment(self.segments[idx][0])
        return data


def get_dataloader(train, val):
    # process signal
    if args.dataset in ('PEMS03', 'PEMS04', 'PEMS08'):
        original_signal = np.load(f'./data/{args.dataset}/{args.dataset}.npz')['data']  # shape: (num_samples, num_nodes, num_features)
        assert args.num_nodes == original_signal.shape[1]
    else:
        raise NotImplementedError
    
    timestamps = np.arange(original_signal.shape[0])
    time_of_day_stamps = timestamps % 288
    day_of_week_stamps = timestamps // 288 % 7

    time_of_day = np.repeat(time_of_day_stamps.reshape(-1, 1, 1), original_signal.shape[1], axis=1)
    day_of_week = np.repeat(day_of_week_stamps.reshape(-1, 1, 1), original_signal.shape[1], axis=1)
    signal = np.concatenate([original_signal, time_of_day, day_of_week], axis=-1)

    train_ends = int(train * signal.shape[0])
    val_ends = int((train + val) * signal.shape[0])
    train_signal = signal[:train_ends].astype(np.float32)
    val_signal = signal[train_ends:val_ends].astype(np.float32)
    test_signal = signal[val_ends:].astype(np.float32)

    args.feature_mean = train_signal[:, :, :args.in_dim].mean(axis=(0, 1), keepdims=True)  # shape: (1, 1, args.in_dim)
    args.feature_std = train_signal[:, :, :args.in_dim].std(axis=(0, 1), keepdims=True)
    args.output_mean = train_signal[:, :, :args.out_dim].mean(axis=(0, 1), keepdims=True)
    args.output_std = train_signal[:, :, :args.out_dim].std(axis=(0, 1), keepdims=True)

    # process adjacency matrix
    if args.dataset in ('PEMS03', 'PEMS04', 'PEMS08'):
        node_order_path = None
        idx2nodeid = {}
        if args.dataset == 'PEMS03':
            node_order_path = f'./data/{args.dataset}/{args.dataset}.txt'
            with open(node_order_path, 'r') as f:
                for i, line in enumerate(f):
                    idx2nodeid[int(line.strip())] = i
        else:
            idx2nodeid = {i: i for i in range(args.num_nodes)}
        distance_df = pd.read_csv(f'./data/{args.dataset}/{args.dataset}.csv')
        distance_info = distance_df.values
        adj = np.zeros((args.num_nodes, args.num_nodes), dtype=np.float32)
        adj_float = get_float_adj(distance_df, args.num_nodes, node_order_path)
        for i, j, d in distance_info:
            adj[idx2nodeid[int(i)], idx2nodeid[int(j)]] = 1
            adj[idx2nodeid[int(j)], idx2nodeid[int(i)]] = 1
        adj_with_self_loop = adj + np.eye(adj.shape[0])
        args.adj = adj_with_self_loop
    else:
        raise NotImplementedError

    train_set = TrafficDataset(train_signal, time_offset=0)
    val_set = TrafficDataset(val_signal, time_offset=train_ends)
    if args.sample_test:
        test_set = TrafficSampledDataset(test_signal, time_offset=val_ends, num_segments=args.num_test_segments)
    else:
        test_set = TrafficDataset(test_signal, time_offset=val_ends)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
        