from utils import *
from .selectors import *
from cfg import args
import torch.nn.functional as F
from .basic_adapter import triplet_ranking_loss


def broadcast_smooth_l1_loss(prediction, target, beta=1.0, reduction='mean'):
    diff = torch.abs(prediction - target)
    loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)

    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(f'Invalid value for reduction: {reduction}')


def multi_ranking_loss(output, positive_output, negative_output, margin=0):
    metric_mae = lambda x, y: torch.abs(x - y).mean(dim=(1, 3))
    metric_huber = lambda x, y: broadcast_smooth_l1_loss(x, y, reduction='none').mean(dim=(1, 3))
    dist_metric = metric_huber
    if isinstance(positive_output, list):
        positive = torch.cat(positive_output, dim=0)
    else:
        positive = positive_output
    if isinstance(negative_output, list):
        negative = torch.cat(negative_output, dim=0)
    else:
        negative = negative_output
    positive_dist = dist_metric(output, positive)  # num_pos x N
    negative_dist = dist_metric(output, negative)  # num_neg x N
    largest_positive_dist = positive_dist.max(dim=0)[0]  # N
    smallest_negative_dist = negative_dist.min(dim=0)[0]  # N
    loss = torch.relu(largest_positive_dist - smallest_negative_dist + margin).mean()
    return loss


def smooth_output(prediction, window_size=3):  # B x T x N x args.out_dim
    window = torch.ones(window_size) / window_size
    window = window.view(1, 1, -1, 1).to(prediction.device)

    batch_size, seq_len, num_nodes, out_dim = prediction.size()
    prediction = prediction.permute(0, 2, 1, 3)  # B x N x T x args.out_dim
    prediction = prediction.contiguous().view(batch_size * num_nodes, 1, seq_len, out_dim)  # (B x N) x 1 x T x args.out_dim
    prediction = F.pad(prediction, (0, 0, window_size // 2, window_size // 2), mode='replicate')
    smoothed_prediction = F.conv2d(prediction, window, padding=0)
    smoothed_prediction = smoothed_prediction.view(batch_size, num_nodes, seq_len, out_dim)
    smoothed_prediction = smoothed_prediction.permute(0, 2, 1, 3)  # B x T x N x args.out_dim
    return smoothed_prediction


class AugAdapter:
    def __init__(self, selector) -> None:
        self.selector = selector
        if args.disable_aug:
            self.aug_list = ['none']
        else:
            self.aug_list = ['none', 'smoothed_output', 'overestimated_output', 'underestimated_output', 'upward_trend_output', 'downward_trend_output']
        self.requires_grad_list = ['none']
        self.cached_result = None

    def get_aug_prediction(self, data, model, aug_type):
        if aug_type == 'none':
            output = model(data)
            prediction = denormalize_output(output)
        elif aug_type == 'smoothed_output':
            output = model(data)
            prediction = denormalize_output(output)
            prediction = smooth_output(prediction)
        elif aug_type == 'overestimated_output':
            output = model(data)
            prediction = denormalize_output(output)
            prediction = prediction * 1.05
        elif aug_type == 'underestimated_output':
            output = model(data)
            prediction = denormalize_output(output)
            prediction = prediction * 0.95
        elif aug_type == 'upward_trend_output':
            output = model(data)
            prediction = denormalize_output(output)
            prediction = prediction * (torch.arange(prediction.size(1)).view(1, -1, 1, 1).to(prediction.device) * 0.01 + 1)
        elif aug_type == 'downward_trend_output':
            output = model(data)
            prediction = denormalize_output(output)
            prediction = prediction * (torch.arange(prediction.size(1)).view(1, -1, 1, 1).to(prediction.device) * -0.01 + 1)
        else:
            raise ValueError(f'Invalid aug_type: {aug_type}')
        return output, prediction
    
    
    def select(self, data, models, iter):
        choices = []
        outputs = []
        for i, model in enumerate(models):
            if args.disable_graph and model.__class__.__name__ == 'GraphBranch':
                continue
            if args.disable_hypergraph and model.__class__.__name__ == 'HypergraphBranch':
                continue
            for aug_type in self.aug_list :
                with torch.set_grad_enabled(aug_type in self.requires_grad_list):
                    output, prediction = self.get_aug_prediction(data, model, aug_type)
                outputs.append(output)
                choice = {'model_name': model.__class__.__name__, 'output': prediction, 'aug_type': aug_type}  # B x T x N x args.out_dim
                choices.append(choice)

        if iter == 0 or self.cached_result is None or args.update_selection:
            result = self.selector.select(data, choices, iter)
            self.cached_result = result
        else:
            result = self.cached_result
        return outputs, choices, result
    
    def run(self, data, models, optimizers):
        assert data['y'].size(0) == 1
        name2idx = {model.__class__.__name__: i for i, model in enumerate(models)}
        for iter in range(args.update_iters):
            outputs, choices, result = self.select(data, models, iter)
            for output, choice in zip(outputs, choices):
                if choice['aug_type'] in self.requires_grad_list:
                    model_idx = name2idx[choice['model_name']]
                    optimizer = optimizers[model_idx]
                    optimizer.zero_grad()
                    loss = multi_ranking_loss(output, result['positive_output'], result['negative_output'])
                    loss.backward()
                    optimizer.step()

        outputs, choices, result = self.select(data, models, args.update_iters)

        return result

def build_aug_adapter():
    return AugAdapter(selector=OptimalSelector())