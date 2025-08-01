from utils import *
import json


def get_time_info(time_idx, time_format_string = "%B %d, %Y, %H:%M, %A"):
    input_start_time = time_idx_to_datetime(time_idx).strftime(time_format_string)
    input_end_time = time_idx_to_datetime(time_idx, offset=args.seq_in_len-1).strftime(time_format_string)
    output_start_time = time_idx_to_datetime(time_idx, offset=args.seq_in_len).strftime(time_format_string)
    output_end_time = time_idx_to_datetime(time_idx, offset=args.seq_in_len+args.seq_out_len-1).strftime(time_format_string)
    return input_start_time, input_end_time, output_start_time, output_end_time


def get_dump_entry(time_idx, iter, history, target, choices, node_idx, candidates):
    assert candidates.size(-1) == 1
    ground_truth = target[0, :, node_idx, :]
    mae_errors = torch.abs(candidates - ground_truth).mean(dim=(1, 2))
    optimal_idx = mae_errors.argmin()

    # prepare dump entry
    input_start_time, input_end_time, output_start_time, output_end_time = get_time_info(time_idx)
    hist = history[0, :, node_idx, 0].detach().cpu().int().tolist()
    gt = target[0, :, node_idx, 0].detach().cpu().int().tolist()
    cand = []
    model_names = [choice['model_name'] for choice in choices]
    aug_types = [choice['aug_type'] for choice in choices] if 'aug_type' in choices[0] else ['none'] * len(choices)
    for cand_idx in range(len(candidates)):
        cand.append(candidates[cand_idx, :, :].squeeze(-1).detach().cpu().int().tolist())
    dump_entry = {
        'time_idx': time_idx,
        'iter': iter,
        'input_start_time': input_start_time,#
        'input_end_time': input_end_time,#
        'output_start_time': output_start_time,#
        'output_end_time': output_end_time,#
        'node_idx': node_idx,#
        'choices': cand,#
        'model_names': model_names,#
        'aug_types': aug_types,#
        'history': hist,#
        'ground_truth': gt,
        'errors': mae_errors.detach().cpu().tolist(),
        'optimal_idx': optimal_idx.item(),
    }
    return dump_entry


class OptimalSelector:
    def __init__(self, dump_path) -> None:
        self.counter = 0
        self.dump_data = []
        self.dump_path = dump_path
    
    def select(self, data, choices, iter=None):
        assert data['y'].size(0) == 1
        target = data['y']
        history = data['x'][:, :, :, :args.out_dim]
        time_idx = data['time_index'].item()

        target = denormalize_output(target)
        history = denormalize_output(history)
        predictions = [choice['output'] for choice in choices]  # shape: B, T, N, args.out_dim

        # output templates
        optimal_prediction = torch.zeros_like(predictions[0])
        worst_prediction = torch.zeros_like(predictions[0])
        sub_optimal_predictions = [torch.zeros_like(predictions[0]) for _ in range(len(predictions) - 1)]
        
        num_nodes = optimal_prediction.size(2)
        for i in range(num_nodes):
            ground_truth = target[0, :, i, :]  # shape: T, args.out_dim

            candidates = [prediction[0, :, i, :] for prediction in predictions]
            candidates = torch.stack(candidates, dim=0)  # shape: num_models, T, args.out_dim

            mae_errors = torch.abs(candidates - ground_truth).mean(dim=(1, 2))  # shape: num_models

            optimal_idx = mae_errors.argmin()
            optimal_prediction[0, :, i, :] = candidates[optimal_idx]

            worst_idx = mae_errors.argmax()
            worst_prediction[0, :, i, :] = candidates[worst_idx]

            sub_optimal_indices = [idx for idx in range(len(predictions)) if idx != optimal_idx]
            for j, idx in enumerate(sub_optimal_indices):
                sub_optimal_predictions[j][0, :, i, :] = candidates[idx]
            
            self.counter += 1

            if args.dump:
                dump_entry = get_dump_entry(time_idx, iter, history, target, choices, i, candidates)
                self.dump_data.append(dump_entry)
        
        optimal_prediction = optimal_prediction.detach()
        worst_prediction = worst_prediction.detach()
        sub_optimal_predictions = [sub_optimal_prediction.detach() for sub_optimal_prediction in sub_optimal_predictions]
        
        optimal_prediction = normalize_output(optimal_prediction)
        worst_prediction = normalize_output(worst_prediction)
        sub_optimal_predictions = [normalize_output(sub_optimal_prediction) for sub_optimal_prediction in sub_optimal_predictions]

        result = {'output': optimal_prediction, 'positive_output': optimal_prediction, 'most_negative_output': worst_prediction, 'negative_output': sub_optimal_predictions}
        return result
    
    def __del__(self):
        if args.dump:
            import json
            with open(self.dump_path, 'w') as f:
                json.dump(self.dump_data, f)


class LLMSelectorFromJson:
    def __init__(self, json_path, prev_dump_path, dump_path):
        self.json_path = json_path
        self.prev_dump_path = prev_dump_path
        self.dump_path = dump_path
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)
        with open(self.prev_dump_path, 'r') as f:
            self.select_data = json.load(f)
        self.counter = 0
        assert len(self.data) == len(self.select_data)
        self.dump_data = []

    def select(self, data, choices, iter=None):
        assert data['y'].size(0) == 1
        assert args.out_dim == 1
        target = data['y']
        history = data['x'][:, :, :, :args.out_dim]
        time_idx = data['time_index'].item()
        target = denormalize_output(target)
        history = denormalize_output(history)
        
        predictions = [choice['output'] for choice in choices]
        optimal_prediction = torch.zeros_like(predictions[0])
        sub_optimal_predictions = [torch.zeros_like(predictions[0]) for _ in range(len(predictions) - 1)]

        for i in range(args.num_nodes):
            if self.data[self.counter]['final_answer'] is None:
                optimal_idx = 0
            else:
                optimal_idx = self.data[self.counter]['final_answer'] - 1
            prev_choices = self.select_data[self.counter]['choices']
            optimal_prediction[0, :, i, 0] = torch.tensor(prev_choices[optimal_idx], dtype=torch.float32, device=target.device)
            sub_optimal_indices = [idx for idx in range(len(predictions)) if idx != optimal_idx]
            for j, idx in enumerate(sub_optimal_indices):
                sub_optimal_predictions[j][0, :, i, 0] = torch.tensor(prev_choices[idx], dtype=torch.float32, device=target.device)
            
            if args.dump and iter == args.update_iters:
                dump_entry = get_dump_entry(time_idx, iter, history, target, choices, i, torch.stack([prediction[0, :, i, :] for prediction in predictions], dim=0))
                self.dump_data.append(dump_entry)
            
            self.counter += 1
        if args.update_selection and iter < args.update_iters:
            self.counter -= args.num_nodes
        
        optimal_prediction = optimal_prediction.detach()
        sub_optimal_predictions = [sub_optimal_prediction.detach() for sub_optimal_prediction in sub_optimal_predictions]

        optimal_prediction = normalize_output(optimal_prediction)
        sub_optimal_predictions = [normalize_output(sub_optimal_prediction) for sub_optimal_prediction in sub_optimal_predictions]

        result = {'output': optimal_prediction, 'positive_output': optimal_prediction, 'negative_output': sub_optimal_predictions}
        return result
    
    def __del__(self):
        if args.dump and self.dump_path is not None:
            import json
            with open(self.dump_path, 'w') as f:
                json.dump(self.dump_data, f)
