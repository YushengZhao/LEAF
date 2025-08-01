from utils import *
from .selectors import *
from cfg import args


def triplet_ranking_loss(output, positive_output, negative_output, margin=0):
    metric_mae = lambda x, y: torch.abs(x - y).mean(dim=(0, 1, 3))
    metric_huber = lambda x, y: torch.nn.SmoothL1Loss(reduction='none')(x, y).mean(dim=(0, 1, 3))
    return torch.relu(metric_huber(output, positive_output) - metric_huber(output, negative_output) + margin).mean()


class BasicAdapter:
    def __init__(self, selector) -> None:
        self.selector = selector

    def select(self, data, models):
        choices = []
        outputs = []
        for i, model in enumerate(models):
            output = model(data)
            outputs.append(output)
            prediction = denormalize_output(output)  # B x T x N x args.out_dim
            choice = {'model_name': model.__class__.__name__, 'output': prediction}
            choices.append(choice)
        result = self.selector.select(data, choices)
        return outputs, choices, result
    
    def run(self, data, models, optimizers):
        for iter in range(args.update_iters):
            outputs, choices, result = self.select(data, models)
            for model, output, optimizer in zip(models, outputs, optimizers):
                optimizer.zero_grad()
                loss = triplet_ranking_loss(output, result['positive_output'], result['most_negative_output'])
                loss.backward()
                optimizer.step()

        outputs, choices, result = self.select(data, models)

        return result
    
        