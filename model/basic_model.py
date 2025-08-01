import torch
import torch.nn as nn
from cfg import args
from abc import ABC, abstractmethod


class BasicModel(nn.Module, ABC):
    def __init__(self, *params, **options):
        super(BasicModel, self).__init__()
        self.embeddings = nn.ModuleDict({
            'input': nn.Linear(args.in_dim, args.hidden_dim),
            'node': nn.Embedding(args.num_nodes, args.hidden_dim),
            'time_of_day': nn.Embedding(288, args.hidden_dim),
            'day_of_week': nn.Embedding(7, args.hidden_dim)
        })
        self.output_layer = nn.Linear(args.hidden_dim * args.seq_in_len, args.out_dim * args.seq_out_len)
        self.collapse_t = False

    @abstractmethod
    def model_forward(self, feat): # B x T x N x D -> B x T x N x D
        pass

    def get_finetune_params(self):
        raise NotImplementedError
    
    def embed(self, data):
        x = data['x']
        input_emb = self.embeddings['input'](x)  # B x T x N x D
        node_idx = torch.arange(0, args.num_nodes).to(args.device)  # N
        node_emb = self.embeddings['node'](node_idx).unsqueeze(0).unsqueeze(0)  # 1 x 1 x N x D
        time_emb = self.embeddings['time_of_day'](data['time_of_day'].long())
        day_emb = self.embeddings['day_of_week'](data['day_of_week'].long())
        feat = input_emb + node_emb + time_emb + day_emb  # B x T x N x D
        return feat
    
    def head_forward(self, out_feat):
        batch_size = out_feat.size(0)
        num_nodes = args.num_nodes
        if not self.collapse_t:
            out_feat = out_feat.transpose(1, 2).reshape((batch_size, num_nodes, -1))  # B x N x (T x D)
            prediction = self.output_layer(out_feat)  # B x N x (T x args.out_dim)
        else:
            # B x N x D
            prediction = self.output_layer(out_feat)  # B x N x D -> B x N x (T x args.out_dim)
        
        prediction = prediction.reshape((batch_size, num_nodes, -1, args.out_dim))  # B x N x T x args.out_dim
        prediction = prediction.transpose(1, 2)  # B x T x N x args.out_dim
        return prediction
    
    def forward(self, data, *params, **options):
        feat = self.embed(data)
        out_feat = self.model_forward(feat)  # B x T x N x D
        prediction = self.head_forward(out_feat)
        
        return prediction