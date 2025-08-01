from .basic_model import BasicModel
import torch
import torch.nn as nn
from utils import *
from cfg import args
import math
from .graph import BasicLayer


class ModuleWithHypergraphLearning(nn.Module):
    def __init__(self, args, adj, depth=3, num_edges=32, hyper=None):
        super(ModuleWithHypergraphLearning, self).__init__()
        self.args = args
        self.depth = depth
        self.adj = adj
        self.hypers = HypergraphLearning(args, num_edges) if hyper is None else hyper
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        for i in range(self.depth):
            x = self.hypers(x)
            if i != self.depth - 1:
                x = self.dropout(x)
        return x
    

class HypergraphLearning(nn.Module):
    def __init__(self, args, num_edges):
        super(HypergraphLearning, self).__init__()
        self.args = args
        self.num_edges = num_edges
        self.edge_clf = torch.randn(args.hidden_dim, self.num_edges) / math.sqrt(self.num_edges)  # D x E
        self.edge_clf = nn.Parameter(self.edge_clf, requires_grad=True)
        self.edge_map = torch.randn(self.num_edges, self.num_edges) / math.sqrt(self.num_edges)
        self.edge_map = nn.Parameter(self.edge_map, requires_grad=True)  # E x E
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(args.hidden_dim)

    def forward(self, x):  # B x T x N x D
        feat = x.reshape(x.size(0), -1, x.size(3))  # B x TN x D
        hyper_assignment = torch.softmax(feat @ self.edge_clf, dim=-1)  # B x TN x E
        hyper_feat = hyper_assignment.transpose(1, 2) @ feat  # B x E x D
        hyper_feat_mapped = self.activation(self.edge_map @ hyper_feat)
        hyper_out = hyper_feat_mapped + hyper_feat
        y = self.activation(hyper_assignment @ hyper_out)
        y = y.reshape(x.size(0), x.size(1), x.size(2), x.size(3))
        y_final = self.norm(y + x)
        return y_final


class HypergraphBackbone(nn.Module):
    def __init__(self, num_layers):
        super(HypergraphBackbone, self).__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(*[BasicLayer(args.predefined_adjs[i],
                                     use_learned_adj=False, padding=2)
                            for j in range(num_layers)])
            for i in range(2)])

    def forward(self, feature):
        feature_list = []
        for layer in self.layers:
            x = layer(feature)
            feature_list.append(x)
        feature = torch.stack(feature_list, dim=3).max(dim=3)[0]  # B x T x N x D
        return feature


class HypergraphBranchEncoder(nn.Module):
    def __init__(self):
        super(HypergraphBranchEncoder, self).__init__()
        self.adj = args.adj
        self.backbone = HypergraphBackbone(args.hgnn_num_backbone_layers)
        self.hyper = HypergraphLearning(args, args.hgnn_num_hyper_edge)
        self.hgnn_core = ModuleWithHypergraphLearning(args, adj=self.adj, depth=args.hgnn_num_head_layers, hyper=self.hyper)

    def forward(self, x):
        x = self.backbone(x)
        output = self.hgnn_core(x)
        return output


class HypergraphBranch(BasicModel):
    def __init__(self, *params, **options):
        super().__init__(*params, **options)
        self.encoder = HypergraphBranchEncoder()

    def model_forward(self, feat):
        return self.encoder(feat)
    
    def get_finetune_params(self):
        finetune_params = []
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.LayerNorm):
                finetune_params.extend(list(module.parameters()))
        return finetune_params
