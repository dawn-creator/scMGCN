import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import GATConv
#GCN
class GCN(nn.Module):
    def __init__(self,in_feats,n_hidden,out_feats,n_layers,dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_feats, n_hidden, activation=F.elu))
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=F.elu))
        self.layers.append(GraphConv(n_hidden, out_feats))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g,features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h

#scMGCNlayer-attention
class scMGCNLayer(nn.Module):
    def __init__(self, num_graph, in_size, hidden_size,out_size, dropout,num_layers):
        super(scMGCNLayer, self).__init__()
        self.gcn_layers = nn.ModuleList()
        self.num_graph = num_graph
        for i in range(num_graph):
            self.gcn_layers.append(GCN(in_size,hidden_size,hidden_size,num_layers,dropout))
            #self.gcn_layers.append(GATConv(in_size,hidden_size,8,dropout,dropout,activation=F.elu))
        self.graph_attention = Attention(hidden_size)
        #self.graph_attention = Attention(hidden_size*8)

    def forward(self, gs, h):
        graph_embeddings = []
        for i, g in enumerate(gs):
            graph_embeddings.append(self.gcn_layers[i](g, h).flatten(1))
        graph_embeddings = torch.stack(graph_embeddings, dim=1)                  # (N, M, D * K)
        return self.graph_attention(graph_embeddings)                            # (N, D * K)

#attention
class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(Attention, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.model(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)
        return (beta * z).sum(1)                       # (N, D * K)


class scMGCN(nn.Module):
    def __init__(self, num_graph, in_size, hidden_size, out_size, dropout,num_layers):
        super(scMGCN, self).__init__()
        self.layer_attention = scMGCNLayer(num_graph, in_size, hidden_size,out_size, dropout,num_layers)
        self.predict = nn.Linear(hidden_size, out_size)
        self.epsilon = torch.FloatTensor([1e-12]).cuda()
        #self.predict = nn.Linear(hidden_size*8, out_size)
        '''
        self.post_mlp = 1
        def add_nonlinear_layers(num_hidden, feat_drop, bns=False):
            return [
                nn.BatchNorm1d(num_hidden, affine=False, track_running_stats=bns),
                nn.PReLU(),
                nn.Dropout(feat_drop)
            ]
        lr_output_layers = [
            [nn.Linear(hidden_size, hidden_size, bias=not False)] + add_nonlinear_layers(hidden_size, dropout, False)
            for _ in range(self.post_mlp - 1)]
        self.lr_output = nn.Sequential(*(
                [ele for li in lr_output_layers for ele in li] + [
            nn.Linear(hidden_size, out_size, bias=False),
            nn.BatchNorm1d(out_size, affine=False, track_running_stats=False)]))
        '''
    def forward(self, g, h):
        h = self.layer_attention(g, h)
        logits = self.predict(h)
        #logits = self.lr_output(h)
        logits = logits / (torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon))
        return logits
