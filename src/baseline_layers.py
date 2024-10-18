import torch
import torch_geometric.nn as pyg
import torch.nn.functional as F
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
#import torch_scatter
from torch_geometric.utils import add_self_loops
from grelu import GRELU

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(0.5), requires_grad=True)
    
    def forward(self, x):
        x = x * F.sigmoid(self.beta * x)
        return x

class GraphAdaptiveMax(pyg.MessagePassing):
    def __init__(self, K=1, emb_dim=16):
        super(GraphAdaptiveMax, self).__init__(aggr='max')
        assert K > 0
        self.K = K
        self.h = nn.ModuleList([nn.Linear(emb_dim, emb_dim) for _ in range(K)])
        self.beta = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.reset_parameters()
        
    def forward(self, x, edge_index):

        out = self.beta * F.relu(x)
        for i in range(self.K):
            xk = self.propagate(edge_index, x=x)
            out += self.h[i](xk)

        return out
    
    def message(self, x_j):
        return x_j

class GraphAdaptiveMedian(pyg.MessagePassing):
    def __init__(self, K=1, emb_dim=16):
        super(GraphAdaptiveMedian, self).__init__(aggr='median')
        assert K > 0
        self.K = K
        self.h = nn.ModuleList([nn.Linear(emb_dim, emb_dim) for _ in range(K)])
        self.beta = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.reset_parameters()
        
    def forward(self, x, edge_index):

        out = self.beta * F.relu(x)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])
        for i in range(self.K):
            xk = self.propagate(edge_index, x=x)
            out += self.h[i](xk)

        return out
    
    def message(self, x_j):
        return x_j

class MaxOut(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 num_pieces=2):
        super(MaxOut, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_pieces = num_pieces
        self.fc = nn.Linear(in_dim, out_dim * num_pieces)
    
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.out_dim, self.num_pieces)
        x, _ = x.max(dim=2)
        return x


class GIN(nn.Module):
    def __init__(self,
                 in_dim,
                 emb_dim,
                 out_dim,
                 num_layers,
                 activation,
                 add_residual=True,
                 num_pieces=2
                 ):
        super(GIN, self).__init__()

        self.add_residual = add_residual
        
        
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_dim, emb_dim))
        self.lins.append(nn.Linear(emb_dim, out_dim))
        self.gnn_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.BatchNorm1d(emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, emb_dim)
            )
            self.gnn_layers.append(pyg.GINConv(nn=mlp, train_eps=True))
            self.bn_layers.append(nn.BatchNorm1d(emb_dim))
        
        self.pool = pyg.global_add_pool
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "maxout":
            self.activation = MaxOut(emb_dim, emb_dim, num_pieces=num_pieces)
        elif activation == "max":
            self.activation = GraphAdaptiveMax(K=num_pieces, emb_dim=emb_dim)
        elif activation == "median":
            self.activation = GraphAdaptiveMedian(K=num_pieces, emb_dim=emb_dim)
        elif activation == "grelu":
            self.activation = GRELU(K=2, emb_dim=emb_dim)
        elif activation == "swish":
            self.activation = Swish()
        elif activation == 'elu':
            self.activation = torch.nn.ELU()
        elif activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid()
        else:
            self.activation = nn.Identity()
        self.activation_name = activation
    def forward(self, x, edge_index, batch):
        x = F.relu(self.lins[0](x))

        for layer, (gnn, bn) in enumerate(zip(self.gnn_layers, self.bn_layers)):
            h = bn(gnn(x, edge_index))
            if isinstance(self.activation, pyg.MessagePassing):
                h = self.activation(h, edge_index)
            elif self.activation_name == 'grelu':
                h = self.activation(h, edge_index, None, batch)
            else:
                h = self.activation(h)

            if self.add_residual:
                x = h + x
            else:
                x = h
        if batch is not None:
            x = self.pool(x, batch)
        x = self.lins[1](x)

        return x

class GINELayer(nn.Module):
    def __init__(self, in_dim, emb_dim, num_edge_emb=4, use_bond_encoder=False):
        super(GINELayer, self).__init__()
        mlp = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.BatchNorm1d(emb_dim, track_running_stats=True),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        
        self.layer = pyg.GINEConv(
            nn=mlp, train_eps=True
        )

        if use_bond_encoder:
            self.edge_emb = BondEncoder(emb_dim=in_dim)
        else:
            self.edge_emb = nn.Embedding(num_embeddings=num_edge_emb, embedding_dim=in_dim)
            nn.init.xavier_uniform_(self.edge_emb.weight)
    
    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index, self.edge_emb(edge_attr))
    
    def reset_parameters(self):
        self.edge_emb.reset_parameters()
        self.layer.reset_parameters()

class GATLayer(nn.Module):
    def __init__(self, in_dim, emb_dim, num_edge_emb=4, use_bond_encoder=False):
        super().__init__()
        self.layer = pyg.GATConv(in_dim, emb_dim)
        if use_bond_encoder:
            self.edge_emb = BondEncoder(emb_dim=in_dim)
        else:
            self.edge_emb = nn.Embedding(num_embeddings=num_edge_emb, embedding_dim=in_dim)
            nn.init.xavier_uniform_(self.edge_emb.weight)
    
    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index, self.edge_emb(edge_attr))

class GINE(nn.Module):
    def __init__(self,
                emb_dim, 
                num_layers, 
                activation,
                add_residual=False, num_tasks=1,
                use_bond_encoder=False,
                num_pieces=2,
                **kwargs):
        super(GINE, self).__init__()
        self.emb_dim = emb_dim
        if use_bond_encoder:
            self.feature_encoder = AtomEncoder(emb_dim=emb_dim)
        else:
            self.feature_encoder = nn.Embedding(num_embeddings=21, embedding_dim=emb_dim)
            nn.init.xavier_normal_(self.feature_encoder.weight)
        
        self.gnn_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        for i in range(num_layers):
            self.gnn_layers.append(
                GINELayer(emb_dim, emb_dim, use_bond_encoder=use_bond_encoder)
            )
            in_dim = emb_dim
            self.bn_layers.append(
                nn.BatchNorm1d(emb_dim, track_running_stats=True)
            )

        self.add_residual = add_residual
        self.final_layers = None
        if num_tasks is not None:
            self.final_layers = nn.Sequential(
                nn.Linear(emb_dim, 2 * emb_dim),
                nn.ReLU(),
                nn.Linear(2 * emb_dim, num_tasks)
            )

        self.pool = pyg.global_add_pool
        
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "maxout":
            self.activation = MaxOut(emb_dim, emb_dim, num_pieces=num_pieces)
        elif activation == "max":
            self.activation = GraphAdaptiveMax(K=num_pieces, emb_dim=emb_dim)
        elif activation == "median":
            self.activation = GraphAdaptiveMedian(K=num_pieces, emb_dim=emb_dim)
        elif activation == "grelu":
            self.activation = GRELU(K=2, emb_dim=emb_dim)
        elif activation == "swish":
            self.activation = Swish()
        elif activation == 'elu':
            self.activation = torch.nn.ELU()
        elif activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid()
        else:
            self.activation = nn.Identity()
        self.activation_name = activation

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.feature_encoder(x.squeeze())
        for gnn, bn in zip(self.gnn_layers, self.bn_layers):
            h = bn(gnn(x, edge_index, edge_attr))

            if isinstance(self.activation, pyg.MessagePassing):
                h = self.activation(h, edge_index)
            elif self.activation_name == 'grelu':
                h = self.activation(h, edge_index, None, batch)
            else:
                h = self.activation(h)

            if self.add_residual:
                x = h + x
            else:
                x = h
        
        x = self.pool(x, batch)
        if self.final_layers is not None:
            out = self.final_layers(x)
        else:
            out = x
        return out

class GCN2(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 activation,
                 alpha,
                 theta,
                 shared_weights=True,
                 dropout=0.0,
                 num_pieces=2,
                 ):
        super(GCN2, self).__init__()

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                pyg.GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=True))

        self.dropout = dropout
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "maxout":
            self.activation = MaxOut(hidden_channels, hidden_channels, num_pieces=num_pieces)
        elif activation == "max":
            self.activation = GraphAdaptiveMax(K=num_pieces, emb_dim=hidden_channels)
        elif activation == "median":
            self.activation = GraphAdaptiveMedian(K=num_pieces, emb_dim=hidden_channels)
        elif activation == "grelu":
            self.activation = GRELU(K=2, emb_dim=hidden_channels)
        elif activation == "swish":
            self.activation = Swish()
        elif activation == 'elu':
            self.activation = torch.nn.ELU()
        elif activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid()
        else:
            self.activation = nn.Identity()
        self.activation_name = activation
    
    def forward(self, x, edge_index):
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for i, conv in enumerate(self.convs):
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, edge_index)
            if isinstance(self.activation, pyg.MessagePassing):
                x = self.activation(x, edge_index)
            elif self.activation_name == 'grelu':
                x = self.activation(x, edge_index, None, torch.zeros(x.shape[0]).to(dtype=torch.long, device=x.device))
            else:
                x = self.activation(x)
            
        
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.lins[1](x)
        return x.log_softmax(dim=-1)

class GAT(nn.Module):
    def __init__(self, 
                 in_channels,
                 hidden_channels,
                 out_channels,
                 heads=8):
        super(GAT, self).__init__()
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))
        self.conv1 = pyg.GATConv(in_channels=hidden_channels, out_channels=hidden_channels, heads=heads, dropout=0.6)
        self.conv2 = pyg.GATConv(hidden_channels * heads, hidden_channels, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = self.lins[0](x).relu()
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.lins[1](x)
        return x

class GAT2(nn.Module):
    def __init__(self, 
                emb_dim,
                num_tasks, 
                num_layers,
                use_bond_encoder,
                ):
        super().__init__()

        self.num_layers = num_layers
        self.emb_dim = emb_dim
        
        if use_bond_encoder:
            self.feature_encoder = AtomEncoder(emb_dim=emb_dim)
        else:
            self.feature_encoder = nn.Embedding(num_embeddings=21, embedding_dim=emb_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATLayer(emb_dim, emb_dim, use_bond_encoder=use_bond_encoder))    

        if num_tasks is not None:
            self.final_layers = nn.Sequential(
                nn.Linear(emb_dim, 2 * emb_dim),
                nn.ReLU(),
                nn.Linear(2 * emb_dim, num_tasks)
            )

        self.pool = pyg.global_add_pool    

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.feature_encoder(x.squeeze())
        x = F.dropout(x, p=0.5, training=self.training)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.pool(x, batch)
        if self.final_layers is not None:
            x = self.final_layers(x)
        return x


class GIN2(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels, 
                 out_channels,
                 num_layers):
        super(GIN2, self).__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))

        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            conv = pyg.GINConv(mlp, train_eps=True)
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(hidden_channels))
    
    
    def forward(self, x, edge_index):
        x = self.lins[0](x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.lins[1](x)
        return x

class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.num_layers = num_layers
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))


        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(pyg.SAGEConv(hidden_channels, hidden_channels))        

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.register_parameters()

    def forward(self, x, edge_index):
        x = self.lins[0](x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.lins[1](x)
        return x


class SAGE2(nn.Module):
    def __init__(self, 
                emb_dim,
                num_tasks, 
                num_layers,
                use_bond_encoder,
                ):
        super().__init__()

        self.num_layers = num_layers
        self.emb_dim = emb_dim
        
        if use_bond_encoder:
            self.feature_encoder = AtomEncoder(emb_dim=emb_dim)
        else:
            self.feature_encoder = nn.Embedding(num_embeddings=21, embedding_dim=emb_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(pyg.SAGEConv(emb_dim, emb_dim))    

        if num_tasks is not None:
            self.final_layers = nn.Sequential(
                nn.Linear(emb_dim, 2 * emb_dim),
                nn.ReLU(),
                nn.Linear(2 * emb_dim, num_tasks)
            )

        self.pool = pyg.global_add_pool    

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.feature_encoder(x.squeeze())
        x = F.dropout(x, p=0.5, training=self.training)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.pool(x, batch)
        if self.final_layers is not None:
            x = self.final_layers(x)
        return x