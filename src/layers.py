import torch
import torch_geometric.nn as pyg
import torch.nn.functional as F
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from src.baseline_layers import GCN2



class GINReLU(nn.Module):
    def __init__(self,
                 in_dim,
                 emb_dim,
                 out_dim,
                 num_layers,
                 graph_pooling,
                 add_residual=True
                 ):
        super(GINReLU, self).__init__()

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
        
        if graph_pooling == 'sum':
            self.pool = pyg.global_add_pool
        elif graph_pooling == 'mean':
            self.pool = pyg.global_mean_pool
        elif graph_pooling == 'max':
            self.pool = pyg.global_max_pool
        elif graph_pooling == 'attention':
            self.pool = pyg.GlobalAttention(gate_nn=nn.Linear(emb_dim, 1))
        else:
            raise ValueError("Invalid graph pooling type.")
    
    def forward(self, x, edge_index, edge_attr, batch):
        x = F.relu(self.lins[0](x))

        for layer, (gnn, bn) in enumerate(zip(self.gnn_layers, self.bn_layers)):
            h = bn(gnn(x, edge_index))
            h = F.relu(h)

            if self.add_residual:
                x = h + x
            else:
                x = h
        x = self.pool(x, batch)
        x = self.lins[1](x)

        return x

class GATLayer(nn.Module):
    def __init__(self, in_dim, emb_dim, num_edge_emb=4, use_bond_encoder=False):
        super(GATLayer, self).__init__()

        self.layer = pyg.GATConv(
            in_channels=in_dim,
            out_channels=emb_dim,
        )

        if use_bond_encoder:
            self.edge_emb = BondEncoder(emb_dim=in_dim)
        else:
            self.edge_emb = nn.Embedding(num_embeddings=num_edge_emb, embedding_dim=in_dim)
            nn.init.xavier_uniform_(self.edge_emb.weight)
    
    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index, self.edge_emb(edge_attr))

class GAT22ReLU(nn.Module):
    def __init__(self,
                in_dim,
                emb_dim, 
                num_layers, 
                graph_pooling, 
                add_residual=False, num_tasks=1,
                feature_encoder=None,
                use_bond_encoder=False,
                **kwargs):
        super(GAT22ReLU, self).__init__()
        self.emb_dim = emb_dim
        self.in_dim = in_dim
        # if use_bond_encoder:
        #     self.feature_encoder = AtomEncoder(emb_dim=emb_dim)
        # else:
        #     if feature_encoder is not None:
        #         self.feature_encoder = feature_encoder
        #     else:
        #         self.feature_encoder = nn.Embedding(num_embeddings=21, embedding_dim=emb_dim)
        #         nn.init.xavier_normal_(self.feature_encoder.weight)
        
        self.gnn_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        for i in range(num_layers):
            self.gnn_layers.append(
                GATLayer(in_dim, emb_dim, use_bond_encoder=use_bond_encoder)
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

        if graph_pooling == 'sum':
            self.pool = pyg.global_add_pool
        elif graph_pooling == 'mean':
            self.pool = pyg.global_mean_pool
        elif graph_pooling == 'max':
            self.pool = pyg.global_max_pool
        elif graph_pooling == 'attention':
            self.pool = pyg.GlobalAttention(gate_nn=nn.Linear(emb_dim, 1))
        else:
            raise ValueError("Invalid graph pooling type.")
    
    def forward(self, x, edge_index, edge_attr, batch):
        
        for gnn, bn in zip(self.gnn_layers, self.bn_layers):
            h = F.relu(bn(gnn(x, edge_index, edge_attr)))

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



class GCN2Layer(nn.Module):
    def __init__(self, emb_dim, layer, num_edge_emb=4, use_bond_encoder=False):
        super(GCN2Layer, self).__init__()

        # self.layer = pyg.GCN2Conv(
        #     emb_dim,
        #     alpha=0.0,
        #     theta=0.0,
        #     layer=layer+1
        # )
        self.layer = pyg.GCNConv(
            in_channels=emb_dim,
            out_channels=emb_dim,
            add_self_loops=False
        )
        if use_bond_encoder:
            self.edge_emb = BondEncoder(emb_dim=1)
        else:
            self.edge_emb = nn.Embedding(num_embeddings=num_edge_emb, embedding_dim=1)
            nn.init.xavier_uniform_(self.edge_emb.weight)
    
    def forward(self, x, x_0, edge_index, edge_attr):
        if edge_attr is None:
            return self.layer(x, edge_index)
        else:
        # print(x.shape, x_0.shape, edge_index.shape, self.edge_emb(edge_attr).shape)
            return self.layer(x, x_0, edge_index, self.edge_emb(edge_attr))

class GCN22ReLU(nn.Module):
    def __init__(self,
                in_dim,
                emb_dim, 
                num_layers, 
                graph_pooling, 
                add_residual=False, num_tasks=1,
                feature_encoder=None,
                use_bond_encoder=False,
                **kwargs):
        super(GCN22ReLU, self).__init__()
        self.emb_dim = emb_dim
        self.in_dim = in_dim
        # if use_bond_encoder:
        #     self.feature_encoder = AtomEncoder(emb_dim=emb_dim)
        # else:
        #     if feature_encoder is not None:
        #         self.feature_encoder = feature_encoder
        #     else:
        #         self.feature_encoder = nn.Embedding(num_embeddings=21, embedding_dim=emb_dim)
        #         nn.init.xavier_normal_(self.feature_encoder.weight)
        
        self.gnn_layers = nn.ModuleList()
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_dim, emb_dim))
        # self.lins.append(nn.Linear(hidden_channels, out_channels))
        # self.bn_layers = nn.ModuleList()

        for i in range(num_layers):
            self.gnn_layers.append(
                GCN2Layer(emb_dim, i, use_bond_encoder=use_bond_encoder)
            )
            in_dim = emb_dim
            # self.bn_layers.append(
            #     nn.BatchNorm1d(emb_dim, track_running_stats=True)
            # )

        self.add_residual = False
        self.final_layers = None
        if num_tasks is not None:
            self.final_layers = nn.Sequential(
                nn.Linear(emb_dim, 2 * emb_dim),
                nn.ReLU(),
                nn.Linear(2 * emb_dim, num_tasks)
            )

        if graph_pooling == 'sum':
            self.pool = pyg.global_add_pool
        elif graph_pooling == 'mean':
            self.pool = pyg.global_mean_pool
        elif graph_pooling == 'max':
            self.pool = pyg.global_max_pool
        elif graph_pooling == 'attention':
            self.pool = pyg.GlobalAttention(gate_nn=nn.Linear(emb_dim, 1))
        else:
            raise ValueError("Invalid graph pooling type.")
    
    def forward(self, x, edge_index, edge_attr, batch):
        x = x_0 = self.lins[0](x).relu()
        for gnn in self.gnn_layers:
            h = F.relu((gnn(x, x_0, edge_index, None))) # since alpha and theta are 0, x_0 doesnt matter
            x = h + x
        
        x = self.pool(x, batch)
        if self.final_layers is not None:
            out = self.final_layers(x)
        else:
            out = x
        return out

class GINELayer(nn.Module):
    def __init__(self, in_dim, emb_dim, num_edge_emb=20, use_bond_encoder=False):
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

class GINEReLU(nn.Module):
    def __init__(self,
                in_dim,
                emb_dim, 
                num_layers, 
                graph_pooling, 
                add_residual=False, num_tasks=1,
                feature_encoder=None,
                use_bond_encoder=False,
                **kwargs):
        super(GINEReLU, self).__init__()
        self.emb_dim = emb_dim
        self.in_dim = in_dim
        # if use_bond_encoder:
        #     self.feature_encoder = AtomEncoder(emb_dim=emb_dim)
        # else:
        #     if feature_encoder is not None:
        #         self.feature_encoder = feature_encoder
        #     else:
        #         self.feature_encoder = nn.Embedding(num_embeddings=21, embedding_dim=emb_dim)
        #         nn.init.xavier_normal_(self.feature_encoder.weight)
        
        self.gnn_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        for i in range(num_layers):
            self.gnn_layers.append(
                GINELayer(in_dim, emb_dim, use_bond_encoder=use_bond_encoder)
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

        if graph_pooling == 'sum':
            self.pool = pyg.global_add_pool
        elif graph_pooling == 'mean':
            self.pool = pyg.global_mean_pool
        elif graph_pooling == 'max':
            self.pool = pyg.global_max_pool
        elif graph_pooling == 'attention':
            self.pool = pyg.GlobalAttention(gate_nn=nn.Linear(emb_dim, 1))
        else:
            raise ValueError("Invalid graph pooling type.")
    
    def forward(self, x, edge_index, edge_attr, batch):
        
        for gnn, bn in zip(self.gnn_layers, self.bn_layers):
            h = F.relu(bn(gnn(x, edge_index, edge_attr)))

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

class GCN2ReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 alpha,
                 theta,
                 pooling_fn,
                 shared_weights=True,
                 dropout=0.0,
                 ):
        super(GCN2ReLU, self).__init__()

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                pyg.GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=True))

        self.dropout = dropout
        pooling_fns = {"sum": pyg.global_add_pool, "mean": pyg.global_mean_pool, "max": pyg.global_max_pool}
        self.pooling_fn = pyg.global_mean_pool#pooling_fns["pooling_fn"]
    
    def forward(self, x, edge_index, edge_attr, batch):
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for i, conv in enumerate(self.convs):
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.relu(conv(x, x_0, edge_index))
        
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.pooling_fn(x, batch)

        x = self.lins[1](x)
        return x

class GCNReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 pooling_fn):
        super(GCNReLU, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(pyg.GCNConv(in_channels, hidden_channels))
        for i in range(num_layers - 2):
            self.convs.append(pyg.GCNConv(hidden_channels, hidden_channels))
        self.convs.append(pyg.GCNConv(hidden_channels, hidden_channels))

        pooling_fns = {"sum": pyg.global_add_pool, "mean": pyg.global_mean_pool, "max": pyg.global_max_pool}
        self.pooling_fn = pooling_fns[pooling_fn] #pyg.global_add_pool #pyg.global_mean_pool #pooling_fns[pooling_fn]
    
    def forward(self, x, edge_index, edge_attr, batch):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
        
        x = self.convs[-1](x, edge_index)
        #x = F.dropout(x, self.dropout, training=self.training)
        #x = self.pooling_fn(x, batch)
        #x = x.sum(dim=0).unsqueeze(0)
        if batch is not None:
            x = self.pooling_fn(x, batch)

        return x
    
class GATReLU(nn.Module):
    def __init__(self,
                in_channels,
                hidden_channels,
                out_channels,
                num_layers,
                heads,
                pooling_fn,
                dropout=0.0):
        super(GATReLU, self).__init__()

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                pyg.GATConv(hidden_channels, hidden_channels, heads)
            )

        self.dropout = dropout
        pooling_fns = {"sum": pyg.global_add_pool, "mean": pyg.global_mean_pool, "max": pyg.global_max_pool}
        self.pooling_fn = pyg.global_mean_pool#pooling_fns["pooling_fn"]
    
    def forward(self, x, edge_index, edge_attr, batch):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[0](x).relu()

        for i, conv in enumerate(self.convs):
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.relu(conv(x, edge_index))
        
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.pooling_fn(x, batch)

        x = self.lins[1](x)
        return x

class SAGEReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 graph_pooling,
                 num_layers
                 ):
        super(SAGEReLU, self).__init__()

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, hidden_channels))

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(pyg.SAGEConv(hidden_channels, hidden_channels))
        
        if graph_pooling == 'sum':
            self.pool = pyg.global_add_pool
        elif graph_pooling == 'mean':
            self.pool = pyg.global_mean_pool
        elif graph_pooling == 'max':
            self.pool = pyg.global_max_pool
        elif graph_pooling == 'attention':
            self.pool = pyg.GlobalAttention(gate_nn=nn.Linear(hidden_channels, 1))
        else:
            raise ValueError("Invalid graph pooling type.")
    
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.lins[0](x).relu()

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        
        x = self.pool(x, batch)
        x = self.lins[1](x)
        
        return x