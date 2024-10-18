import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn as pyg
from ogb.graphproppred.mol_encoder import AtomEncoder

from activations import CPABActivationSame, CPABActivationDifferent, CPABActivationGNN, IdentityActivation, \
    ReLUActivation
from layers import GINELayer, GATLayer, GCN2Layer


class GINNet(nn.Module):
    def __init__(self,
                 in_dim,
                 emb_dim,
                 out_dim,
                 num_layers,
                 graph_pooling,
                 add_residual=True,
                 shared_activation=True,
                 radius=None,
                 theta_pooling=None,
                 theta_hidden_dim=None,
                 theta_num_layers=None,
                 time_integration=None,
                 tess_size=None,
                 transform_theta=False,
                 dropout=0.0,
                 activation="relu",
                 use_tanh=True,
                 channel_wise=False,
                 **kwargs):
        super(GINNet, self).__init__()

        self.add_residual = add_residual
        self.dropout = dropout

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

        self.shared_activation = shared_activation
        self.time_integration = time_integration

        if self.shared_activation:
            if activation == "relu":
                self.activation = ReLUActivation()
            elif activation == "identity":
                self.activation = IdentityActivation()
            elif activation == "cpab":
                self.activation = CPABActivationDifferent(
                    radius=radius,
                    tess_size=tess_size,
                    channel=emb_dim
                )
            elif activation == "cpab_same_theta":
                self.activation = CPABActivationSame(
                    radius=radius,
                    tess_size=tess_size,
                    channel=1,
                    transform_theta=transform_theta,
                    use_tanh=use_tanh
                )
            elif activation == "cpab_gnn":
                self.activation = CPABActivationGNN(
                    radius=radius,
                    in_channels=emb_dim,
                    hidden_channels=emb_dim,
                    pooling_fn=theta_pooling,
                    num_layers=theta_num_layers,
                    tess_size=tess_size,
                    channel=emb_dim if channel_wise else 1,
                    backbone="gin",
                    transform_theta=transform_theta,
                    use_bond_encoder=False,
                    use_tanh=use_tanh
                )
            else:
                self.activation = ReLUActivation()
        else:
            if activation == "relu":
                self.activation = ReLUActivation()
            elif activation == "identity":
                self.activation = IdentityActivation()
            elif activation == "cpab":
                self.activation = nn.ModuleList()
                for _ in range(num_layers):
                    self.activation.append(
                        CPABActivationDifferent(
                            radius=radius,
                            tess_size=tess_size,
                            channel=emb_dim,
                        )
                    )
            elif activation == "cpab_same_theta":
                self.activation = nn.ModuleList()
                for _ in range(num_layers):
                    self.activation.append(
                        CPABActivationSame(
                            radius=radius,
                            channel=1,
                            tess_size=tess_size,
                            transform_theta=transform_theta,
                            use_tanh=use_tanh

                        )
                    )
            elif activation == "cpab_gnn":
                self.activation = nn.ModuleList()
                for _ in range(num_layers):
                    self.activation.append(
                        CPABActivationGNN(
                            radius=radius,
                            in_channels=emb_dim,
                            hidden_channels=emb_dim,
                            pooling_fn=theta_pooling,
                            num_layers=theta_num_layers,
                            tess_size=tess_size,
                            channel=emb_dim if channel_wise else 1,
                            backbone="gin",
                            transform_theta=transform_theta,
                            use_bond_encoder=False,
                            use_tanh=use_tanh
                        )
                    )
            else:
                self.activation = ReLUActivation()

    def forward(self, x, edge_index, batch):
        x = F.relu(self.lins[0](x))
        L = len(self.gnn_layers)
        for layer, (gnn, bn) in enumerate(zip(self.gnn_layers, self.bn_layers)):
            h = bn(gnn(x, edge_index))
            if not self.shared_activation:
                h, _ = self.activation[layer](h, edge_index, None, batch,
                                              (layer + 1) / L if self.time_integration else 1)
            else:
                h, _ = self.activation(h, edge_index, None, batch, (layer + 1) / L if self.time_integration else 1)

            h = F.dropout(h, p=self.dropout, training=self.training)
            if self.add_residual:
                x = h + x
            else:
                x = h
        x = self.pool(x, batch)
        x = self.lins[1](x)

        return x


class GINENetv2(nn.Module):
    def __init__(self,
                 emb_dim,
                 num_layers,
                 graph_pooling,
                 activation,
                 add_residual=True,
                 num_tasks=1,
                 feature_encoder=None,
                 shared_activation=True,
                 radius=None,
                 theta_hidden_dim=None,
                 theta_pooling=None,
                 theta_num_layers=None,
                 time_integration=None,
                 tess_size=None,
                 use_bond_encoder=None,
                 transform_theta=None,
                 dropout=0.0,
                 use_tanh=None,
                 **kwargs):
        super(GINENetv2, self).__init__()

        self.dropout = dropout
        self.use_bond_encoder = use_bond_encoder

        if feature_encoder is not None:
            self.feature_encoder = feature_encoder
        else:
            if use_bond_encoder:
                self.feature_encoder = AtomEncoder(emb_dim=emb_dim)
            else:
                self.feature_encoder = nn.Embedding(num_embeddings=21, embedding_dim=emb_dim)
                nn.init.xavier_normal_(self.feature_encoder.weight)

        self.gnn_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        for _ in range(num_layers):
            if kwargs['backbone'] == "gine":
                self.gnn_layers.append(
                    GINELayer(
                        in_dim=emb_dim,
                        emb_dim=emb_dim,
                        use_bond_encoder=use_bond_encoder
                    )
                )
            elif kwargs['backbone'] == "gat22":
                self.gnn_layers.append(
                    GATLayer(
                        in_dim=emb_dim,
                        emb_dim=emb_dim,
                        use_bond_encoder=use_bond_encoder
                    )
                )
            elif kwargs['backbone'] == "gcn22":
                self.gnn_layers.append(
                    GCN2Layer(
                        layer=_,
                        emb_dim=emb_dim,
                        use_bond_encoder=use_bond_encoder
                    )
                )
            
            elif kwargs['backbone'] == "sage2":
                self.gnn_layers.append(
                    pyg.SAGEConv(
                        in_channels=emb_dim,
                        out_channels=emb_dim
                    )
                )
            self.bn_layers.append(
                nn.BatchNorm1d(emb_dim)
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

        self.shared_activation = shared_activation
        self.time_integration = time_integration

        if activation == "relu":
            self.activation = ReLUActivation()
        elif activation == "identity":
            self.activation = IdentityActivation()
        elif activation == "cpab":
            if self.shared_activation:
                self.activation = CPABActivationDifferent(
                    radius=radius,
                    tess_size=tess_size,
                    channel=emb_dim
                )
            else:
                self.activation = nn.ModuleList()
                for _ in range(num_layers):
                    self.activation.append(
                        CPABActivationDifferent(
                            radius=radius,
                            tess_size=tess_size,
                            channel=emb_dim
                        )
                    )
        elif activation == "cpab_same_theta":
            if self.shared_activation:
                self.activation = CPABActivationSame(
                    radius=radius,
                    tess_size=tess_size,
                    channel=1,
                    transform_theta=transform_theta,
                    use_tanh=use_tanh
                )
            else:
                self.activation = nn.ModuleList()
                for _ in range(num_layers):
                    self.activation.append(
                        CPABActivationSame(
                            radius=radius,
                            tess_size=tess_size,
                            channel=1,
                            transform_theta=transform_theta,
                            use_tanh=use_tanh
                        )
                    )
        elif activation == "cpab_gnn":
            if self.shared_activation:
                self.activation = CPABActivationGNN(
                    radius=radius,
                    in_channels=emb_dim,
                    hidden_channels=theta_hidden_dim,
                    pooling_fn=theta_pooling,
                    num_layers=theta_num_layers,
                    tess_size=tess_size,
                    channel=emb_dim if kwargs['channel_wise'] else 1,
                    backbone=kwargs['backbone'],
                    transform_theta=transform_theta,
                    use_bond_encoder=use_bond_encoder,
                    use_tanh=use_tanh
                )
            else:
                self.activation = nn.ModuleList()
                for _ in range(num_layers):
                    self.activation.append(
                        CPABActivationGNN(
                            radius=radius,
                            in_channels=emb_dim,
                            hidden_channels=theta_hidden_dim,
                            pooling_fn=theta_pooling,
                            num_layers=theta_num_layers,
                            tess_size=tess_size,
                            channel=emb_dim if kwargs['channel_wise'] else 1,
                            backbone=kwargs['backbone'],
                            transform_theta=transform_theta,
                            use_bond_encoder=use_bond_encoder,
                            use_tanh=use_tanh
                        )
                    )
        self.add_residual = add_residual

        self.final_layers = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.BatchNorm1d(2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, num_tasks)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = x_0 = self.feature_encoder(x.squeeze())
        thetas = []
        L = len(self.gnn_layers)

        for layer, (gnn, bn) in enumerate(zip(self.gnn_layers, self.bn_layers)):
            if isinstance(gnn, GCN2Layer):
                h = bn(gnn(x, x_0, edge_index, None))
                h = F.dropout(h, p=self.dropout, training=self.training)
            elif isinstance(gnn, pyg.SAGEConv):
                h = gnn(x, edge_index)
            else:
                h = bn(gnn(x, edge_index, edge_attr))

            if self.shared_activation:
                h, tet = self.activation(h, edge_index, edge_attr, batch, (layer + 1) / L if self.time_integration else 1)
            else:
                h, tet = self.activation[layer](h, edge_index, edge_attr, batch,
                                              (layer + 1) / L if self.time_integration else 1)

            thetas.append(tet)
            if self.use_bond_encoder:
                h = F.dropout(h, p=self.dropout, training=self.training)
            if self.add_residual:
                x = h + x
            else:
                x = h

        x = self.pool(x, batch)
        x = self.final_layers(x)
        return x, thetas


class GINENet(nn.Module):
    def __init__(self,
                 emb_dim,
                 num_layers,
                 graph_pooling,
                 activation,
                 add_residual=True,
                 num_tasks=1,
                 feature_encoder=None,
                 shared_activation=True,
                 radius=None,
                 theta_hidden_dim=None,
                 theta_pooling=None,
                 theta_num_layers=None,
                 time_integration=None,
                 tess_size=None,
                 use_bond_encoder=None,
                 transform_theta=None,
                 dropout=0.0,
                 use_tanh=None,
                 **kwargs):
        super(GINENet, self).__init__()
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.use_bond_encoder = use_bond_encoder

        if feature_encoder is not None:
            self.feature_encoder = feature_encoder
        else:
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

        self.shared_activation = shared_activation
        self.time_integration = time_integration

        if shared_activation == False:
            self.activation = nn.ModuleList()
            for i in range(num_layers):
                if activation == "cpab":
                    self.activation.append(CPABActivationDifferent(
                        radius=radius,
                        channel=self.emb_dim,
                        tess_size=tess_size
                    ))
                elif activation == "cpab_same_theta":
                    self.activation.append(CPABActivationSame(
                        radius=radius,
                        channel=1,
                        tess_size=tess_size,
                        transform_theta=transform_theta,
                        use_tanh=use_tanh
                    ))
                elif activation == "cpab_gnn":
                    self.activation.append(CPABActivationGNN(
                        radius=radius,
                        in_channels=self.emb_dim,
                        # hidden_channels=theta_hidden_dim,
                        hidden_channels=self.emb_dim,
                        pooling_fn=theta_pooling,
                        num_layers=theta_num_layers,
                        backbone="gine",
                        channel=self.emb_dim,
                        transform_theta=transform_theta,
                        tess_size=tess_size,
                        use_bond_encoder=use_bond_encoder,
                        use_tanh=use_tanh
                    ))
                elif activation == "identity":
                    self.activation.append(IdentityActivation())
                else:
                    self.activation.append(ReLUActivation())
        else:
            if activation == "cpab":
                self.activation = CPABActivationDifferent(
                    radius=radius,
                    channel=self.emb_dim,
                    tess_size=tess_size
                )
            elif activation == "cpab_same_theta":
                self.activation = CPABActivationSame(
                    radius=radius,
                    channel=1,
                    tess_size=tess_size,
                    transform_theta=transform_theta,
                    use_tanh=use_tanh
                )
            elif activation == "cpab_gnn":
                self.activation = CPABActivationGNN(
                    radius=radius,
                    in_channels=self.emb_dim,
                    # hidden_channels=theta_hidden_dim,
                    hidden_channels=self.emb_dim,
                    pooling_fn=theta_pooling,
                    num_layers=theta_num_layers,
                    backbone="gine",
                    channel=self.emb_dim,
                    transform_theta=transform_theta,
                    tess_size=tess_size,
                    use_bond_encoder=use_bond_encoder,
                    use_tanh=use_tanh
                )
                # self.activation = CPABActivationGNN(kwargs.get("SYMMETRIC_RADIUS", 1), kwargs.get("RADIUS", 10), self.emb_dim, kwargs.get("THETA_HIDDEN_DIM", 128), kwargs.get("THETA_POOLING", 'max'), kwargs.get("THETA_NUM_LAYERS", 6), channel=self.emb_dim, multi_graph=kwargs.get("MULTI_GRAPH", True))
            elif activation == "identity":
                self.activation = IdentityActivation()
            else:
                self.activation = ReLUActivation()

    def forward(self, x, edge_index, edge_attr, batch):

        x = self.feature_encoder(x.squeeze())
        thetas = []
        L = len(self.gnn_layers)
        for layer, (gnn, bn) in enumerate(zip(self.gnn_layers, self.bn_layers)):
            h = gnn(x, edge_index, edge_attr)
            h = bn(h)
            if not self.shared_activation:
                h, _ = self.activation[layer](h, edge_index, edge_attr, batch,
                                              (layer + 1) / L if self.time_integration else 1)
            else:
                h, _ = self.activation(h, edge_index, edge_attr, batch, (layer + 1) / L if self.time_integration else 1)
            thetas.append(_)
            if self.use_bond_encoder:
                h = F.dropout(h, p=self.dropout, training=self.training)
            if self.add_residual:
                x = h + x
            else:
                x = h

        x = self.pool(x, batch)
        out = self.final_layers(x)
        return out, thetas


class NodeClassifier(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 activation,
                 shared_activation,
                 backbone,
                 alpha_gcn2=None,
                 theta_gcn2=None,
                 radius=None,
                 theta_hidden_dim=None,
                 theta_pooling=None,
                 theta_num_layers=None,
                 time_integration=None,
                 transform_theta=True,
                 use_tanh=True,
                 tess_size=16,
                 **kwargs):
        super(NodeClassifier, self).__init__()

        self.shared_activation = shared_activation
        self.time_integration = time_integration
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if backbone == "gcn":
                if i == 0:
                    self.convs.append(
                        pyg.GCNConv(in_channels, hidden_channels)
                    )
                elif i == num_layers - 1:
                    self.convs.append(
                        pyg.GCNConv(hidden_channels, out_channels)
                    )
                else:
                    self.convs.append(
                        pyg.GCNConv(hidden_channels, hidden_channels)
                    )
            elif backbone == "gcn2":
                self.convs.append(
                    pyg.GCN2Conv(
                        hidden_channels,
                        alpha_gcn2,
                        theta_gcn2,
                        i + 1,
                        shared_weights=True,
                        normalize=True
                    )
                )
            elif backbone == "gat":
                self.convs.append(
                    pyg.GATConv(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        heads=2,
                        residual=True,
                        concat=False
                    )
                )
            elif backbone == "gin":
                mlp = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.BatchNorm1d(hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels)
                )
                self.convs.append(
                    pyg.GINConv(
                        nn=mlp,
                        train_eps=True
                    )
                )
            elif backbone == "sage":
                self.convs.append(pyg.SAGEConv(hidden_channels, hidden_channels))
            else:
                raise ValueError("Invalid backbone type.")

        if backbone == "gcn2" or backbone == "gat" or backbone == "gin" or backbone == "sage":
            self.lins = nn.ModuleList()
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

        if shared_activation == False:
            if activation == "cpab":
                self.activation = nn.ModuleList([
                    CPABActivationDifferent(
                        radius=radius,
                        channel=hidden_channels,
                        tess_size=tess_size
                    ) for _ in range(num_layers)
                ])
            elif activation == "cpab_same_theta":
                self.activation = nn.ModuleList([
                    CPABActivationSame(
                        radius=radius,
                        channel=1,
                        transform_theta=transform_theta,
                        use_tanh=use_tanh,
                        tess_size=tess_size
                    ) for _ in range(num_layers)
                ])
            elif activation == "cpab_gnn":
                self.activation = nn.ModuleList([
                    CPABActivationGNN(
                        radius=radius,
                        in_channels=hidden_channels,
                        hidden_channels=theta_hidden_dim,
                        pooling_fn=theta_pooling,
                        num_layers=theta_num_layers,
                        backbone=backbone,
                        channel=hidden_channels,
                        dropout=dropout,
                        tess_size=tess_size
                    ) for _ in range(num_layers)
                ])
            elif activation == "identity":
                self.activation = nn.ModuleList([
                    IdentityActivation() for _ in range(num_layers)
                ])
            else:
                self.activation = nn.ModuleList([
                    ReLUActivation() for _ in range(num_layers)
                ])
        else:
            if activation == "cpab":
                self.activation = CPABActivationDifferent(
                    radius=radius,
                    channel=hidden_channels
                )
            elif activation == "cpab_same_theta":
                self.activation = CPABActivationSame(
                    radius=radius,
                    channel=1,
                    transform_theta=transform_theta,
                    use_tanh=use_tanh
                )
            elif activation == "cpab_gnn":
                self.activation = CPABActivationGNN(
                radius=radius,
                in_channels=hidden_channels,
                hidden_channels=theta_hidden_dim,
                pooling_fn=theta_pooling,
                num_layers=theta_num_layers,
                backbone="gin",
                channel=hidden_channels,
                dropout=dropout,
                use_tanh=use_tanh,
                use_bond_encoder=False,
                tess_size=tess_size,
                transform_theta=transform_theta,
            )
            elif activation == "identity":
                self.activation = IdentityActivation()
            else:
                self.activation = ReLUActivation()

        self.bns = torch.nn.ModuleList()
        for i in range(num_layers):
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

    def forward(self, x, edge_index, batch):
        thetas = []
        if hasattr(self, "lins"):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x_0 = self.lins[0](x).relu()
        for i, conv in enumerate(self.convs):
            if isinstance(conv, pyg.conv.GCN2Conv):
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = conv(x, x_0, edge_index)
                if not self.shared_activation:
                    x, _ = self.activation[i](x, edge_index, None, batch,
                                              (i + 1) / len(self.convs) if self.time_integration else 1)
                else:
                    x, _ = self.activation(x, edge_index, None, batch,
                                           (i + 1) / len(self.convs) if self.time_integration else 1)
                thetas.append(_)
            elif isinstance(conv, pyg.conv.GATConv):
                x = self.bns[i](conv(x, edge_index))
                if not self.shared_activation:
                    x, _ = self.activation[i](x, edge_index, None, batch,
                                              (i + 1) / len(self.convs) if self.time_integration else 1)
                else:
                    x, _ = self.activation(x, edge_index, None, batch,
                                           (i + 1) / len(self.convs) if self.time_integration else 1)
                x = F.dropout(x, p=self.dropout, training=self.training)
                thetas.append(_)
            elif isinstance(conv, pyg.conv.SAGEConv):
                h = self.bns[i](conv(x, edge_index))
                if not self.shared_activation:
                    h, _ = self.activation[i](h, edge_index, None, batch,
                                              (i + 1) / len(self.convs) if self.time_integration else 1)
                else:
                    h, _ = self.activation(h, edge_index, None, batch,
                                           (i + 1) / len(self.convs) if self.time_integration else 1)
                thetas.append(_)
                h = F.dropout(h, p=self.dropout, training=self.training)
            elif isinstance(conv, pyg.conv.GINConv):
                h = self.bns[i](conv(x, edge_index))
                if not self.shared_activation:
                    h, _ = self.activation[i](h, edge_index, None, batch,
                                              (i + 1) / len(self.convs) if self.time_integration else 1)
                else:
                    h, _ = self.activation(h, edge_index, None, batch,
                                           (i + 1) / len(self.convs) if self.time_integration else 1)
                thetas.append(_)
                h = F.dropout(h, p=self.dropout, training=self.training)
                x = h + x
            else:
                x = conv(x, edge_index)
                if i != len(self.convs) - 1:
                    if not self.shared_activation:
                        x, _ = self.activation[i](x, edge_index, None, batch,
                                                  (i + 1) / len(self.convs) if self.time_integration else 1)
                    else:
                        x, _ = self.activation(x, edge_index, None, batch,
                                               (i + 1) / len(self.convs) if self.time_integration else 1)
                    thetas.append(_)
                x = F.dropout(x, p=self.dropout, training=self.training)
            # if hasattr(self, "lins"):
            #     if not self.shared_activation:
            #         x, _ = self.activation[i](x, edge_index, None, batch,
            #                                   (i + 1) / len(self.convs) if self.time_integration else 1)
            #     else:
            #         x, _ = self.activation(x, edge_index, None, batch,
            #                                (i + 1) / len(self.convs) if self.time_integration else 1)
            #     thetas.append(_)
            # else:
            #     if i != len(self.convs) - 1:
            #         if not self.shared_activation:
            #             x, _ = self.activation[i](x, edge_index, None, batch,
            #                                       (i + 1) / len(self.convs) if self.time_integration else 1)
            #         else:
            #             x, _ = self.activation(x, edge_index, None, batch,
            #                                    (i + 1) / len(self.convs) if self.time_integration else 1)
            #         thetas.append(_)

        if hasattr(self, "lins"):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins[1](x)

        return x.log_softmax(dim=-1), thetas






####################



class approxNet(nn.Module):
    def __init__(self,
                 emb_dim,
                 num_layers,
                 graph_pooling,
                 activation,
                 add_residual=True,
                 num_tasks=1,
                 feature_encoder=None,
                 shared_activation=True,
                 radius=None,
                 theta_hidden_dim=None,
                 theta_pooling=None,
                 theta_num_layers=None,
                 time_integration=None,
                 tess_size=None,
                 use_bond_encoder=None,
                 transform_theta=None,
                 dropout=0.0,
                 use_tanh=None,
                 **kwargs):
        super(GINENetv2, self).__init__()

        self.dropout = dropout
        self.use_bond_encoder = use_bond_encoder

        if feature_encoder is not None:
            self.feature_encoder = feature_encoder
        else:
            if use_bond_encoder:
                self.feature_encoder = AtomEncoder(emb_dim=emb_dim)
            else:
                self.feature_encoder = nn.Embedding(num_embeddings=21, embedding_dim=emb_dim)
                nn.init.xavier_normal_(self.feature_encoder.weight)

        self.gnn_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.gnn_layers.append(
                GINELayer(
                    in_dim=emb_dim,
                    emb_dim=emb_dim,
                    use_bond_encoder=use_bond_encoder
                )
            )
            self.bn_layers.append(
                nn.BatchNorm1d(emb_dim)
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

        self.shared_activation = shared_activation
        self.time_integration = time_integration

        if activation == "relu":
            self.activation = ReLUActivation()
        elif activation == "identity":
            self.activation = IdentityActivation()
        elif activation == "cpab":
            if self.shared_activation:
                self.activation = CPABActivationDifferent(
                    radius=radius,
                    tess_size=tess_size,
                    channel=emb_dim
                )
            else:
                self.activation = nn.ModuleList()
                for _ in range(num_layers):
                    self.activation.append(
                        CPABActivationDifferent(
                            radius=radius,
                            tess_size=tess_size,
                            channel=emb_dim
                        )
                    )
        elif activation == "cpab_same_theta":
            if self.shared_activation:
                self.activation = CPABActivationSame(
                    radius=radius,
                    tess_size=tess_size,
                    channel=1,
                    transform_theta=transform_theta,
                    use_tanh=use_tanh
                )
            else:
                self.activation = nn.ModuleList()
                for _ in range(num_layers):
                    self.activation.append(
                        CPABActivationSame(
                            radius=radius,
                            tess_size=tess_size,
                            channel=1,
                            transform_theta=transform_theta,
                            use_tanh=use_tanh
                        )
                    )
        elif activation == "cpab_gnn":
            if self.shared_activation:
                self.activation = CPABActivationGNN(
                    radius=radius,
                    in_channels=emb_dim,
                    hidden_channels=theta_hidden_dim,
                    pooling_fn=theta_pooling,
                    num_layers=theta_num_layers,
                    tess_size=tess_size,
                    channel=emb_dim if kwargs['channel_wise'] else 1,
                    backbone="gine",
                    transform_theta=transform_theta,
                    use_bond_encoder=use_bond_encoder,
                    use_tanh=use_tanh
                )
            else:
                self.activation = nn.ModuleList()
                for _ in range(num_layers):
                    self.activation.append(
                        CPABActivationGNN(
                            radius=radius,
                            in_channels=emb_dim,
                            hidden_channels=theta_hidden_dim,
                            pooling_fn=theta_pooling,
                            num_layers=theta_num_layers,
                            tess_size=tess_size,
                            channel=emb_dim if kwargs['channel_wise'] else 1,
                            backbone="gin",
                            transform_theta=transform_theta,
                            use_bond_encoder=use_bond_encoder,
                            use_tanh=use_tanh
                        )
                    )
        self.add_residual = add_residual

        self.final_layers = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.BatchNorm1d(2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, num_tasks)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        #x = self.feature_encoder(x.squeeze())
        thetas = []
        L = len(self.gnn_layers)

        for layer, (gnn, bn) in enumerate(zip(self.gnn_layers, self.bn_layers)):
            h = bn(gnn(x, edge_index, edge_attr))

            if self.shared_activation:
                h, tet = self.activation(h, edge_index, edge_attr, batch, (layer + 1) / L if self.time_integration else 1)
            else:
                h, tet = self.activation[layer](h, edge_index, edge_attr, batch,
                                              (layer + 1) / L if self.time_integration else 1)

            thetas.append(tet)
            if self.use_bond_encoder:
                h = F.dropout(h, p=self.dropout, training=self.training)
            if self.add_residual:
                x = h + x
            else:
                x = h

        x = self.pool(x, batch)
        x = self.final_layers(x)
        return x, thetas


class LinkPredictorBackbone(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 activation,
                 radius=None,
                 theta_hidden_dim=None,
                 theta_pooling=None,
                 theta_num_layers=None,
                 transform_theta=True,
                 use_tanh=True,
                 tess_size=16,
                 **kwargs):
        super(LinkPredictorBackbone, self).__init__()

        self.time_integration = 0
        self.convs = nn.ModuleList()
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.bns = nn.ModuleList()
        # self.convs.append(pyg.GCNConv(in_channels, hidden_channels))
        for i in range(num_layers):
            self.convs.append(
                pyg.GCNConv(hidden_channels, hidden_channels)
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        # self.convs.append(pyg.GCNConv(hidden_channels, out_channels))

        self.dropout = dropout

        if activation == "cpab":
            self.activation = CPABActivationDifferent(
                radius=radius,
                channel=hidden_channels
            )
        elif activation == "cpab_same_theta":
            self.activation = CPABActivationSame(
                radius=radius,
                channel=1,
                transform_theta=transform_theta,
                use_tanh=use_tanh
            )
        elif activation == "cpab_gnn":
            self.activation = CPABActivationGNN(
                radius=radius,
                in_channels=hidden_channels,
                hidden_channels=theta_hidden_dim,
                pooling_fn=theta_pooling,
                num_layers=theta_num_layers,
                backbone="gin",
                channel=hidden_channels,
                dropout=dropout,
                use_tanh=use_tanh,
                use_bond_encoder=False,
                tess_size=tess_size,
                transform_theta=transform_theta,
            )
        elif activation == "identity":
            self.activation = IdentityActivation()
        else:
            self.activation = ReLUActivation()

    def forward(self, x, edge_index, edge_attr, batch=None):
        thetas = []
        x = self.lins[0](x).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        for conv, bn in zip(self.convs, self.bns): # TODO: changed
            x = bn(conv(x, edge_index))
            x, _ = self.activation(x, edge_index, edge_attr, batch, 1)
            thetas.append(_)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # x = self.convs[-1](x, edge_index)
        x = self.lins[1](x)

        return x, thetas

class LinkPredictor(nn.Module):
    def __init__(self, in_channels=256, hidden_channels=256, out_channels=1, num_layers=3, dropout=0.0):
        super(LinkPredictor, self).__init__()
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
    
    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
    
    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return F.sigmoid(x)