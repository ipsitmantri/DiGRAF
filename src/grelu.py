import torch
import torch_geometric.nn as pyg
import torch.nn.functional as F
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn.conv.appnp import APPNP
import torch_geometric

import torch


def kron(a, b):
    """
    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast. The number of them mush
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    return res.reshape(siz0 + siz1)


def batched_kronecker_product(A, B):
    """
    Compute the batched Kronecker product of two tensors A and B.

    Args:
    - A: Tensor of shape (batch_size, m, n)
    - B: Tensor of shape (batch_size, p, q)

    Returns:
    - kronecker_product: Tensor of shape (batch_size, m * p, n * q)
    """
    batch_size, m, n = A.size()
    _, p, q = B.size()

    # Reshape A and B to be 3D tensors
    A_reshaped = A.view(batch_size, m, 1, n, 1)
    B_reshaped = B.view(batch_size, 1, p, 1, q)

    # Compute the Kronecker product
    kronecker_product = torch.einsum('bik, bjn -> bijnk', A_reshaped, B_reshaped)

    # Reshape the result to have shape (batch_size, m * p, n * q)
    kronecker_product = kronecker_product.view(batch_size, m * p, n * q)

    return kronecker_product


class GRELU(nn.Module):
    def __init__(self,
                 K=2,
                 emb_dim=64,
                 ):
        super(GRELU, self).__init__()
        self.emb_dim = emb_dim
        self.K = K
        self.mlp_node_wise = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(emb_dim, 1))

        self.mlp_channel_wise = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(emb_dim, 2 * K * emb_dim))
        self.diffusion = APPNP(K=1, alpha=0.1)
        self.pool = pyg.global_mean_pool
        # for m in self.mlp_node_wise.modules():
        #     if isinstance(m, torch.nn.Linear):
        #         torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x, edge_index, edge_attr, batch):
        diffuse_x = self.diffusion(x, edge_index)
        # node-wise block:
        gamma = self.mlp_node_wise(diffuse_x)
        gamma = torch.softmax(gamma, dim=0)

        # channel-wise block:
        pooled_diffuse_x = self.pool(diffuse_x, batch)
        pooled_diffuse_x = self.mlp_channel_wise(pooled_diffuse_x)
        pooled_diffuse_x = torch.tanh(pooled_diffuse_x)
        alpha_kc = pooled_diffuse_x[:, :self.K * self.emb_dim]
        beta_kc = pooled_diffuse_x[:, self.K * self.emb_dim:]
        alpha_kc = alpha_kc.reshape(alpha_kc.shape[0], self.K, self.emb_dim)  # BXKXC
        beta_kc = alpha_kc.reshape(beta_kc.shape[0], self.K, self.emb_dim)  # BXKXC
        # gamma is of shape NX1
        # kron products:
        batch_size = beta_kc.shape[0]
        out = torch.zeros_like(x)

        gamma_dense, gamma_dense_mask = torch_geometric.utils.to_dense_batch(gamma, batch, fill_value=0)
        alpha_batch = kron(alpha_kc, gamma_dense).reshape(batch_size, self.K, gamma_dense.shape[1],
                                                          self.emb_dim).transpose(1, 2)
        beta_batch = kron(beta_kc, gamma_dense).reshape(batch_size, self.K, gamma_dense.shape[1],
                                                        self.emb_dim).transpose(1, 2)
        xtmp = x.t().unsqueeze(0)
        out2 = alpha_batch[gamma_dense_mask, :, :].permute(1, 2, 0) * xtmp + beta_batch[gamma_dense_mask, :, :].permute(
            1, 2, 0)
        out = out2.max(dim=0)[0].t()
        if False:
            for b in range(batch_size):
                alpha = torch.kron(alpha_kc[b, :, :], gamma[batch == b].squeeze())  # KXCXN
                beta = torch.kron(beta_kc[b, :, :], gamma[batch == b].squeeze())  # KXCXN - > KXNXC
                alpha = alpha.reshape(self.K, self.emb_dim, -1)
                beta = beta.reshape(self.K, self.emb_dim, -1)

                # activate:
                # x is of shape nxc
                xtemp = x[batch == b].t().unsqueeze(0)
                xtemp = alpha * xtemp + beta
                xtemp = xtemp.max(dim=0)[0].t()
                out[batch == b] = xtemp

        return out


# act = GRELU(K=2, emb_dim=7)
# nnodes = 32
# import networkx as nx

# G = nx.erdos_renyi_graph(n=nnodes, p=0.5)
# from torch_geometric.utils import from_networkx

# graph = from_networkx(G)
# edge_index = graph.edge_index
# batch = torch.zeros(nnodes).long()
# batch[2:] = 1
# batch[4:] = 1

# x = torch.randn(nnodes, 7)
# print(x.shape)
# print(act(x, edge_index, None, batch).shape)
