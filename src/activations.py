import torch
import torch.nn as nn
import torch.nn.functional as F
import difw
from difw.core.utility import Parameters
from difw.core.tessellation import Tessellation
import torch_geometric.nn as pyg
import einops
from torch_geometric.utils import to_dense_batch
from layers import GINEReLU, GCN2ReLU, GCNReLU, GINReLU, GATReLU, GCN22ReLU, GAT22ReLU, SAGEReLU


def interpolate_to_support(x: torch.Tensor, y: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """
    This is a rudimentary implementation of numpy.interp for the 1D case only. If the x values are not unique, this behaves differently to np.interp.
    :param x: The original coordinates.
    :param y: The original values.
    :param support: The support points to which y shall be interpolated.
    :return:
    """
    # Evaluate the forward difference for all except the edge points
    slope = torch.zeros_like(x)
    slope[1:-1] = ((y[1:] - y[:-1]) / (x[1:] - x[:-1]))[1:]

    # Evaluate which of the support points are within the range of x
    support_nonzero_mask = (support >= x.min()) & (support <= x.max())
    # Subset the support points accordingly
    support_nonzero = support[support_nonzero_mask]
    # Get the indices of the closest point to the left for each support point
    support_insert_indices = torch.searchsorted(x, support_nonzero)
    # Get the offset from the point to the left to the support point
    support_nonzero_offset = support_nonzero - x[support_insert_indices]
    # Calculate the value for the nonzero support: value of the point to the left plus slope times offset
    support_nonzero_values = y[support_insert_indices] + slope[support_insert_indices - 1] * support_nonzero_offset

    # Create the output tensor and place the nonzero support
    support_values = torch.zeros_like(support).float()
    support_values[support_nonzero_mask] = support_nonzero_values

    return support_values


import torch


def interpolate_linear_1d(x, y, x_query):
    """
    Perform linear interpolation for 1D data.

    Args:
        x (Tensor): 1D tensor of x-coordinates (sorted in ascending order).
        y (Tensor): 1D tensor of y-coordinates corresponding to x.
        x_query (Tensor): 1D tensor of query points for interpolation.

    Returns:
        Tensor: 1D tensor of interpolated values at x_query.
    """
    # Ensure inputs are tensors
    # x = torch.tensor(x)
    # y = torch.tensor(y)
    # x_query = torch.tensor(x_query)

    # Find indices of closest points for each query point
    indices = torch.searchsorted(x, x_query)

    # Handle boundary cases
    indices = torch.clamp(indices, 1, len(x) - 1)
    return indices
    # Get neighboring points and values for interpolation
    x_left = x[indices - 1]
    x_right = x[indices]
    y_left = y[indices - 1]
    y_right = y[indices]

    # Compute linear interpolation
    slope = (y_right - y_left) / (x_right - x_left)
    interpolated_values = y_left + slope * (x_query - x_left)

    return interpolated_values


class MyCpab(difw.Cpab):
    def __init__(self, xmin, xmax,
                 tess_size: int,
                 backend: str = "numpy",
                 device: str = "cpu",
                 zero_boundary: bool = True,
                 basis: str = "rref", ):

        super(MyCpab, self).__init__(tess_size, backend, device, zero_boundary, basis)
        self._check_input(tess_size, backend, device, zero_boundary, basis)

        # Parameters
        self.params = Parameters()
        self.params.nc = tess_size
        self.params.zero_boundary = zero_boundary
        self.params.xmin = xmin
        self.params.xmax = xmax
        self.params.nSteps1 = 10
        self.params.nSteps2 = 5
        self.params.precomputed = False
        self.params.use_slow = False
        self.params.basis = basis

        # Initialize tesselation
        self.tess = Tessellation(
            self.params.nc, self.params.xmin, self.params.xmax, self.params.zero_boundary, basis=self.params.basis,
        )

        # Extract parameters from tesselation
        self.params.B = self.tess.B
        self.params.D, self.params.d = self.tess.D, self.tess.d

        self.backend_name = backend

        # Load backend and set device
        self.device = device.lower()
        self.backend_name = backend
        if self.backend_name == "numpy":
            from difw.backend.numpy import functions as backend
        elif self.backend_name == "pytorch":
            from difw.backend.pytorch import functions as backend

            self.params.B = backend.to(self.params.B, device=self.device)
            self.params.B = self.params.B.contiguous()
        self.backend = backend

        # Assert that we have a recent version of the backend
        self.backend.assert_version()


class IdentityActivation(nn.Module):
    def __init__(self):
        super(IdentityActivation, self).__init__()

    def forward(self, x, edge_index, edge_attr, batch, time):
        return x, -1


class ReLUActivation(nn.Module):
    def __init__(self):
        super(ReLUActivation, self).__init__()

    def forward(self, x, edge_index, edge_attr, batch, time):
        return F.relu(x), -1


class CPABActivationSame(nn.Module):
    def __init__(self, radius, tess_size=16, zero_boundary=True, channel=1, transform_theta=False, use_tanh=False):
        super(CPABActivationSame, self).__init__()
        self.transform_theta = transform_theta
        self.use_tanh = use_tanh
        self.T = MyCpab(0, 1, tess_size, backend='pytorch', device='gpu', zero_boundary=zero_boundary)
        self.theta = nn.Parameter(torch.randn(channel, tess_size - 1 if zero_boundary else tess_size + 1),
                                  requires_grad=True)
        self.theta.requires_grad = True
        self.channel = channel
        self.radius = radius
        self.tess_size = tess_size - 1 if zero_boundary else tess_size + 1

    def forward(self, x, edge_index, edge_attr, batch, time):
        # x.shape = [n_nodes, n_channels]
        # We are applying the transformation for each channel
        # and we are treating each node as a point in the grid
        # So, as per libcpab, we have the following shape correspondence:
        # channel = 1 (this will later be a batch of graphs, for Cora, this is 1)
        # n_channels = n_channels
        # width = n_nodes

        # Notebook | Code
        # batch_size | n_channel
        # n_channel | 1
        # n_features | n_nodes
        # same theta for all channels
        # TODO: make sure that it uses the integration time internally

        theta = self.theta
        if self.transform_theta:
            if self.use_tanh:
                theta = torch.tanh(theta)
            else:
                theta = 2 * ((theta - (theta).min(dim=0)[0]) / (theta.max(dim=0)[0] - theta.min(dim=0)[0])) - 1



        x_flat = x.flatten()

        # Sort the features
        # x_flat_sorted, perm = torch.sort(x_flat)
        x_flat_sorted = x_flat
        # define boundaries check who doesnt comply with rules:
        x_flat_before_act = x_flat_sorted.clone()

        # rescale the values to (0, 1) because CPAB expects the same
        radius = self.radius
        x_flat_sorted = (x_flat_sorted + radius) / (2 * radius)
        # Find all index that do not lie in the domain of CPAB, used later to set the T(x) = x for x outside the domain
        not_within_domain = torch.logical_or((x_flat_sorted >= 1), (x_flat_sorted <= 0))
        x_t = self.T.transform_grid(x_flat_sorted, theta, time=time)
        x_t = x_t.squeeze()
        # rescale the values back to original domain
        x_t = (x_t) * 2 * radius - radius
        ## exterminate:
        x_t[not_within_domain] = x_flat_before_act[not_within_domain]

        # unsort the features
        # unsorted_x_flat = torch.gather(x_t, 0, perm.argsort())
        unsorted_x_flat = x_t
        x_t2 = unsorted_x_flat.squeeze()

        x = x_t2.reshape(x.shape)
        return x, theta


class CPABActivationDifferent(nn.Module):
    def __init__(self,
                 radius,
                 tess_size=16,
                 zero_boundary=True,
                 channel=1):
        super(CPABActivationDifferent, self).__init__()
        self.T = MyCpab(0, 1, tess_size, backend='pytorch', device='gpu', zero_boundary=zero_boundary)
        self.theta = nn.Parameter(torch.zeros(channel, tess_size - 1 if zero_boundary else tess_size + 1),
                                  requires_grad=True)
        self.theta.requires_grad = True
        self.channel = channel
        self.radius = radius
        self.tess_size = tess_size - 1 if zero_boundary else tess_size + 1

    def forward(self, x, edge_index, edge_attr, batch, time):
        theta = self.theta
        if self.transform_theta:
            if self.use_tanh:
                theta = torch.tanh(theta)
            else:
                theta = 2 * ((theta - (theta).min(dim=0)[0]) / (theta.max(dim=0)[0] - theta.min(dim=0)[0])) - 1

        x_reshaped = x.T
        # Sort the features
        x_reshaped_sorted, perm = torch.sort(x_reshaped, dim=1)
        # define boundaries check who doesnt comply with rules:
        x_reshaped_before_act = x_reshaped_sorted.clone()

        # rescale the values to (0, 1) because CPAB expects the same
        radius = self.radius
        x_reshaped_sorted = (x_reshaped_sorted + radius) / (2 * radius)
        # Find all index that do not lie in the domain of CPAB, used later to set the T(x) = x for x outside the domain
        not_within_domain = torch.logical_or((x_reshaped_sorted >= 1), (x_reshaped_sorted <= 0))
        x_t = self.T.transform_grid(x_reshaped_sorted, theta, time=time)
        # rescale the values back to original domain
        x_t = (x_t) * 2 * radius - radius
        ## exterminate:
        x_t[not_within_domain] = x_reshaped_before_act[not_within_domain]

        # unsort the features
        unsorted_x_reshaped = torch.gather(x_t, 1, perm.argsort(1))
        x_t2 = unsorted_x_reshaped.squeeze()

        # transpose it back
        x = x_t2.T

        return x, theta


class CPABActivationGNN(nn.Module):
    def __init__(self,
                 radius,
                 in_channels,
                 hidden_channels,
                 pooling_fn='mean',
                 num_layers=2,
                 tess_size=16,
                 zero_boundary=True,
                 channel=1,
                 backbone="gine",
                 transform_theta=False,
                 use_bond_encoder=False,
                 use_tanh=False,
                 dropout = 0,
                 **kwargs):
        super(CPABActivationGNN, self).__init__()
        self.dropout = dropout
        self.T = MyCpab(0, 1, tess_size, backend='pytorch', device='gpu',
                        zero_boundary=zero_boundary)
        self.channel = channel
        self.tess_size = tess_size - 1 if zero_boundary else tess_size + 1
        self.zero_boundary = zero_boundary
        out_channels = self.tess_size
        self.radius = radius
        self.transform_theta = transform_theta
        self.use_tanh = use_tanh
        self.theta_gnn = ThetaGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            pooling_fn=pooling_fn,
            backbone=backbone,
            use_bond_encoder=use_bond_encoder,
            dropout=dropout,
            **kwargs)

    def forward(self, x, edge_index, edge_attr, batch, time):
        theta = self.theta_gnn(x, edge_index, edge_attr, batch)
        # theta = F.sigmoid(theta)
        # B = theta.shape[0]
        # T = self.tess_size
        # C = self.channel
        # theta = theta.view((B, C, T))
        theta_orig = theta.clone()
        ####
        if False:
            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)
            # start.record()

            base_points = torch.linspace(0, 1, 100, device=x.device)
            dict_points = base_points.unsqueeze(0)
            dict_points = torch.repeat_interleave(dict_points, repeats=theta.shape[0], dim=0)
            if self.transform_theta:
                if self.use_tanh:
                    theta = torch.tanh(theta)
                else:
                    theta = 2 * ((theta - (theta).min(dim=0)[0]) / (theta.max(dim=0)[0] - theta.min(dim=0)[0])) - 1

            transform_dict = self.T.transform_grid(dict_points, theta, time=time)

            x_rescaled = (x + self.radius) / (2 * self.radius)
            # Find all index that do not lie in the domain of CPAB, used later to set the T(x) = x for x outside the domain
            not_within_domain = torch.logical_or((x_rescaled >= 1), (x_rescaled <= 0))

            x_batch, mask = to_dense_batch(x_rescaled, batch)
            x_batch_2 = x_batch.flatten()

            indices = interpolate_linear_1d(base_points, base_points, x_batch_2)
            indices = indices.reshape(x_batch.shape[0], -1)
            if False:
                y_transform = torch.gather(transform_dict, 1, indices)
                y_transform = y_transform.reshape(x_batch.shape)

            ##
            x_left = base_points[indices - 1]
            x_right = base_points[indices]

            y_left = torch.gather(transform_dict, 1, indices - 1)
            y_right = torch.gather(transform_dict, 1, indices)

            # Compute linear interpolation
            slope = (y_right - y_left) / (x_right - x_left)
            interpolated_values = y_left + slope * (x_batch.reshape(x_batch.shape[0], -1) - x_left)
            y_transform = interpolated_values.reshape(x_batch.shape)

            ##

            xnew = y_transform[mask]
            xnew = (xnew) * 2 * self.radius - self.radius
            xnew[not_within_domain] = x[not_within_domain]
            # end.record()
            # torch.cuda.synchronize()

            # print(start.elapsed_time(end))
            return xnew, theta

            if False:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.plot(dict_points[0, :].squeeze().detach().cpu(), transform_dict[0, :].detach().cpu())
                plt.plot(dict_points[0, :].squeeze().detach().cpu(), transform_dict[1, :].detach().cpu())
                plt.plot(dict_points[0, :].squeeze().detach().cpu(), transform_dict[2, :].detach().cpu())
                plt.plot(dict_points[0, :].squeeze().detach().cpu(), transform_dict[3, :].detach().cpu())
                plt.plot(dict_points[0, :].squeeze().detach().cpu(), transform_dict[4, :].detach().cpu())
                plt.plot(dict_points[0, :].squeeze().detach().cpu(), transform_dict[5, :].detach().cpu())
                plt.plot(dict_points[0, :].squeeze().detach().cpu(), transform_dict[6, :].detach().cpu())
                plt.plot(dict_points[0, :].squeeze().detach().cpu(), transform_dict[7, :].detach().cpu())
                plt.show()
            ###
        else:
            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)
            # start.record()
            theta = einops.rearrange(theta, 'b (T c) -> (b c) T', T=self.tess_size)
            theta = theta.contiguous()

            x_batch, mask = to_dense_batch(x, batch)
            C = x_batch.shape[2]
            theta = torch.repeat_interleave(theta, C, dim=0)

            # N = x_batch.shape[1]
            # x_batch_permute = x_batch.permute(0, 2, 1)
            # x_reshaped = x_batch_permute.reshape((B * C, N))
            x_reshaped = einops.rearrange(x_batch, 'b n c -> (b c) n').contiguous()

            # Sort the featues based on value, because CPAB expects a sorted grid
            x_reshaped_sorted, perm = torch.sort(x_reshaped, dim=1)
            # x_reshaped_sorted = x_reshaped
            # define boundaries check who doesnt comply with rules:
            x_batch_before_act = x_reshaped_sorted.clone()

            # rescale the values to (0, 1) because CPAB expects the same
            radius = self.radius
            x_reshaped_sorted = (x_reshaped_sorted + radius) / (2 * radius)
            # Find all index that do not lie in the domain of CPAB, used later to set the T(x) = x for x outside the domain
            not_within_domain = torch.logical_or((x_reshaped_sorted >= 1), (x_reshaped_sorted <= 0))
            if self.transform_theta:
                if self.use_tanh:
                    theta = torch.tanh(theta)
                else:
                    theta = 2 * ((theta - (theta).min(dim=0)[0]) / (theta.max(dim=0)[0] - theta.min(dim=0)[0])) - 1
            x_t = self.T.transform_grid(x_reshaped_sorted, theta, time=time)
            # rescale the values back to original domain
            x_t = (x_t) * 2 * radius - radius
            ## exterminate:
            x_t[not_within_domain] = x_batch_before_act[not_within_domain]

            # unsort the features
            unsorted_x_reshaped = torch.gather(x_t, 1, perm.argsort(1))
            # unsorted_x_reshaped = x_t
            x_t2 = unsorted_x_reshaped.squeeze()

            # x_t = x_t.view((B, N, C))
            x_t2 = einops.rearrange(x_t2, '(b c) n -> b n c', b=x_batch.shape[0]).contiguous()

            # unbatch
            x = x_t2[mask]
            # end.record()
            # torch.cuda.synchronize()

            # print(start.elapsed_time(end))
            return x, theta_orig #theta


class ThetaGNN(nn.Module):
    def __init__(self,
                 backbone,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 pooling_fn,
                 use_bond_encoder,
                 dropout=0,
                 **kwargs):
        super(ThetaGNN, self).__init__()
        self.backbone = backbone
        self.dropout = dropout
        if self.backbone == "gine":

            self.gnn = GINEReLU(
                # emb_dim=in_channels,
                in_dim=in_channels,
                emb_dim=hidden_channels,
                num_layers=num_layers,
                graph_pooling=pooling_fn,
                num_tasks=None,
                use_bond_encoder=use_bond_encoder
            )

        elif self.backbone == "gcn2":

            self.gnn = GCN2ReLU(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                num_layers=num_layers,
                alpha=0.0, #kwargs["ALPHA_GCN2"],
                theta=0.0, #kwargs["THETA_GCN2"],
                pooling_fn=pooling_fn,
                dropout=dropout
            )
        
        elif self.backbone == "gat":
            self.gnn = GATReLU(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                num_layers=num_layers,
                heads=1,
                pooling_fn=pooling_fn,
                dropout=dropout,
            )
        
        elif self.backbone == "gat22":
            self.gnn = GAT22ReLU(
                in_dim=in_channels,
                emb_dim=hidden_channels,
                num_layers=num_layers,
                graph_pooling=pooling_fn,
                num_tasks=None,
                use_bond_encoder=use_bond_encoder
            )

        elif self.backbone == "gin":
            self.gnn = GINReLU(
                in_dim=in_channels,
                emb_dim=hidden_channels,
                out_dim=hidden_channels,
                num_layers=num_layers,
                graph_pooling=pooling_fn,
            )
        
        elif self.backbone == "gcn22":
            self.gnn = GCN22ReLU(
                in_dim=in_channels,
                emb_dim=hidden_channels,
                num_layers=num_layers,
                graph_pooling=pooling_fn,
                num_tasks=None,
                use_bond_encoder=use_bond_encoder,
            )
        
        elif self.backbone == "sage2":
            self.gnn = SAGEReLU(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                graph_pooling=pooling_fn,
                num_layers=num_layers
            )

        else:

            self.gnn = GCNReLU(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                num_layers=num_layers,
                pooling_fn=pooling_fn
            )
        self.lin = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, x, edge_index, edge_attr, batch):

        x = self.gnn(x, edge_index, edge_attr, batch)
        # if self.backbone == "gine":
        #    x = self.lin(x)
        x = self.lin(x)

        return x
