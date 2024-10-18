import torch_geometric
import torch_geometric.transforms
from torch.utils.data import DataLoader as TDL
import wandb
from datetime import datetime
from data_utils import get_data
from models import GINENet, NodeClassifier, GINNet, GINENetv2, LinkPredictorBackbone, LinkPredictor
import torch.optim as optim
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import io
from torch_geometric.seed import seed_everything
from torch_geometric.loader import DataLoader
from PIL import Image
import einops
from ogb.graphproppred import Evaluator
from ogb.linkproppred import Evaluator as LinkEvaluator
from sklearn.model_selection import StratifiedKFold
import numpy as np


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_without_cpab(model):
    gnn_params = count_parameters(model.gnn_layers)
    bn_params = count_parameters(model.bn_layers)
    lin_params = count_parameters(model.final_layers)
    enc_params = count_parameters(model.feature_encoder)
    act_params = count_parameters(model.activation.theta_gnn)

    print(f"Model excluding theta_gnn: {gnn_params + bn_params + lin_params + enc_params}, Theta GNN: {act_params}", flush=True)


def cov_regu(thetas, cpabInstance):
    cov_cpa = cpabInstance.covariance_cpa(length_scale=0.1, output_variance=1)
    cov_loss = 0
    for curr_theta in thetas:
        cov_loss += (curr_theta @ cov_cpa @ curr_theta.t()).mean()
    cov_loss = cov_loss / len(thetas)
    return cov_loss


def on_train_epoch_end(model, track_thetas, device, uses_gnn=None, batch=None):
    # self.log("train_acc", self.train_accuracy)
    # TODO: additionaly, plot for the last 10 epochs, not as an animation, on the same plot
    grid = model.activation.T.uniform_meshgrid(100)
    # if not uses_gnn:
    fig = plt.figure()
    num_layers = len(track_thetas)
    grid = model.activation.T.uniform_meshgrid(100).to(device)
    for layer in range(num_layers):
        fig = plt.figure()
        theta_dummy = track_thetas[layer].clone().detach()
        # print(f"THETA: {theta_dummy}")
        # print(f"THETA_ABS_MAX: {theta_dummy.abs().max()}")
        # print(f"THETA_MIN: {torch.min(einops.rearrange(theta_dummy, '(b c) T -> b c T', c = 16), dim=2)[0]}")
        # print(f"THETA_MAX: {torch.min(einops.rearrange(theta_dummy, '(b c) T -> b c T', c = 16), dim=2)[0]}")
        grid_t = model.activation.T.transform_grid(grid, theta_dummy, time=1)
        plt.plot(grid.cpu().numpy(), grid_t.T.cpu().numpy())

        # plt.legend()
        fig.savefig("./try.png")
        plt.close()
    # else:
    #     for layer in range(uses_gnn):
    #         fig = plt.figure()
    #         for key, thetas in track_thetas.items():
    #             theta_dummy = thetas[layer].clone().detach().cpu()
    #             grid_t = model.activation.T.transform_grid(grid, theta_dummy, time=(layer+1)/num_layers)
    #             plt.plot(grid, grid_t.T, label=key)
    #         buf = io.BytesIO()
    #         fig.savefig(buf, format='png')
    #         buf.seek(0)
    #         image = Image.open(buf)
    #         image = wandb.Image(image)
    #         wandb.log({f"deformgrid_{layer}": [image]})


def get_optimizer(model, **configs):
    if configs['BACKBONE'] == "gcn":
        if configs['SHARED_ACTIVATION']:
            if configs['ACTIVATION'] in ['relu', 'identity']:
                optimizer = optim.Adam(
                    [dict(params=model.convs.parameters(), weight_decay=configs['CONV_WD'], lr=configs['CONV_LR'])])

            elif configs['ACTIVATION'] == "cpab_gnn":
                optimizer = optim.Adam([
                    dict(params=model.convs.parameters(), weight_decay=configs['CONV_WD'], lr=configs['CONV_LR']),
                    dict(params=model.activation.theta_gnn.parameters(), weight_decay=configs['THETA_WD'],
                         lr=configs['THETA_LR']),
                ])
                # optimizer = optim.Adam(model.parameters(),  weight_decay=configs['CONV_WD'], lr=configs['CONV_LR'])

            else:
                optimizer = optim.Adam([
                    dict(params=model.convs.parameters(), weight_decay=configs['CONV_WD'], lr=configs['CONV_LR']),
                    dict(params=[model.activation.theta], weight_decay=configs['THETA_WD'], lr=configs['THETA_LR']),
                ])
        else:
            if configs['ACTIVATION'] in ['relu', 'identity']:
                optimizer = optim.Adam(
                    [dict(params=model.convs.parameters(), weight_decay=configs['CONV_WD'], lr=configs['CONV_LR'])])

            elif configs['ACTIVATION'] == "cpab_gnn":
                optimizer = optim.Adam([
                    dict(params=model.convs.parameters(), weight_decay=configs['CONV_WD'], lr=configs['CONV_LR']),
                    dict(params=model.activation.theta_gnn.parameters(), weight_decay=configs['THETA_WD'],
                         lr=configs['THETA_LR']),
                ])

            else:
                optimizer = optim.Adam([
                    dict(params=model.convs.parameters(), weight_decay=configs['CONV_WD'], lr=configs['CONV_LR']),
                    dict(params=[layer.theta for layer in model.activation], weight_decay=configs['THETA_WD'],
                         lr=configs['THETA_LR']),
                ])
    elif configs['BACKBONE'] == "gcn2" or configs['BACKBONE'] == "gat":
        if configs["SHARED_ACTIVATION"]:
            if configs['ACTIVATION'] in ['relu', 'identity']:
                optimizer = optim.Adam([
                    dict(params=model.convs.parameters(), weight_decay=configs['CONV_WD'], lr=configs['CONV_LR']),
                    dict(params=model.lins.parameters(), weight_decay=configs['LINEAR_WD'], lr=configs['LINEAR_LR']),
                ])

            elif configs['ACTIVATION'] == "cpab_gnn":
                optimizer = optim.Adam([
                    dict(params=model.convs.parameters(), weight_decay=configs['CONV_WD'], lr=configs['CONV_LR']),
                    dict(params=model.lins.parameters(), weight_decay=configs['LINEAR_WD'], lr=configs['LINEAR_LR']),
                    dict(params=model.activation.theta_gnn.parameters(), weight_decay=configs['THETA_WD'],
                         lr=configs['THETA_LR']),
                ])
            else:

                optimizer = optim.Adam([
                    dict(params=model.convs.parameters(), weight_decay=configs['CONV_WD'], lr=configs['CONV_LR']),
                    dict(params=model.lins.parameters(), weight_decay=configs['LINEAR_WD'], lr=configs['LINEAR_LR']),
                    dict(params=[model.activation.theta], weight_decay=configs['THETA_WD'], lr=configs['THETA_LR']),
                ])
        else:
            if configs['ACTIVATION'] in ['relu', 'identity']:
                optimizer = optim.Adam([
                    dict(params=model.convs.parameters(), weight_decay=configs['CONV_WD'], lr=configs['CONV_LR']),
                    dict(params=model.lins.parameters(), weight_decay=configs['LINEAR_WD'], lr=configs['LINEAR_LR']),
                ])

            elif configs['ACTIVATION'] == "cpab_gnn":
                optimizer = optim.Adam([
                    dict(params=model.convs.parameters(), weight_decay=configs['CONV_WD'], lr=configs['CONV_LR']),
                    dict(params=model.lins.parameters(), weight_decay=configs['LINEAR_WD'], lr=configs['LINEAR_LR']),
                    dict(params=model.activation.parameters(), weight_decay=configs['THETA_WD'],
                         lr=configs['THETA_LR']),
                ])
            else:

                optimizer = optim.Adam([
                    dict(params=model.convs.parameters(), weight_decay=configs['CONV_WD'], lr=configs['CONV_LR']),
                    dict(params=model.lins.parameters(), weight_decay=configs['LINEAR_WD'], lr=configs['LINEAR_LR']),
                    dict(params=[layer.theta for layer in model.activation], weight_decay=configs['THETA_WD'],
                         lr=configs['THETA_LR']),
                ])

    elif configs["BACKBONE"] == "gine":
        if configs['SHARED_ACTIVATION']:
            if configs['ACTIVATION'] in ['relu', 'identity']:
                optimizer = optim.Adam([
                    dict(params=model.gnn_layers.parameters(), lr=configs['CONV_LR'], weight_decay=configs['CONV_WD']),
                    dict(params=model.bn_layers.parameters(), lr=configs['CONV_LR'], weight_decay=configs['CONV_WD']),
                    dict(params=model.feature_encoder.parameters(), lr=configs['CONV_LR'],
                         weight_decay=configs['CONV_WD']),
                    dict(params=model.final_layers.parameters(), lr=configs['CONV_LR'],
                         weight_decay=configs['CONV_WD']),
                ])
            elif configs['ACTIVATION'] == "cpab_gnn":
                optimizer = optim.Adam([
                    dict(params=model.gnn_layers.parameters(), lr=configs['CONV_LR'], weight_decay=configs['CONV_WD']),
                    dict(params=model.bn_layers.parameters(), lr=configs['CONV_LR'], weight_decay=configs['CONV_WD']),
                    dict(params=model.feature_encoder.parameters(), lr=configs['CONV_LR'],
                         weight_decay=configs['CONV_WD']),
                    dict(params=model.final_layers.parameters(), lr=configs['CONV_LR'],
                         weight_decay=configs['CONV_WD']),
                    dict(params=model.activation.theta_gnn.parameters(), lr=configs['THETA_LR'],
                         weight_decay=configs['THETA_WD']),
                ])
            else:
                optimizer = optim.Adam([
                    dict(params=model.gnn_layers.parameters(), lr=configs['CONV_LR'], weight_decay=configs['CONV_WD']),
                    dict(params=model.bn_layers.parameters(), lr=configs['CONV_LR'], weight_decay=configs['CONV_WD']),
                    dict(params=model.feature_encoder.parameters(), lr=configs['CONV_LR'],
                         weight_decay=configs['CONV_WD']),
                    dict(params=model.final_layers.parameters(), lr=configs['CONV_LR'],
                         weight_decay=configs['CONV_WD']),
                    dict(params=[model.activation.theta], lr=configs['THETA_LR'], weight_decay=configs['THETA_WD']),
                ])
        else:
            if configs['ACTIVATION'] in ['relu', 'identity']:
                optimizer = optim.Adam([
                    dict(params=model.gnn_layers.parameters(), lr=configs['CONV_LR'], weight_decay=configs['CONV_WD']),
                    dict(params=model.bn_layers.parameters(), lr=configs['CONV_LR'], weight_decay=configs['CONV_WD']),
                    dict(params=model.feature_encoder.parameters(), lr=configs['CONV_LR'],
                         weight_decay=configs['CONV_WD']),
                    dict(params=model.final_layers.parameters(), lr=configs['CONV_LR'],
                         weight_decay=configs['CONV_WD']),
                ])
            elif configs['ACTIVATION'] == "cpab_gnn":
                optimizer = optim.Adam([
                    dict(params=model.gnn_layers.parameters(), lr=configs['CONV_LR'], weight_decay=configs['CONV_WD']),
                    dict(params=model.bn_layers.parameters(), lr=configs['CONV_LR'], weight_decay=configs['CONV_WD']),
                    dict(params=model.feature_encoder.parameters(), lr=configs['CONV_LR'],
                         weight_decay=configs['CONV_WD']),
                    dict(params=model.final_layers.parameters(), lr=configs['CONV_LR'],
                         weight_decay=configs['CONV_WD']),
                    dict(params=model.activation.parameters(), lr=configs['THETA_LR'],
                         weight_decay=configs['THETA_WD']),
                ])
            else:
                optimizer = optim.Adam([
                    dict(params=model.gnn_layers.parameters(), lr=configs['CONV_LR'], weight_decay=configs['CONV_WD']),
                    dict(params=model.bn_layers.parameters(), lr=configs['CONV_LR'], weight_decay=configs['CONV_WD']),
                    dict(params=model.feature_encoder.parameters(), lr=configs['CONV_LR'],
                         weight_decay=configs['CONV_WD']),
                    dict(params=model.final_layers.parameters(), lr=configs['CONV_LR'],
                         weight_decay=configs['CONV_WD']),
                    dict(params=[layer.theta for layer in model.activation], lr=configs['THETA_LR'],
                         weight_decay=configs['THETA_WD']),
                ])
    elif configs['BACKBONE'] == "gin":
        if configs['SHARED_ACTIVATION']:
            if configs['ACTIVATION'] in ['relu', 'identity']:
                optimizer = optim.Adam([
                    dict(params=model.gnn_layers.parameters(), lr=configs['CONV_LR'], weight_decay=configs['CONV_WD']),
                    dict(params=model.bn_layers.parameters(), lr=configs['CONV_LR'], weight_decay=configs['CONV_WD']),
                    dict(params=model.lins.parameters(), lr=configs['CONV_LR'], weight_decay=configs['CONV_WD']),
                ])
            elif configs['ACTIVATION'] == "cpab_gnn":
                optimizer = optim.Adam([
                    dict(params=model.gnn_layers.parameters(), lr=configs['CONV_LR'], weight_decay=configs['CONV_WD']),
                    dict(params=model.bn_layers.parameters(), lr=configs['CONV_LR'], weight_decay=configs['CONV_WD']),
                    dict(params=model.lins.parameters(), lr=configs['CONV_LR'], weight_decay=configs['CONV_WD']),
                    dict(params=model.activation.theta_gnn.parameters(), lr=configs['THETA_LR'],
                         weight_decay=configs['THETA_WD']),
                ])
            else:
                optimizer = optim.Adam([
                    dict(params=model.gnn_layers.parameters(), lr=configs['CONV_LR'], weight_decay=configs['CONV_WD']),
                    dict(params=model.bn_layers.parameters(), lr=configs['CONV_LR'], weight_decay=configs['CONV_WD']),
                    dict(params=model.lins.parameters(), lr=configs['CONV_LR'], weight_decay=configs['CONV_WD']),
                    dict(params=[model.activation.theta], lr=configs['THETA_LR'], weight_decay=configs['THETA_WD']),
                ])
        else:
            if configs['ACTIVATION'] in ['relu', 'identity']:
                optimizer = optim.Adam([
                    dict(params=model.gnn_layers.parameters(), lr=configs['CONV_LR'], weight_decay=configs['CONV_WD']),
                    dict(params=model.bn_layers.parameters(), lr=configs['CONV_LR'], weight_decay=configs['CONV_WD']),
                    dict(params=model.lins.parameters(), lr=configs['CONV_LR'], weight_decay=configs['CONV_WD']),
                ])
            elif configs['ACTIVATION'] == "cpab_gnn":
                optimizer = optim.Adam([
                    dict(params=model.gnn_layers.parameters(), lr=configs['CONV_LR'], weight_decay=configs['CONV_WD']),
                    dict(params=model.bn_layers.parameters(), lr=configs['CONV_LR'], weight_decay=configs['CONV_WD']),
                    dict(params=model.lins.parameters(), lr=configs['CONV_LR'], weight_decay=configs['CONV_WD']),
                    dict(params=model.activation.parameters(), lr=configs['THETA_LR'],
                         weight_decay=configs['THETA_WD']),
                ])
            else:
                optimizer = optim.Adam([
                    dict(params=model.gnn_layers.parameters(), lr=configs['CONV_LR'], weight_decay=configs['CONV_WD']),
                    dict(params=model.bn_layers.parameters(), lr=configs['CONV_LR'], weight_decay=configs['CONV_WD']),
                    dict(params=model.lins.parameters(), lr=configs['CONV_LR'], weight_decay=configs['CONV_WD']),
                    dict(params=[layer.theta for layer in model.activation], lr=configs['THETA_LR'],
                         weight_decay=configs['THETA_WD']),
                ])
    else:
        raise ValueError("Invalid backbone")

    return optimizer


def train_one_epoch(model, optimizer, loss_module, data, flag=None):
    model.train()
    optimizer.zero_grad()

    if data.get('train_mask', None) is not None:
        x, edge_index = data.x, data.edge_index
        batch = torch.zeros(data.num_nodes).to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu")).long()
        #batch = torch.zeros((x.shape[0])).to(x.device).long()
        out, thetas = model(x, edge_index, batch)
        loss = loss_module(out[data.train_mask], data.y[data.train_mask])
    else:
        out, thetas = model(data, batch=data.batch)
        loss = loss_module(out, data.y)

    loss.backward()
    optimizer.step()
    return float(loss), thetas


@torch.no_grad()
def test(model, data, flag=None):
    model.eval()
    accs = []
    if data.get('train_mask', None) is None:
        x, edge_index = data.x, data.adj_t
        batch = torch.zeros((x.shape[0])).to(x.device).long()
        out, _ = model(x, edge_index, batch)
        pred = out.argmax(dim=-1)
        accs.append(int((pred == data.y).sum()) / int(data.y.shape[0]))
    else:
        x, edge_index = data.x, data.edge_index
        # batch = torch.ones_like(x).to(x.device)
        batch = torch.zeros((x.shape[0])).to(x.device).long()
        out, _ = model(x, edge_index, batch)
        pred = out.argmax(dim=-1)
        if flag != "Flickr":
            for mask in [data.train_mask, data.val_mask, data.test_mask]:
                accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
        else:
            for mask in [data.train_mask, data.test_mask]:
                accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))

    return accs


def train_node_classifier(**configs):
    if configs['TASK'] == 'train':
        wandb.login()
        wandb.finish()
        datetime_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run = wandb.init(project=configs['PROJECT_NAME'], name=f"{configs['ACTIVATION']}_{datetime_stamp}",
                         tags=configs['TAGS'])
        wandb.config.update({k: v for k, v in configs.items() if k not in ["wandb_sweep"]})

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    seed_everything(configs['SEED'])

    folds = 10
    if configs['DATASET_NAME'] == "Flickr" and configs['ACTIVATION'] == "relu":
        folds = 10
    val_accs = []
    test_accs = []
    for fold in range(folds):
        node_dataset = get_data(configs['DATASET_NAME'], configs['BACKBONE'])
        data = node_dataset[0].to(device)
        model = NodeClassifier(
            in_channels=node_dataset.num_node_features,
            hidden_channels=configs['HIDDEN_DIM'],
            out_channels=node_dataset.num_classes,
            num_layers=configs['NUM_LAYERS'],
            dropout=configs['DROPOUT'],
            activation=configs['ACTIVATION'],
            backbone=configs['BACKBONE'],
            shared_activation=configs['SHARED_ACTIVATION'],
            theta_hidden_dim=configs['THETA_HIDDEN_DIM'],
            theta_num_layers=configs['THETA_NUM_LAYERS'],
            theta_pooling=configs['THETA_POOLING'],
            alpha_gcn2=configs['ALPHA_GCN2'],
            theta_gcn2=configs['THETA_GCN2'],
            radius=configs['RADIUS'],
            accelerator=configs['ACCELERATOR'],
            time_integration=configs['TIME_INTEGRATION'],
            transform_theta=configs['TRANSFORM_THETA'],
            use_tanh=configs['USE_TANH']
        ).to(device=device)
        # if configs['ACTIVATION'] == "cpab_gnn":
        print("Num Parameters: ", count_parameters(model), flush=True)
            # count_parameters_without_cpab(model)
        loss_module = nn.NLLLoss()
        actual_backbone = configs['BACKBONE']
        configs['BACKBONE'] = "gcn2"
        optimizer = get_optimizer(model, **configs)
        configs['BACKBONE'] = actual_backbone
    # wandb.watch(model, log='all')

        best_val_acc = test_acc = 0
        for epoch in range(1, configs['MAX_EPOCHS'] + 1 + 1000):
            loss, thetas = train_one_epoch(model, optimizer, loss_module, data, flag=configs['DATASET_NAME'])
            regu_loss = 0.0
            if configs['SHARED_ACTIVATION']:
                if configs['ACTIVATION'] != "relu":
                    regu_loss = cov_regu(thetas, model.activation.T)
            else:
                if configs['ACTIVATION'] != "relu":
                    regu_loss = cov_regu(thetas, model.activation[0].T)
            loss += configs['REG_COEFF'] * regu_loss
            if configs['DATASET_NAME'] not in ['BlogCatalog', "Flickr"]:
                train_acc, val_acc, tmp_test_acc = test(model, data)
            else:
                train_acc, val_acc = test(model, data, flag="Flickr")
                tmp_test_acc = 0.0
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
            wandb.log({f"FOLD_{fold}/loss/train": loss, f"FOLD_{fold}/acc/train": train_acc, f"FOLD_{fold}/acc/val": val_acc, f"FOLD_{fold}/acc/test": test_acc})
            # print({"loss/train": loss, "acc/train": train_acc, "acc/val": val_acc, "acc/test": test_acc})

        val_accs.append(best_val_acc)
        test_accs.append(test_acc)

    if configs['TASK'] == 'train':
        run.finish()
    return {"val_acc": val_accs, "test_acc": test_accs}


def train_wrapper():
    run = wandb.init()
    print("wandb config", wandb.config)
    configs = wandb.config

    train_node_classifier(**configs)


def train_gine_wrapper():
    run = wandb.init()
    print("wandb config", wandb.config)
    configs = wandb.config

    train_gine(**configs)


@torch.no_grad()
def evaluate_obgb(model, loader, evaluator, device):
    model.eval()
    with torch.no_grad():
        y_true = []
        y_pred = []

        for idx, data in enumerate(loader):
            data = data.to(device)
            out, _ = model(data.x, data.edge_index, data.edge_attr, data.batch)

            y_true.append(data.y.view(out.shape).detach().cpu())
            y_pred.append(out.detach().cpu())

        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()

        input_dict = {"y_true": y_true, "y_pred": y_pred}
        if loader.dataset.task_type == 'regression':
            return evaluator.eval(input_dict)['rmse']
        else:
            return evaluator.eval(input_dict)['rocauc']


@torch.no_grad()
def test_gine(model, loader, device):
    model.eval()
    with torch.no_grad():
        total_error = 0
        for data in loader:
            data = data.to(device)
            out, thetas = model(data.x, data.edge_index, data.edge_attr, data.batch)
            total_error += (out.squeeze() - data.y).abs().sum().item()


            ## just for plotting:
            if False:
                import matplotlib.pyplot as plt
                import networkx as nx
                from torch_geometric.utils import from_networkx, to_networkx
                import difw
                T = difw.Cpab(tess_size=16, backend="numpy", device="cpu", zero_boundary=True, basis="qr")
                grid = T.uniform_meshgrid(100)
                grid_t = T.transform_grid(grid, thetas[0][:, :].detach().cpu().numpy())
                plt.figure()
                plt.plot(grid_t.squeeze().T)
                plt.show()
    return total_error / len(loader.dataset)


def train_gine(**configs):
    if configs['TASK'] == 'train':
        wandb.login()
        wandb.finish()
        datetime_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run = wandb.init(project=configs['PROJECT_NAME'], name=f"{configs['ACTIVATION']}_{datetime_stamp}",
                         tags=configs['TAGS'])
        wandb.config.update({k: v for k, v in configs.items() if k not in ["wandb_sweep"]})

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print("Device: ", device)
    print("CUDA CHECK: ", torch.cuda.is_available())

    seed_everything(configs['SEED'])

    if configs['DATASET_NAME'] == "OGB" or 'ogbg' in configs['DATASET_NAME']:
        dataset = get_data(configs['DATASET_NAME'], configs['BACKBONE'])

    model = GINENetv2(
        emb_dim=configs['HIDDEN_DIM'],
        num_layers=configs['NUM_LAYERS'],
        graph_pooling=configs['GRAPH_POOLING'],
        activation=configs['ACTIVATION'],
        shared_activation=configs['SHARED_ACTIVATION'],
        theta_hidden_dim=configs['THETA_HIDDEN_DIM'],
        theta_num_layers=configs['THETA_NUM_LAYERS'],
        theta_pooling=configs['THETA_POOLING'],
        radius=configs['RADIUS'],
        transform_theta=configs['TRANSFORM_THETA'],
        time_integration=configs['TIME_INTEGRATION'],
        tess_size=configs['TESS_SIZE'],
        use_bond_encoder=True if (configs['DATASET_NAME'] == "OGB" or 'ogbg' in configs['DATASET_NAME']) else False,
        num_tasks=1 if 'zinc' in configs['DATASET_NAME'].lower() else dataset.num_tasks,
        dropout=configs['DROPOUT'],
        use_tanh=configs['USE_TANH'],
        channel_wise=False,
        backbone=configs['BACKBONE']
    ).to(device=device)

    # if configs['ACTIVATION'] == "cpab_gnn":
    print("Num Parameters: ", count_parameters(model), flush=True)
    # count_parameters_without_cpab(model)

    if configs['DATASET_NAME'] == "ZINC":
        train_dataset, val_dataset, test_dataset = get_data(configs['DATASET_NAME'], configs['BACKBONE'])
        train_loader = DataLoader(train_dataset, batch_size=configs['BATCH_SIZE'], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=configs['BATCH_SIZE'], num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=configs['BATCH_SIZE'], num_workers=0 )
        evaluator = None
        criterion = nn.L1Loss()
        configs['BACKBONE'] = 'gine'
        optimizer = get_optimizer(model, **configs)
        # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=configs['SCHEDULER_STEP_SIZE'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                         patience=configs['SCHEDULER_STEP_SIZE'], min_lr=0.0000001)

    elif configs['DATASET_NAME'] == "OGB" or 'ogbg' in configs['DATASET_NAME']:
        dataset = get_data(configs['DATASET_NAME'], configs['BACKBONE'])
        split_idx = dataset.get_idx_split()
        train_loader = DataLoader(dataset[split_idx['train']], batch_size=configs['BATCH_SIZE'], shuffle=True,
                                  num_workers=0)
        val_loader = DataLoader(dataset[split_idx['valid']], batch_size=configs['BATCH_SIZE'], num_workers=0)
        test_loader = DataLoader(dataset[split_idx['test']], batch_size=configs['BATCH_SIZE'], num_workers=0)
        if configs['DATASET_NAME'] == "OGB":
            evaluate_text = 'ogbg-molhiv'
        else:
            evaluate_text = configs['DATASET_NAME']
        evaluator = Evaluator(name=evaluate_text)  # 'ogbg-molhiv'

        criterion = nn.BCEWithLogitsLoss()
        if 'esol' in configs['DATASET_NAME']:
            criterion = nn.MSELoss()
        # criterion = torch.nn.BCELoss()

        configs['BACKBONE'] = "gine"
        optimizer = get_optimizer(model, **configs)
        # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=configs['SCHEDULER_STEP_SIZE'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                         patience=configs['SCHEDULER_STEP_SIZE'], min_lr=0.0000001)
        if 'esol' in configs['DATASET_NAME']:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                             patience=configs['SCHEDULER_STEP_SIZE'], min_lr=0.0000001)

    # wandb.watch(model)

    best_val_mae = test_mae = 1e9
    best_val_rocauc = test_rocauc = 0
    if 'esol' in configs['DATASET_NAME']:
        best_val_rocauc = test_rocauc = 10000

    times = []

    total_loss = 0
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    batch_count = 0
    elapsed_times = []
    for epoch in range(1, configs['MAX_EPOCHS'] + 1):
        total_loss = 0
        if configs['TRACK_RUNNING_STATS']:
            model.train()
        else:
            model.eval()
        start = time.time()
        for idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            # # if batch_count > 200 and batch_count < 300:
            # start_event.record()
            out, thetas = model(data.x, data.edge_index, data.edge_attr, data.batch)
            if configs['USE_REGULARIZATION']:
                if configs['SHARED_ACTIVATION']:
                    regu_loss = cov_regu(thetas, model.activation.T)
                else:
                    regu_loss = cov_regu(thetas, model.activation[0].T)
            else:
                regu_loss = 0.0
            if configs['DATASET_NAME'] == "ZINC":
                loss = criterion(out.squeeze(), data.y.squeeze()) + configs['REG_COEFF'] * regu_loss
            elif configs['DATASET_NAME'] == "OGB" or 'ogbg' in configs['DATASET_NAME']:

                is_labeled = data.y == data.y
                loss = criterion(out.to(torch.float32)[is_labeled], data.y.to(torch.float32)[is_labeled])
                loss = loss + configs['REG_COEFF'] * regu_loss
            # end_event.record()
            # torch.cuda.synchronize()
            # elapsed_times.append(start_event.elapsed_time(end_event))
            # wandb.log({"elapsed_time": start_event.elapsed_time(end_event)})
            loss.backward()
            # if batch_count > 200 and batch_count < 300:
            # end_event.record()
            # torch.cuda.synchronize()
            # elapsed_times.append(start_event.elapsed_time(end_event))
            # wandb.log({"elapsed_time": start_event.elapsed_time(end_event)})
            # print({"elapsed_time": start_event.elapsed_time(end_event)})
            optimizer.step()
            total_loss += loss.item() * data.num_graphs

        if evaluator is not None:
            train_rocauc = evaluate_obgb(model, train_loader, evaluator, device)
            val_rocauc = evaluate_obgb(model, val_loader, evaluator, device)
            tmp_test_rocauc = evaluate_obgb(model, test_loader, evaluator, device)
            if 'esol' in configs['DATASET_NAME']:
                # in this case we measure MSE so we want to minimize.
                if val_rocauc < best_val_rocauc:
                    best_val_rocauc = val_rocauc
                    test_rocauc = tmp_test_rocauc
            else:
                if val_rocauc > best_val_rocauc:
                    best_val_rocauc = val_rocauc
                    test_rocauc = tmp_test_rocauc
            scheduler.step(best_val_rocauc)
            # print(f"Epoch {epoch}: Loss: {total_loss / len(train_loader.dataset)}, rocauc/train: {train_rocauc}, rocauc/val: {val_rocauc}, rocauc/test: {test_rocauc}, rocauc/tmp_test: {tmp_test_rocauc}")
            wandb.log({f"SEED_{configs['SEED']}/loss/train": total_loss / len(train_loader.dataset),
                       f"SEED_{configs['SEED']}/rocauc/train": train_rocauc,
                       f"SEED_{configs['SEED']}/rocauc/val": val_rocauc,
                       f"SEED_{configs['SEED']}/rocauc/test": test_rocauc,
                       f"SEED_{configs['SEED']}/rocauc/tmp_test": tmp_test_rocauc})
        else:
            val_mae = test_gine(model, val_loader, device)
            tmp_test_mae = test_gine(model, test_loader, device)
        
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                test_mae = tmp_test_mae

            scheduler.step(best_val_mae)
            # print(f"Epoch {epoch}: Loss: {total_loss / len(train_loader.dataset)} mae/val: {val_mae}, mae/test: {test_mae}")
            wandb.log({f"SEED_{configs['SEED']}/loss/train": total_loss / len(train_loader.dataset),
                       f"SEED_{configs['SEED']}/mae/val": val_mae, f"SEED_{configs['SEED']}/mae/test": test_mae,
                       f"SEED_{configs['SEED']}/mae/tmp_test": tmp_test_mae})

        times.append(time.time() - start)

    print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')
    # print(f'Mean Elapsed Time: {torch.tensor(elapsed_times)[100:150].mean()}ms')
    # print(f'Std Elapsed Time: {torch.tensor(elapsed_times)[100:150].std()}ms')

    if configs['TASK'] == 'train':
        run.finish()
    return {f"best_val_mae": best_val_mae, "test_mae": test_mae, "best_val_rocauc": best_val_rocauc,
            "test_rocauc": test_rocauc}


@torch.no_grad()
def test_tu(model, loader, device):
    model.eval()

    total_correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)


def train_tu(**configs):
    if configs['TASK'] == 'train':
        wandb.login()
        wandb.finish()
        datetime_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run = wandb.init(project=configs['PROJECT_NAME'], name=f"{configs['ACTIVATION']}_{datetime_stamp}",
                         tags=configs['TAGS'])
        wandb.config.update({k: v for k, v in configs.items() if k not in ["wandb_sweep"]})

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    seed_everything(configs['SEED'])

    dataset = get_data(configs['DATASET_NAME'], backbone="gine")

    folds = 10
    val_matrix = torch.zeros(size=(folds, configs['MAX_EPOCHS']))

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=configs['SEED'])
    labels = np.zeros(len(dataset))
    idx_list = []
    for idx in skf.split(labels, labels):
        idx_list.append(idx)

    for fold in range(folds):
        train_idx, val_idx = idx_list[fold]
        train_loader = DataLoader(dataset[torch.tensor(train_idx)], configs['BATCH_SIZE'], shuffle=True)
        val_loader = DataLoader(dataset[torch.tensor(val_idx)], configs['BATCH_SIZE'])
        model = GINNet(
            in_dim=dataset.num_features,
            emb_dim=configs['HIDDEN_DIM'],
            out_dim=dataset.num_classes,
            num_layers=configs['NUM_LAYERS'],
            graph_pooling=configs['GRAPH_POOLING'],
            activation=configs['ACTIVATION'],
            shared_activation=configs['SHARED_ACTIVATION'],
            theta_hidden_dim=configs['THETA_HIDDEN_DIM'],
            theta_num_layers=configs['THETA_NUM_LAYERS'],
            theta_pooling=configs['THETA_POOLING'],
            radius=configs['RADIUS'],
            transform_theta=configs['TRANSFORM_THETA'],
            time_integration=configs['TIME_INTEGRATION'],
            tess_size=configs['TESS_SIZE'],
            use_bond_encoder=True if (configs['DATASET_NAME'] == "OGB" or 'ogbg' in configs['DATASET_NAME']) else False,
            dropout=configs['DROPOUT'],
            channel_wise=False,
            use_tanh=configs['USE_TANH']
        ).to(device=device)

        optimizer = get_optimizer(model, **configs)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                         patience=configs['SCHEDULER_STEP_SIZE'], min_lr=0.0000001)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=configs['SCHEDULER_STEP_SIZE'])
        model.train()
        b_val_acc = 0
        for epoch in range(configs['MAX_EPOCHS']):
            model.train()
            epoch_loss = 0.0
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data.x, data.edge_index, data.batch)
                loss = F.cross_entropy(out, data.y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss * data.num_graphs
            epoch_loss = epoch_loss / len(train_loader.dataset)
            train_acc = test_tu(model, train_loader, device)
            val_acc = test_tu(model, val_loader, device)

            if val_acc > b_val_acc:
                b_val_acc = val_acc
            scheduler.step(b_val_acc)

            val_matrix[fold, epoch] = val_acc
            if val_acc > b_val_acc:
                b_val_acc = val_acc
            scheduler.step(b_val_acc)

            wandb.log({f"fold_{fold}/loss/train": epoch_loss, f"fold_{fold}/acc/train": train_acc,
                       f"fold_{fold}/acc/val": val_acc})
            # print(f"Fold: {fold + 1} - Epoch: {epoch+1} - Loss: {epoch_loss} - Train: {train_acc} - Val: {val_acc}")

    best_val_acc = torch.max(torch.mean(val_matrix, dim=0))

    best_epoch_idx = torch.argmax(torch.mean(val_matrix, dim=0, keepdim=True))

    val_std = val_matrix[:, best_epoch_idx].std()

    wandb.log({"Best Val Acc": best_val_acc, "Best Epoch Stddev": val_std})
    print(f"Best Val Acc: {best_val_acc} -  Best Epoch Stddev: {val_std}")

    if configs['TASK'] == 'train':
        run.finish()
    return {"best_val_acc": best_val_acc, "val_std": val_std}

    run.finish()
    return {"val_acc": best_val_mae, "test_acc": test_mae}


@torch.no_grad()
def test_link_predictor(model, predictor, data, split_edge, evaluator, batch_size):
    model.eval()
    predictor.eval()
    batch = torch.zeros(data.num_nodes).to(device=data.x.device).long()
    h,_ = model(data.x, data.adj_t, data.edge_weight, batch)

    pos_train_edge = split_edge['train']['edge'].to(h.device)
    pos_valid_edge = split_edge['valid']['edge'].to(h.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(h.device)
    pos_test_edge = split_edge['test']['edge'].to(h.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(h.device)

    pos_train_preds = []
    for perm in TDL(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in TDL(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in TDL(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    h,_ = model(data.x, data.full_adj_t, data.edge_weight, batch)

    pos_test_preds = []
    for perm in TDL(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in TDL(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    
    for K in [10, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results


def train_link_predictor(**configs):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    seed_everything(configs['SEED'])
    dataset = get_data(configs['DATASET_NAME'], configs['BACKBONE'])
    data = dataset[0]
    
    edge_index = data.edge_index
    # data.edge_weight = data.edge_weight.view(-1).to(torch.float)
    data.edge_weight = data.edge_weight.to(torch.long)
    data = data.to(device)  
    adj_t = torch_geometric.utils.to_torch_coo_tensor(data.edge_index.to(device))
    # adj_t = torch_geometric.utils.to_torch_csr_tensor(data.edge_index)
    data.adj_t = adj_t
    # data = torch_geometric.transforms.ToSparseTensor()(data)

    split_edge = dataset.get_edge_split()

    # # Use training + validation edges for inference on test set.
    # if args.use_valedges_as_input:
    #     val_edge_index = split_edge['valid']['edge'].t()
    #     full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
    #     data.full_adj_t = SparseTensor.from_edge_index(full_edge_index).t()
    #     data.full_adj_t = data.full_adj_t.to_symmetric()
    # else:
    data.full_adj_t = data.adj_t

    data = data.to(device)

    model = LinkPredictorBackbone(
        in_channels=data.num_features,
        hidden_channels=configs['HIDDEN_DIM'],
        out_channels=configs['HIDDEN_DIM'],
        num_layers=configs['NUM_LAYERS'],
        dropout=configs['DROPOUT'],
        activation=configs['ACTIVATION'],
        theta_hidden_dim=configs['THETA_HIDDEN_DIM'],
        theta_num_layers=configs['THETA_NUM_LAYERS'],
        theta_pooling=configs['THETA_POOLING'],
        radius=configs['RADIUS'],
        accelerator=configs['ACCELERATOR'],
        tess_size=configs['TESS_SIZE'],
        use_tanh=configs['USE_TANH'],
        transform_theta=configs['TRANSFORM_THETA']
    ).to(device=device)

    print("Num Parameters: ", count_parameters(model), flush=True)

    predictor = LinkPredictor(in_channels=configs['HIDDEN_DIM'], hidden_channels=configs['HIDDEN_DIM']).to(device=device)
    evaluator = LinkEvaluator(name=configs['DATASET_NAME'])
    optimizer = get_optimizer(model, **configs)
    optimizer.add_param_group({'params': predictor.parameters(), 'lr': configs['CONV_LR'], 'weight_decay':configs['CONV_WD']})
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                         patience=configs['SCHEDULER_STEP_SIZE'], min_lr=0.0000001)

    best_hits_at_10 = 0
    best_hits_at_50 = 0
    best_hits_at_100 = 0
    test_hits_at_10 = 0
    test_hits_at_50 = 0
    test_hits_at_100 = 0
    for epoch in range(1, configs['MAX_EPOCHS'] + 1):
        model.train()
        predictor.train()

        pos_train_edge = split_edge['train']['edge'].to(data.x.device)

        total_loss = total_examples = 0
        for perm in TDL(range(pos_train_edge.size(0)), configs['BATCH_SIZE'] * 1024,
                            shuffle=True):

            optimizer.zero_grad()
            batch = torch.zeros(data.num_nodes).to(device=device).long()
            h, thetas = model(data.x, data.adj_t, data.edge_weight, batch)
            # print(epoch, h.shape, flush=True)

            if configs['USE_REGULARIZATION']:
                if configs['SHARED_ACTIVATION']:
                    regu_loss = cov_regu(thetas, model.activation.T)
                else:
                    regu_loss = cov_regu(thetas, model.activation[0].T)
            else:
                regu_loss = 0.0

            edge = pos_train_edge[perm].t()
            pos_out = predictor(h[edge[0]], h[edge[1]])
            pos_loss = -torch.log(pos_out + 1e-15).mean()

            # Just do some trivial random sampling.
            edge = torch.randint(0, data.num_nodes, edge.size(), dtype=torch.long,
                                device=h.device)

            neg_out = predictor(h[edge[0]], h[edge[1]])
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

            loss = pos_loss + neg_loss + configs['REG_COEFF'] * regu_loss
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

            optimizer.step()

            num_examples = pos_out.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples
        
        epoch_loss = total_loss / total_examples

        results = test_link_predictor(model, predictor, data, split_edge, evaluator, configs['BATCH_SIZE'] * 1024)

        val_hits_at_10 = results['Hits@10'][1]
        tmp_test_hits_at_10 = results['Hits@10'][2]

        val_hits_at_50 = results['Hits@50'][1]
        tmp_test_hits_at_50 = results['Hits@50'][2]

        val_hits_at_100 = results['Hits@100'][1]
        tmp_test_hits_at_100 = results['Hits@100'][2]

        if (val_hits_at_50 > best_hits_at_50):
            best_hits_at_10 = val_hits_at_10
            best_hits_at_50 = val_hits_at_50
            best_hits_at_100 = val_hits_at_100

            test_hits_at_10 = tmp_test_hits_at_10
            test_hits_at_50 = tmp_test_hits_at_50
            test_hits_at_100 = tmp_test_hits_at_100
        
        scheduler.step(best_hits_at_50)


        wandb.log({
            f"SEED_{configs['SEED']}/loss/train": epoch_loss,
            f"SEED_{configs['SEED']}/Hits@10/train": results["Hits@10"][0],
            f"SEED_{configs['SEED']}/Hits@10/val": val_hits_at_10,
            f"SEED_{configs['SEED']}/Hits@10/tmp_test": tmp_test_hits_at_10,
            f"SEED_{configs['SEED']}/Hits@10/test": test_hits_at_10,
            f"SEED_{configs['SEED']}/Hits@50/train": results["Hits@50"][0],
            f"SEED_{configs['SEED']}/Hits@50/val":val_hits_at_50,
            f"SEED_{configs['SEED']}/Hits@50/tmp_test": tmp_test_hits_at_50,
            f"SEED_{configs['SEED']}/Hits@50/test": test_hits_at_50,
            f"SEED_{configs['SEED']}/Hits@100/train": results["Hits@100"][0],
            f"SEED_{configs['SEED']}/Hits@100/val": val_hits_at_100,
            f"SEED_{configs['SEED']}/Hits@100/tmp_test": tmp_test_hits_at_100,
            f"SEED_{configs['SEED']}/Hits@100/test": test_hits_at_100,
        })

    return {"best_val_hits_at_50": best_hits_at_50, "test_hits_at_50": test_hits_at_50}