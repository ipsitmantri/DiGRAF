import argparse
from IPython import embed
from matplotlib.font_manager import weight_dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from baseline_layers import GIN, GINE, GCN2, GAT, GIN2, SAGE, SAGE2, GAT2
from data_utils import get_data
import wandb
import torch_geometric.nn as pyg
from torch_geometric.seed import seed_everything
from torch_geometric.loader import DataLoader
import torch.optim as optim
from ogb.graphproppred import Evaluator
from sklearn.model_selection import StratifiedKFold
import numpy as np

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def evaluate_obgb(model, loader, evaluator, device):
    model.eval()
    with torch.no_grad():
        y_true = []
        y_pred = []

        for idx, data in enumerate(loader):
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)

            y_true.append(data.y.view(out.shape).detach().cpu())
            y_pred.append(out.detach().cpu())

        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()

        input_dict = {"y_true": y_true, "y_pred": y_pred}
        if loader.dataset.task_type == "regression":
            return evaluator.eval(input_dict)['rmse']
        else:
            return evaluator.eval(input_dict)['rocauc']


def run_gine(args):
    seeds = [1, 2, 3, 4, 5]
    # seeds = [1]

    run = wandb.init(project="Graph Activation Functions")  # project="Graph Activation Functions", entity="eliasof", name="gine_" + args.activation

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(device)
    wandb.config.update({"seeds": seeds})
    wandb.config.update(args)

    if args.dataset == "ZINC":
        train_dataset, val_dataset, test_dataset = get_data(args.dataset, "gine")
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0)
        evaluator = None
        criterion = nn.L1Loss()

        num_tasks = 1
        use_bond_encoder = False

    elif "ogbg" in args.dataset:
        dataset = get_data(args.dataset, "gine")
        split_idx = dataset.get_idx_split()
        train_loader = DataLoader(dataset[split_idx['train']], batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(dataset[split_idx['valid']], batch_size=args.batch_size, num_workers=0)
        test_loader = DataLoader(dataset[split_idx['test']], batch_size=args.batch_size, num_workers=0)
        evaluator = Evaluator(name=args.dataset)  # 'ogbg-molhiv'
        criterion = nn.BCEWithLogitsLoss()
        if "olesol" in args.dataset:
            criterion = nn.MSELoss()
        num_tasks = dataset.num_tasks
        use_bond_encoder = True

    val_metrics = []
    test_metrics = []
    elapsed_times = []
    for seed in seeds:
        seed_everything(seed)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        model = GINE(emb_dim=args.hidden_channels,
                     num_tasks=num_tasks,
                     activation=args.activation,
                     num_layers=args.num_layers,
                     add_residual=args.add_residual,
                     use_bond_encoder=use_bond_encoder,
                     num_pieces=args.num_pieces).to(device=device)
        print(f"Number of parameters: {count_parameters(model)}")
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        mode = 'min'
        if "ZINC" in args.dataset:
            mode = 'min'
        elif "olesol" in args.dataset:
            mode = 'min'
        else:
            mode = "max"
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=0.5, patience=300,
                                                         min_lr=0.0000001)
        best_val_mae = test_mae = 1e9
        best_val_rocauc = test_rocauc = 0
        best_val_rmse = test_rmse = 1e9
        for epoch in range(args.num_epochs):
            total_loss = 0.0
            # model.train()
            model.eval()

            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                # start_event.record()
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)

                if args.dataset == "ZINC":
                    loss = criterion(out.squeeze(), data.y.squeeze())
                    loss.backward()
                    # end_event.record()
                    # torch.cuda.synchronize()
                    # wandb.log({"elapsed_time": start_event.elapsed_time(end_event)})
                    # elapsed_times.append(start_event.elapsed_time(end_event))
                    # print({"elapsed_time": start_event.elapsed_time(end_event)})
                    
                    optimizer.step()
                    total_loss += loss.item() * data.num_graphs

                else:
                    is_labeled = data.y == data.y
                    loss = criterion(out.to(torch.float32)[is_labeled], data.y.to(torch.float32)[is_labeled])
                    loss.backward()
                    # end_event.record()
                    # torch.cuda.synchronize()
                    # wandb.log({"elapsed_time": start_event.elapsed_time(end_event)})
                    # elapsed_times.append(start_event.elapsed_time(end_event))
                    # print({"elapsed_time": start_event.elapsed_time(end_event)})
                    optimizer.step()
                    total_loss += loss.item() * data.num_graphs

            if args.dataset == "ZINC":
                model.eval()
                val_mae = 0.0
                tmp_test_mae = 0.0
                with torch.no_grad():
                    for data in val_loader:
                        data = data.to(device)
                        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                        val_mae += (out.squeeze() - data.y.squeeze()).abs().sum().item()
                    val_mae = val_mae / len(val_loader.dataset)
                    for data in test_loader:
                        data = data.to(device)
                        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                        tmp_test_mae += (out.squeeze() - data.y.squeeze()).abs().sum().item()
                    tmp_test_mae = tmp_test_mae / len(test_loader.dataset)
                    if val_mae < best_val_mae:
                        best_val_mae = val_mae
                        test_mae = tmp_test_mae

                    scheduler.step(best_val_mae)
                    wandb.log({f"SEED_{seed}/loss/train": total_loss / len(train_loader.dataset),
                               f"SEED_{seed}/mae/val": val_mae, f"SEED_{seed}/mae/test": test_mae,
                               f"SEED_{seed}/mae/tmp_test": tmp_test_mae})
            else:
                model.eval()
                if "olesol" in args.dataset:
                    train_rmse = evaluate_obgb(model, train_loader, evaluator, device)
                    val_rmse = evaluate_obgb(model, val_loader, evaluator, device)
                    tmp_test_rmse = evaluate_obgb(model, test_loader, evaluator, device)

                    if val_rmse < best_val_rmse:
                        best_val_rmse = val_rmse
                        test_rmse = tmp_test_rmse
                    scheduler.step(best_val_rmse)
                    wandb.log({f"SEED_{seed}/loss/train": total_loss / len(train_loader.dataset),
                               f"SEED_{seed}/rmse/train": train_rmse, f"SEED_{seed}/rmse/val": val_rmse,
                               f"SEED_{seed}/rmse/test": test_rmse, f"SEED_{seed}/rmse/tmp_test": tmp_test_rmse})
                else:
                    train_rocauc = evaluate_obgb(model, train_loader, evaluator, device)
                    val_rocauc = evaluate_obgb(model, val_loader, evaluator, device)
                    tmp_test_rocauc = evaluate_obgb(model, test_loader, evaluator, device)

                    if val_rocauc > best_val_rocauc:
                        best_val_rocauc = val_rocauc
                        test_rocauc = tmp_test_rocauc
                    scheduler.step(best_val_rocauc)
                    wandb.log({f"SEED_{seed}/loss/train": total_loss / len(train_loader.dataset),
                            f"SEED_{seed}/rocauc/train": train_rocauc, f"SEED_{seed}/rocauc/val": val_rocauc,
                            f"SEED_{seed}/rocauc/test": test_rocauc, f"SEED_{seed}/rocauc/tmp_test": tmp_test_rocauc})

        if best_val_mae == 1e9 and best_val_rmse == 1e9: # monitoring rocauc
            val_metrics.append(best_val_rocauc)
            test_metrics.append(test_rocauc)
        elif best_val_mae == 1e9 and best_val_rocauc == 0: # monitoring rmse
            val_metrics.append(best_val_rmse)
            test_metrics.append(test_rmse)
        else: # monitoring mae
            val_metrics.append(val_mae)
            test_metrics.append(test_mae)

    val_mean = np.mean(val_metrics)
    val_std = np.std(val_metrics)
    test_mean = np.mean(test_metrics)
    test_std = np.std(test_metrics)

    # print(f'Mean Elapsed Time: {torch.tensor(elapsed_times)[100:].mean()}ms')
    # print(f'Std Elapsed Time: {torch.tensor(elapsed_times)[100:].std()}ms')

    wandb.log({"val_mean": val_mean, "val_std": val_std, "test_mean": test_mean, "test_std": test_std})
    run.finish()


def run_sage2(args):
    seeds = [1, 2, 3, 4, 5]
    # seeds = [1]

    run = wandb.init(project="Graph Activation Functions")  # project="Graph Activation Functions", entity="eliasof", name="gine_" + args.activation

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(device)
    wandb.config.update({"seeds": seeds})
    wandb.config.update(args)

    if args.dataset == "ZINC":
        train_dataset, val_dataset, test_dataset = get_data(args.dataset, "gine")
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0)
        evaluator = None
        criterion = nn.L1Loss()

        num_tasks = 1
        use_bond_encoder = False

    elif "ogbg" in args.dataset:
        dataset = get_data(args.dataset, "gine")
        split_idx = dataset.get_idx_split()
        train_loader = DataLoader(dataset[split_idx['train']], batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(dataset[split_idx['valid']], batch_size=args.batch_size, num_workers=0)
        test_loader = DataLoader(dataset[split_idx['test']], batch_size=args.batch_size, num_workers=0)
        evaluator = Evaluator(name=args.dataset)  # 'ogbg-molhiv'
        criterion = nn.BCEWithLogitsLoss()
        if "olesol" in args.dataset:
            criterion = nn.MSELoss()
        num_tasks = dataset.num_tasks
        use_bond_encoder = True

    val_metrics = []
    test_metrics = []
    elapsed_times = []
    for seed in seeds:
        seed_everything(seed)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        if args.backbone == "sage2":
            model = SAGE2(
                emb_dim=args.hidden_channels,
                num_tasks=num_tasks,
                num_layers=args.num_layers,
                use_bond_encoder=use_bond_encoder
            ).to(device)
        elif args.backbone == "gat2":
            model = GAT2(
                emb_dim=args.hidden_channels,
                num_tasks=num_tasks,
                num_layers=args.num_layers,
                use_bond_encoder=use_bond_encoder
            ).to(device)
        else:
            raise ValueError("Invalid backbone....!!")
        print(f"Number of parameters: {count_parameters(model)}")
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        mode = 'min'
        if "ZINC" in args.dataset:
            mode = 'min'
        elif "olesol" in args.dataset:
            mode = 'min'
        else:
            mode = "max"
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=0.5, patience=300,
                                                         min_lr=0.0000001)
        best_val_mae = test_mae = 1e9
        best_val_rocauc = test_rocauc = 0
        best_val_rmse = test_rmse = 1e9
        for epoch in range(args.num_epochs):
            total_loss = 0.0
            # model.train()
            model.eval()

            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                # start_event.record()
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)

                if args.dataset == "ZINC":
                    loss = criterion(out.squeeze(), data.y.squeeze())
                    loss.backward()
                    # end_event.record()
                    # torch.cuda.synchronize()
                    # wandb.log({"elapsed_time": start_event.elapsed_time(end_event)})
                    # elapsed_times.append(start_event.elapsed_time(end_event))
                    # print({"elapsed_time": start_event.elapsed_time(end_event)})
                    
                    optimizer.step()
                    total_loss += loss.item() * data.num_graphs

                else:
                    is_labeled = data.y == data.y
                    loss = criterion(out.to(torch.float32)[is_labeled], data.y.to(torch.float32)[is_labeled])
                    loss.backward()
                    # end_event.record()
                    # torch.cuda.synchronize()
                    # wandb.log({"elapsed_time": start_event.elapsed_time(end_event)})
                    # elapsed_times.append(start_event.elapsed_time(end_event))
                    # print({"elapsed_time": start_event.elapsed_time(end_event)})
                    optimizer.step()
                    total_loss += loss.item() * data.num_graphs

            if args.dataset == "ZINC":
                model.eval()
                val_mae = 0.0
                tmp_test_mae = 0.0
                with torch.no_grad():
                    for data in val_loader:
                        data = data.to(device)
                        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                        val_mae += (out.squeeze() - data.y.squeeze()).abs().sum().item()
                    val_mae = val_mae / len(val_loader.dataset)
                    for data in test_loader:
                        data = data.to(device)
                        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                        tmp_test_mae += (out.squeeze() - data.y.squeeze()).abs().sum().item()
                    tmp_test_mae = tmp_test_mae / len(test_loader.dataset)
                    if val_mae < best_val_mae:
                        best_val_mae = val_mae
                        test_mae = tmp_test_mae

                    scheduler.step(best_val_mae)
                    wandb.log({f"SEED_{seed}/loss/train": total_loss / len(train_loader.dataset),
                               f"SEED_{seed}/mae/val": val_mae, f"SEED_{seed}/mae/test": test_mae,
                               f"SEED_{seed}/mae/tmp_test": tmp_test_mae})
            else:
                model.eval()
                if "olesol" in args.dataset:
                    train_rmse = evaluate_obgb(model, train_loader, evaluator, device)
                    val_rmse = evaluate_obgb(model, val_loader, evaluator, device)
                    tmp_test_rmse = evaluate_obgb(model, test_loader, evaluator, device)

                    if val_rmse < best_val_rmse:
                        best_val_rmse = val_rmse
                        test_rmse = tmp_test_rmse
                    scheduler.step(best_val_rmse)
                    wandb.log({f"SEED_{seed}/loss/train": total_loss / len(train_loader.dataset),
                               f"SEED_{seed}/rmse/train": train_rmse, f"SEED_{seed}/rmse/val": val_rmse,
                               f"SEED_{seed}/rmse/test": test_rmse, f"SEED_{seed}/rmse/tmp_test": tmp_test_rmse})
                else:
                    train_rocauc = evaluate_obgb(model, train_loader, evaluator, device)
                    val_rocauc = evaluate_obgb(model, val_loader, evaluator, device)
                    tmp_test_rocauc = evaluate_obgb(model, test_loader, evaluator, device)

                    if val_rocauc > best_val_rocauc:
                        best_val_rocauc = val_rocauc
                        test_rocauc = tmp_test_rocauc
                    scheduler.step(best_val_rocauc)
                    wandb.log({f"SEED_{seed}/loss/train": total_loss / len(train_loader.dataset),
                            f"SEED_{seed}/rocauc/train": train_rocauc, f"SEED_{seed}/rocauc/val": val_rocauc,
                            f"SEED_{seed}/rocauc/test": test_rocauc, f"SEED_{seed}/rocauc/tmp_test": tmp_test_rocauc})

        if best_val_mae == 1e9 and best_val_rmse == 1e9: # monitoring rocauc
            val_metrics.append(best_val_rocauc)
            test_metrics.append(test_rocauc)
        elif best_val_mae == 1e9 and best_val_rocauc == 0: # monitoring rmse
            val_metrics.append(best_val_rmse)
            test_metrics.append(test_rmse)
        else: # monitoring mae
            val_metrics.append(val_mae)
            test_metrics.append(test_mae)

    val_mean = np.mean(val_metrics)
    val_std = np.std(val_metrics)
    test_mean = np.mean(test_metrics)
    test_std = np.std(test_metrics)

    # print(f'Mean Elapsed Time: {torch.tensor(elapsed_times)[100:].mean()}ms')
    # print(f'Std Elapsed Time: {torch.tensor(elapsed_times)[100:].std()}ms')

    wandb.log({"val_mean": val_mean, "val_std": val_std, "test_mean": test_mean, "test_std": test_std})
    run.finish()


@torch.no_grad()
def test_gin(model, loader, device):
    model.eval()

    total_correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)


def run_gin(args):
    seed = 1

    #run = wandb.init(project="Graph Activation Functions", entity="mmkipsit", name="gin_" + args.activation)
    run = wandb.init()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    wandb.config.update({"seed": seed})
    wandb.config.update(args)

    seed_everything(seed)

    dataset = get_data(args.dataset, backbone="gin")

    folds = 10
    val_matrix = torch.zeros(size=(folds, args.num_epochs))

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    labels = np.zeros(len(dataset))
    idx_list = []
    for idx in skf.split(labels, labels):
        idx_list.append(idx)

    for fold in range(folds):
        train_idx, val_idx = idx_list[fold]
        train_loader = DataLoader(dataset[torch.tensor(train_idx)], args.batch_size, shuffle=True)
        val_loader = DataLoader(dataset[torch.tensor(val_idx)], args.batch_size)
        model = GIN(
            in_dim=dataset.num_features,
            emb_dim=args.hidden_channels,
            out_dim=dataset.num_classes,
            num_layers=args.num_layers,
            activation=args.activation,
            add_residual=args.add_residual,
            num_pieces=args.num_pieces
        ).to(device=device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=300,
                                                         min_lr=0.0000001)
        model.train()
        b_val_acc = 0.0
        for epoch in range(args.num_epochs):
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
            train_acc = test_gin(model, train_loader, device)
            val_acc = test_gin(model, val_loader, device)

            val_matrix[fold, epoch] = val_acc
            if val_acc > b_val_acc:
                b_val_acc = val_acc
            scheduler.step(b_val_acc)

            wandb.log({f"fold_{fold}/loss/train": epoch_loss, f"fold_{fold}/acc/train": train_acc,
                       f"fold_{fold}/acc/val": val_acc})

    best_val_acc = torch.max(torch.mean(val_matrix, dim=0))

    best_epoch_idx = torch.argmax(torch.mean(val_matrix, dim=0, keepdim=True))

    val_std = val_matrix[:, best_epoch_idx].std()

    wandb.log({"Best Val Acc": best_val_acc, "Best Epoch Stddev": val_std})
    print(f"Best Val Acc: {best_val_acc} -  Best Epoch Stddev: {val_std}")

    run.finish()


@torch.no_grad()
def test_gcn(model, data, flag="Flickr"):
    model.eval()
    accs = []
    x, edge_index = data.x, data.edge_index
    batch = torch.ones_like(x).to(x.device)
    out = model(x, edge_index)
    pred = out.argmax(dim=-1)
    if flag != "Flickr":
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    else:
        for mask in [data.train_mask, data.test_mask]:
            accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    # for mask in [data.train_mask, data.val_mask, data.test_mask]:
    #     accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))

    return accs


def run_gcn(args):
    run = wandb.init(project="Graph Activation Functions", entity="mmkipsit", name="gcn_" + args.activation)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    wandb.config.update(args)

    val_metrics = []
    test_metrics = []
    folds = 10 
    seed_everything(args.seed)
    for fold in range(folds):
        node_dataset = get_data(args.dataset, backbone="gcn2")
        data = node_dataset[0].to(device)
        model = GCN2(
            in_channels=node_dataset.num_node_features,
            hidden_channels=args.hidden_channels,
            out_channels=node_dataset.num_classes,
            num_layers=args.num_layers,
            alpha=args.alpha_gcn2,
            theta=args.theta_gcn2,
            dropout=args.dropout,
            activation=args.activation,
            num_pieces=args.num_pieces,
            # alpha_appnp=args.alpha_appnp,
            # add_residual=args.add_residual
        ).to(device)
        print(f"Number of parameters: {count_parameters(model)}")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=300,
                                                         min_lr=0.0000001)

        best_val_acc = test_acc = 0
        for epoch in range(args.num_epochs):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if args.dataset not in ['BlogCatalog', "Flickr"]:
                train_acc, val_acc, tmp_test_acc = test_gcn(model, data, flag=args.dataset)
            else:
                train_acc, val_acc = test_gcn(model, data, flag="Flickr")
                tmp_test_acc = 0.0
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
            scheduler.step(val_acc)
            wandb.log(
                {f"FOLD_{fold}/loss/train": loss, f"FOLD_{fold}/acc/train": train_acc, f"FOLD_{fold}/acc/val": val_acc,
                 f"FOLD_{fold}/acc/test": test_acc})

        val_metrics.append(best_val_acc)
        test_metrics.append(test_acc)

    val_mean = np.mean(val_metrics)
    val_std = np.std(val_metrics)
    test_mean = np.mean(test_metrics)
    test_std = np.std(test_metrics)

    wandb.log({"val_mean": val_mean, "val_std": val_std, "test_mean": test_mean, "test_std": test_std})
    run.finish()


def run_gat(args):
    run = wandb.init(project="Graph Activation Functions", entity="mmkipsit", name="gat_" + args.activation)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    wandb.config.update(args)

    val_metrics = []
    test_metrics = []
    folds = 10 
    seed_everything(args.seed)
    for fold in range(folds):
        node_dataset = get_data(args.dataset, backbone="gat")
        data = node_dataset[0].to(device)
        model = GAT(
            in_channels=node_dataset.num_node_features,
            hidden_channels=args.hidden_channels,
            out_channels=node_dataset.num_classes,
        ).to(device)
        print(f"Number of parameters: {count_parameters(model)}")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=300,
                                                         min_lr=0.0000001)

        best_val_acc = test_acc = 0
        for epoch in range(args.num_epochs):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if args.dataset not in ['BlogCatalog', "Flickr"]:
                train_acc, val_acc, tmp_test_acc = test_gcn(model, data, flag=args.dataset)
            else:
                train_acc, val_acc = test_gcn(model, data, flag="Flickr")
                tmp_test_acc = 0.0
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
            scheduler.step(val_acc)
            wandb.log(
                {f"FOLD_{fold}/loss/train": loss, f"FOLD_{fold}/acc/train": train_acc, f"FOLD_{fold}/acc/val": val_acc,
                 f"FOLD_{fold}/acc/test": test_acc})

        val_metrics.append(best_val_acc)
        test_metrics.append(test_acc)

    val_mean = np.mean(val_metrics)
    val_std = np.std(val_metrics)
    test_mean = np.mean(test_metrics)
    test_std = np.std(test_metrics)

    wandb.log({"val_mean": val_mean, "val_std": val_std, "test_mean": test_mean, "test_std": test_std})
    run.finish()

def run_gin2(args):
    run = wandb.init(project="Graph Activation Functions", entity="mmkipsit", name="gin_" + args.activation)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    wandb.config.update(args)

    val_metrics = []
    test_metrics = []
    folds = 10 
    seed_everything(args.seed)
    for fold in range(folds):
        node_dataset = get_data(args.dataset, backbone="gin")
        data = node_dataset[0].to(device)
        print(data.x.shape)
        model = GIN2(
            in_channels=node_dataset.num_node_features,
            hidden_channels=args.hidden_channels,
            out_channels=node_dataset.num_classes,
            num_layers=args.num_layers
        ).to(device)
        print(f"Number of parameters: {count_parameters(model)}")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=300,
                                                         min_lr=0.0000001)

        best_val_acc = test_acc = 0
        for epoch in range(args.num_epochs):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if args.dataset not in ['BlogCatalog', "Flickr"]:
                train_acc, val_acc, tmp_test_acc = test_gcn(model, data, flag=args.dataset)
            else:
                train_acc, val_acc = test_gcn(model, data, flag="Flickr")
                tmp_test_acc = 0.0
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
            scheduler.step(val_acc)
            wandb.log(
                {f"FOLD_{fold}/loss/train": loss, f"FOLD_{fold}/acc/train": train_acc, f"FOLD_{fold}/acc/val": val_acc,
                 f"FOLD_{fold}/acc/test": test_acc})

        val_metrics.append(best_val_acc)
        test_metrics.append(test_acc)

    val_mean = np.mean(val_metrics)
    val_std = np.std(val_metrics)
    test_mean = np.mean(test_metrics)
    test_std = np.std(test_metrics)

    wandb.log({"val_mean": val_mean, "val_std": val_std, "test_mean": test_mean, "test_std": test_std})
    run.finish()


def run_sage(args):
    run = wandb.init(project="Graph Activation Functions", entity="mmkipsit", name="sage_" + args.activation)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    wandb.config.update(args)

    val_metrics = []
    test_metrics = []
    folds = 10 
    seed_everything(args.seed)
    for fold in range(folds):
        node_dataset = get_data(args.dataset, backbone="sage")
        data = node_dataset[0].to(device)
        model = SAGE(
            in_channels=node_dataset.num_node_features,
            hidden_channels=args.hidden_channels,
            out_channels=node_dataset.num_classes,
            num_layers=args.num_layers,
        ).to(device)
        print(f"Number of parameters: {count_parameters(model)}")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=300,
                                                         min_lr=0.0000001)

        best_val_acc = test_acc = 0
        for epoch in range(args.num_epochs):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if args.dataset not in ['BlogCatalog', "Flickr"]:
                train_acc, val_acc, tmp_test_acc = test_gcn(model, data, flag=args.dataset)
            else:
                train_acc, val_acc = test_gcn(model, data, flag="Flickr")
                tmp_test_acc = 0.0
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
            scheduler.step(val_acc)
            wandb.log(
                {f"FOLD_{fold}/loss/train": loss, f"FOLD_{fold}/acc/train": train_acc, f"FOLD_{fold}/acc/val": val_acc,
                 f"FOLD_{fold}/acc/test": test_acc})

        val_metrics.append(best_val_acc)
        test_metrics.append(test_acc)

    val_mean = np.mean(val_metrics)
    val_std = np.std(val_metrics)
    test_mean = np.mean(test_metrics)
    test_std = np.std(test_metrics)

    wandb.log({"val_mean": val_mean, "val_std": val_std, "test_mean": test_mean, "test_std": test_std})
    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--activation", type=str, default="relu", help="Choose from ['relu', 'gelu', 'tanh', 'leaky_relu', 'prelu', 'maxout', 'max', 'median', 'swish', 'grelu', 'identity', 'elu', 'sigmoid']")
    parser.add_argument("--backbone", type=str, default="gine", help="choose from ['gine', 'gcn2', 'gin', 'gat', 'gin2']")
    parser.add_argument('--dataset', type=str, default='ZINC')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_channels', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=7)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--add_residual', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=1, help="random seed")
    parser.add_argument('--alpha_gcn2', type=float, default=0.0)
    parser.add_argument('--theta_gcn2', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--num_pieces', type=int, default=2, help="Also use this for graph adaptive max and median")
    parser.add_argument('--alpha_appnp', type=float, default=0.1, help="Also use this for grelu")

    args = parser.parse_args()

    if args.backbone == "gine":
        run_gine(args)
    elif args.backbone == "gcn2":
        run_gcn(args)
    elif args.backbone == "gin":
        run_gin(args)
    elif args.backbone == "gat":
        run_gat(args)
    elif args.backbone == "gin2":
        run_gin2(args)
    elif args.backbone == "sage":
        run_sage(args)
    elif args.backbone == "sage2":
        run_sage2(args)
    elif args.backbone == "gat2":
        run_sage2(args)
    else:
        raise ValueError("invalid backbone")
