from train import train_node_classifier, train_wrapper, train_gine, train_gine_wrapper, train_tu, train_link_predictor
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import numpy as np
from datetime import datetime
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ACCELERATOR', type=str, default='gpu', help='Type of accelerator (e.g., gpu or cpu)')
    parser.add_argument('--ACTIVATION', type=str, default='cpab_gnn', help='Activation function')
    parser.add_argument('--ALPHA_GCN2', type=int, default=0, help='Alpha value for GCN2')
    parser.add_argument('--BACKBONE', type=str, default='gcn2', help='Model backbone')
    parser.add_argument('--BATCH_SIZE', type=int, default=64, help='Batch size')
    parser.add_argument('--CHECKPOINT_PATH', type=str, default='saved_models/GNNs', help='Path to save checkpoints')
    parser.add_argument('--CONV_LR', type=float, default=0.05, help='Learning rate for convolution layers')
    parser.add_argument('--CONV_WD', type=float, default=1e-5, help='Weight decay for convolution layers')
    parser.add_argument('--DATASET_NAME', type=str, default='Cora', help='Name of the dataset')
    parser.add_argument('--DATASET_PATH', type=str, default='data/', help='Path to the dataset')
    parser.add_argument('--DROPOUT', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--HIDDEN_DIM', type=int, default=16, help='Dimension of hidden layers')
    parser.add_argument('--LINEAR_LR', type=float, default=0.0001, help='Learning rate for linear layers')
    parser.add_argument('--LINEAR_WD', type=float, default=0, help='Weight decay for linear layers')
    parser.add_argument('--MAX_EPOCHS', type=int, default=1000, help='Maximum number of training epochs')
    parser.add_argument('--MODEL_NAME', type=str, default='CPABGCN', help='Name of the model')
    parser.add_argument('--NUM_LAYERS', type=int, default=4, help='Number of layers in the model')
    parser.add_argument('--PROJECT_NAME', type=str, default='Diffeomorphic Graph Neural Networks', help='Project name')
    parser.add_argument('--RADIUS', type=float, default=8.930990839297044, help='Radius value')
    parser.add_argument('--REG_COEFF', type=float, default=0.1, help='Regularization coefficient')
    parser.add_argument('--SCHEDULER_STEP_SIZE', type=int, default=50, help='Step size for the scheduler')
    parser.add_argument('--SEED', type=int, default=1, help='Random seed')
    parser.add_argument('--SHARED_ACTIVATION', type=bool, default=True, help='Use shared activation')
    parser.add_argument('--TASK', type=str, default='sweep', help='Task type')
    parser.add_argument('--TESS_SIZE', type=int, default=1, help='Tessellation size')
    parser.add_argument('--THETA_GCN2', type=float, default=0, help='Theta GCN2 value')
    parser.add_argument('--THETA_HIDDEN_DIM', type=int, default=64, help='Theta hidden dimension')
    parser.add_argument('--THETA_LR', type=float, default=0.001, help='Learning rate for theta layers')
    parser.add_argument('--THETA_NUM_LAYERS', type=int, default=2, help='Number of theta layers')
    parser.add_argument('--THETA_POOLING', type=str, default='sum', help='Theta pooling method')
    parser.add_argument('--THETA_WD', type=float, default=0, help='Weight decay for theta layers')
    parser.add_argument('--TIME_INTEGRATION', type=int, default=0, help='Time integration steps')
    parser.add_argument('--TRACK_RUNNING_STATS', type=bool, default=True, help='Track running stats')
    parser.add_argument('--TRANSFORM_THETA', type=bool, default=True, help='Transform theta')
    parser.add_argument('--USE_REGULARIZATION', type=bool, default=True, help='Use regularization')
    parser.add_argument('--USE_TANH', type=bool, default=True, help='Use Tanh activation function')
    parser.add_argument('--GRAPH_POOLING', type=str, default="sum", help="Graph Pooling Operation")

    # Parse and return arguments
    return parser.parse_args()




@hydra.main(version_base=None, config_path='../conf', config_name='config')
def main(cfg: DictConfig) -> None:
    print(f"Configuration:\n{cfg}")
    if cfg.TASK == "train":
        print("Training...")
        if cfg.BACKBONE == "gine":
            train_gine(**cfg)
        elif cfg.DATASET_NAME in ["MUTAG", "PROTEINS", "PTC_MR", "NCI1", "NCI109"]:
            train_tu(**cfg)
        else:
            train_node_classifier(**cfg)
    
    elif cfg.TASK == "sweep":
        print("Sweep...")
        seeds = cfg.SEEDS
        val_metrics = []
        test_metrics = []
        wandb.login()
        wandb.finish()
        datetime_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run = wandb.init(project=cfg.PROJECT_NAME, name=f"{cfg.ACTIVATION}_{datetime_stamp}", tags=cfg.TAGS)
        config = OmegaConf.to_container(cfg, resolve=True)
        wandb.config.update({k: v for k, v in config.items() if k not in ["wandb_sweep"]})
        for seed in seeds:
            cfg.SEED = seed
            print(f"Seed: {cfg.SEED}")
            if cfg.DATASET_NAME == "ZINC":
                result_dict = train_gine(**cfg)
                val_metrics.append(result_dict['best_val_mae'])
                test_metrics.append(result_dict['test_mae'])
            elif cfg.DATASET_NAME == "OGB" or "ogbg" in cfg.DATASET_NAME:
                result_dict = train_gine(**cfg)
                val_metrics.append(result_dict['best_val_rocauc'])
                test_metrics.append(result_dict['test_rocauc'])
            elif cfg.DATASET_NAME == "ogbl-collab":
                result_dict = train_link_predictor(**cfg)
                val_metrics.append(result_dict['best_val_hits_at_50'])
                test_metrics.append(result_dict['test_hits_at_50'])
        
        if cfg.DATASET_NAME in ["MUTAG", "PROTEINS", "PTC_MR", "NCI1", "NCI109"]:
            result_dict = train_tu(**cfg)
            val_metrics.append(result_dict['best_val_acc'])
            test_metrics.append(result_dict['val_std'])
        elif cfg.DATASET_NAME in ["Cora", "CiteSeer", "PubMed"]:
            result_dict = train_node_classifier(**cfg)
            val_metrics = result_dict['val_acc']
            test_metrics = result_dict['test_acc']
        
        elif cfg.DATASET_NAME in ["Flickr", "BlogCatalog"]:
            result_dict = train_node_classifier(**cfg)
            val_metrics = result_dict['val_acc']
            test_metrics = result_dict['test_acc']
        mean_val_metric = np.mean(val_metrics)
        std_val_metric = np.std(val_metrics)

        mean_test_metric = np.mean(test_metrics)
        std_test_metric = np.mean(test_metrics)
        if not (cfg.DATASET_NAME in ["MUTAG", "PROTEINS", "PTC_MR", "NCI1", "NCI109"]):
            std_test_metric = np.std(test_metrics)

        for metric1, metric2 in zip(val_metrics, test_metrics):
            wandb.log({"metrics/val": metric1, "metrics/test": metric2})
        
        wandb.log({"metrics/val_mean": mean_val_metric, "metrics/val_std": std_val_metric, "metrics/test_mean": mean_test_metric, "metrics/test_std": std_test_metric})
        run.finish()


if __name__ == "__main__":
    args = parse_args()
    cfg = OmegaConf.create(args.__dict__)
    base_conf = OmegaConf.create({"SEEDS": [1, 2, 3, 4, 5], "TAGS": ['digraf']})
    cfg = OmegaConf.merge(base_conf, cfg)
    main(cfg)
