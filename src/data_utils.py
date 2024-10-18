import torch
from torch_geometric.datasets import Planetoid, TUDataset, ZINC, AttributedGraphDataset, Flickr
import torch_geometric.transforms as T
from torch_geometric.transforms import BaseTransform
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset

class CustomNormalize(BaseTransform):
    def __init__(self):
        super(CustomNormalize, self).__init__()
        # self.normalize = T.NormalizeFeatures(attrs=['x'])
        self.norm = T.GCNNorm()
    def __call__(self, data):
        # print(data)
        # edge_attr = data.edge_attr.clone()
        # # data = self.normalize(data)
        data = self.norm(data)
        # data.edge_attr = edge_attr
        return data

class SparseToDense(BaseTransform):
    def __init__(self):
        super(SparseToDense, self).__init__()
    
    def __call__(self, data):
        x_dense = data.x.to_dense()
        data.x = x_dense.to(dtype=torch.float32)
        return data

def get_data(dataset_name, backbone="gcn2"):
    if backbone == "gcn2":
        transform = T.Compose([T.NormalizeFeatures(), T.GCNNorm()])
    elif backbone == "gcn":
        transform = T.Compose([T.NormalizeFeatures()])
    else:
        transform = T.Compose([T.NormalizeFeatures()])
    if dataset_name == "Cora":
        dataset = Planetoid(root='./data/Cora', name='Cora', transform=transform, split="random")
    elif dataset_name == "CiteSeer":
        dataset = Planetoid(root='./data/CiteSeer', name='CiteSeer', transform=transform, split="random")
    elif dataset_name == "PubMed":
        print('Using PubMed dataset')
        dataset = Planetoid(root='./data/PubMed', name='PubMed', transform=transform, split="random")
    elif dataset_name in ["MUTAG", "PROTEINS", "PTC_MR", "NCI1", "NCI109"]:
        dataset = TUDataset(root='./data/'+dataset_name, name=dataset_name)
    elif dataset_name == "ZINC":
        transform = CustomNormalize() if backbone == "gcn22" else None
        train_dataset = ZINC(root='./data/ZINC', subset=True, split='train', transform=transform)
        val_dataset = ZINC(root='./data/ZINC', subset=True, split='val', transform=transform)
        test_dataset = ZINC(root='./data/ZINC', subset=True, split='test', transform=transform) 
        return train_dataset, val_dataset, test_dataset
    elif dataset_name == "OGB":
        dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root='./data/OGB')
        return dataset
    elif 'ogbg' in dataset_name:
        transform = None
        dataset = PygGraphPropPredDataset(name=dataset_name, root='./data/'+dataset_name, transform=transform)
        return dataset
    # elif dataset_name == "Flickr":
    #     transform = T.Compose([T.RandomNodeSplit(split='train_rest', num_test=0.8), T.NormalizeFeatures(), T.GCNNorm()])
    #     dataset = Flickr(root="./data/" + dataset_name, transform=transform)
    elif "ogbl" in dataset_name:
        dataset = PygLinkPropPredDataset(name=dataset_name, root='./data/'+dataset_name)
        return dataset
    elif dataset_name in ['BlogCatalog', "Flickr"]:
        transform = T.Compose([T.GCNNorm(), T.RandomNodeSplit(split='train_rest', num_test=0.8, num_val=0.0), SparseToDense()])
        dataset = AttributedGraphDataset(root="./data/" + dataset_name, name=dataset_name, transform=transform)
    else:
        assert False, "Unknown dataset: %s" % dataset_name

    return dataset
