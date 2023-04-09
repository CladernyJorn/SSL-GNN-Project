import dgl.dataloading
from dgl.data import (
    CoraGraphDataset, 
    CiteseerGraphDataset, 
    PubmedGraphDataset
)
import torch

GRAPH_DICT = {
    "cora": CoraGraphDataset,
    "citeseer": CiteseerGraphDataset,
    "pubmed": PubmedGraphDataset,
}

def load_dataset(dataset_name):
    assert dataset_name in GRAPH_DICT, f"Unknow dataset: {dataset_name}."
    dataset = GRAPH_DICT[dataset_name]()
    graph = dataset[0]
    graph = graph.remove_self_loop()
    graph = graph.add_self_loop()
    num_features = graph.ndata["feat"].shape[1]
    num_classes = dataset.num_classes
    return graph, (num_features, num_classes)

def load_dataloader(dataset_name,budget,num_iters,epoch_num,device):
    assert dataset_name in GRAPH_DICT, f"Unknow dataset: {dataset_name}."
    dataset = GRAPH_DICT[dataset_name]()
    graph = dataset[0]
    graph = graph.remove_self_loop()
    graph = graph.add_self_loop()
    num_features = graph.ndata["feat"].shape[1]
    num_classes = dataset.num_classes

    train_mask = graph.ndata["train_mask"]
    val_mask = graph.ndata["val_mask"]
    test_mask = graph.ndata["test_mask"]
    labels = graph.ndata["label"]

    sampler=dgl.dataloading.SAINTSampler('node',budget=budget)
     # whole graph will be input for node classification pretraining ( transductive )
    iters=num_iters if num_iters!=0 else epoch_num*graph.number_of_nodes()//budget
    pretrain_dataloader=dgl.dataloading.DataLoader(graph, torch.arange(iters),sampler)

    return graph,pretrain_dataloader, iters, (num_features, num_classes)