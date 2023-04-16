import os
import numpy as np
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader
import dgl.dataloading
from dgl.data import (
    CoraGraphDataset,
    CiteseerGraphDataset,
    PubmedGraphDataset
)
from ogb.nodeproppred import DglNodePropPredDataset
from sklearn.preprocessing import StandardScaler
from utils.augmentation import mask_edge

torch.multiprocessing.set_sharing_strategy('file_system')
GRAPH_DICT = {
    "cora": CoraGraphDataset,
    "citeseer": CiteseerGraphDataset,
    "pubmed": PubmedGraphDataset,
    "ogbn-arxiv": DglNodePropPredDataset
}


def load_dataset(dataset_name):
    assert dataset_name in GRAPH_DICT, f"Unknown full graph dataset: {dataset_name}."
    if dataset_name.startswith("ogbn"):
        dataset = GRAPH_DICT[dataset_name](dataset_name)
    else:
        dataset = GRAPH_DICT[dataset_name]()
    if dataset_name == "ogbn-arxiv":
        graph, labels = dataset[0]
        num_nodes = graph.num_nodes()
        split_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph = preprocess(graph)
        if not torch.is_tensor(train_idx):
            train_idx = torch.as_tensor(train_idx)
            val_idx = torch.as_tensor(val_idx)
            test_idx = torch.as_tensor(test_idx)
        feat = graph.ndata["feat"]
        feat = scale_feats(feat)
        graph.ndata["feat"] = feat
        train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True)
        val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True)
        test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True)
        graph.ndata["label"] = labels.view(-1)
        graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"] = train_mask, val_mask, test_mask
    else:
        graph = dataset[0]
        graph = graph.remove_self_loop()
        graph = graph.add_self_loop()
    num_features = graph.ndata["feat"].shape[1]
    num_classes = dataset.num_classes
    return graph, (num_features, num_classes)


def load_large_dataset(dataset_name, data_dir, ego_graphs_file_path):
    if dataset_name.startswith("ogbn"):
        dataset = DglNodePropPredDataset(dataset_name, root=os.path.join(data_dir, "dataset"))
        graph, label = dataset[0]
        if "year" in graph.ndata:
            del graph.ndata["year"]
        if not graph.is_multigraph:
            graph = preprocess(graph)
        graph = graph.remove_self_loop().add_self_loop()
        split_idx = dataset.get_idx_split()
        labels = label.view(-1)
        feats = graph.ndata.pop("feat")
        if dataset_name in ("ogbn-arxiv", "ogbn-papers100M"):
            feats = scale_feats(feats)
        train_lbls = labels[split_idx["train"]]
        val_lbls = labels[split_idx["valid"]]
        test_lbls = labels[split_idx["test"]]
        labels = torch.cat([train_lbls, val_lbls, test_lbls])
        if not os.path.exists(ego_graphs_file_path):
            raise FileNotFoundError(f"{ego_graphs_file_path} doesn't exist")
        else:
            nodes = torch.load(ego_graphs_file_path)
        return feats, graph, labels, split_idx, nodes
    else:
        raise NotImplementedError


def load_dataloader(load_type, dataset_name, args):
    # load_type in [pretrain, eval]
    feats, graph, labels, split_idx, ego_graph_nodes = load_large_dataset(dataset_name, args.data_dir,
                                                                          args.ego_graph_file_path)
    if args.sampling_method == "saint" and load_type == "pretrain":
        # saint sampling only for pretraining
        sampler = dgl.dataloading.SAINTSampler('node', budget=args.saint_budget)
        iters = args.num_iters if args.num_iters != 0 else graph.number_of_nodes() // args.budget
        graph.ndata['feat'] = feats
        dataloader = dgl.dataloading.DataLoader(graph, torch.arange(iters), sampler, device=args.device,
                                                batch_size=args.batch_size)
    elif args.sampling_method == "lc" or load_type == "eval":
        if load_type == "pretrain":
            batch_size = args.batch_size
            drop_edge_rate = args.drop_edge_rate
            shuffle = True
            if dataset_name == "ogbn-papers100M":
                ego_graph_nodes = ego_graph_nodes[0] + ego_graph_nodes[1] + ego_graph_nodes[2] + ego_graph_nodes[3]
            else:
                ego_graph_nodes = ego_graph_nodes[0] + ego_graph_nodes[1] + ego_graph_nodes[2]
        else:  # lc for eval
            batch_size = args.batch_size_f
            drop_edge_rate = 0
            shuffle = False
            ego_graph_nodes = ego_graph_nodes[0] + ego_graph_nodes[1] + ego_graph_nodes[2]
            num_train, num_val, num_test = [split_idx[k].shape[0] for k in ["train", "valid", "test"]]
            train_g_idx = np.arange(0, num_train)
            val_g_idx = np.arange(num_train, num_train + num_val)
            test_g_idx = np.arange(num_train + num_val, num_train + num_val + num_test)
            train_lbls, val_lbls, test_lbls = labels[train_g_idx], labels[val_g_idx], labels[test_g_idx]

        dataloader = OnlineLCLoader(
            root_nodes=ego_graph_nodes,
            graph=graph,
            feats=feats,
            drop_edge_rate=drop_edge_rate,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            persistent_workers=True,
            num_workers=1
        )
    else:
        raise NotImplementedError

    if load_type == "pretrain":
        return feats.shape[1], dataloader
    else:
        return (num_train, num_val, num_test), (train_lbls, val_lbls, test_lbls), dataloader


def preprocess(graph):
    feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat

    graph = graph.remove_self_loop().add_self_loop()
    graph.create_formats_()
    return graph


def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats


class OnlineLCLoader(DataLoader):
    def __init__(self, root_nodes, graph, feats, labels=None, drop_edge_rate=0, **kwargs):
        self.graph = graph
        self.labels = labels
        self._drop_edge_rate = drop_edge_rate
        self.ego_graph_nodes = root_nodes
        self.feats = feats

        dataset = np.arange(len(root_nodes))
        kwargs["collate_fn"] = self.__collate_fn__
        super().__init__(dataset, **kwargs)

    def drop_edge(self, g):
        if self._drop_edge_rate <= 0:
            return g, g

        g = g.remove_self_loop()
        mask_index1 = mask_edge(g, self._drop_edge_rate)
        mask_index2 = mask_edge(g, self._drop_edge_rate)
        g1 = dgl.remove_edges(g, mask_index1).add_self_loop()
        g2 = dgl.remove_edges(g, mask_index2).add_self_loop()
        return g1, g2

    def __collate_fn__(self, batch_idx):
        ego_nodes = [self.ego_graph_nodes[i] for i in batch_idx]
        subgs = [self.graph.subgraph(ego_nodes[i]) for i in range(len(ego_nodes))]
        sg = dgl.batch(subgs)

        nodes = torch.from_numpy(np.concatenate(ego_nodes)).long()
        num_nodes = [x.shape[0] for x in ego_nodes]
        cum_num_nodes = np.cumsum([0] + num_nodes)[:-1]

        if self._drop_edge_rate > 0:
            drop_g1, drop_g2 = self.drop_edge(sg)

        sg = sg.remove_self_loop().add_self_loop()
        sg.ndata["feat"] = self.feats[nodes]
        targets = torch.from_numpy(cum_num_nodes)

        if self.labels != None:
            label = self.labels[batch_idx]
        else:
            label = None

        if self._drop_edge_rate > 0:
            return sg, targets, label, nodes, drop_g1, drop_g2
        else:
            return sg, targets, label, nodes


class LinearProbingDataLoader(DataLoader):
    def __init__(self, idx, feats, labels=None, **kwargs):
        self.labels = labels
        self.feats = feats

        kwargs["collate_fn"] = self.__collate_fn__
        super().__init__(dataset=idx, **kwargs)

    def __collate_fn__(self, batch_idx):
        feats = self.feats[batch_idx]
        label = self.labels[batch_idx]

        return feats, label
