import os
import torch.multiprocessing
from ogb.nodeproppred import DglNodePropPredDataset
from datasets.dataset_utils import *


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
