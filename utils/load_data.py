import os
import numpy as np
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader
import dgl.dataloading
from utils.augmentation import mask_edge
from datasets import load_small_dataset, load_large_dataset,load_amazon_exp_dataset


# torch.multiprocessing.set_sharing_strategy('file_system')

def load_dataset(dataset_name):
    if dataset_name in ['cora', 'citeseer', 'pubmed']:
        return load_small_dataset(dataset_name)
    elif dataset_name=='amazon_experimental_dataset':
        return load_amazon_exp_dataset()
def load_dataloader(load_type, dataset_name, args):
    # load_type in [pretrain, eval]
    feats, graph, labels, split_idx, ego_graph_nodes = load_large_dataset(dataset_name, args.data_dir,
                                                                          args.ego_graph_file_path)
    if args.pretrain_sampling_method == "saint" and load_type == "pretrain":
        # saint sampling only for pretraining
        if not args.no_verbose:
            print("loading data with Saint sampler for pretraining")
        sampler = dgl.dataloading.SAINTSampler('node', budget=args.saint_budget)
        iters = graph.number_of_nodes() // args.batch_size
        graph.ndata['feat'] = feats
        dataloader = dgl.dataloading.DataLoader(graph, torch.arange(iters), sampler, device=args.device)

    elif args.pretrain_sampling_method == "clustergcn" and load_type == "pretrain":
        # clustergcn sampling only for pretraining
        if not args.no_verbose:
            print("loading data with ClusterGCN sampler for pretraining")
        sampler = dgl.dataloading.ClusterGCNSampler(graph, k=args.cluster_gcn_num_parts,
                                                    cache_path=args.cluster_gcn_cache_path)
        graph.ndata['feat'] = feats
        dataloader = dgl.dataloading.DataLoader(graph, torch.arange(args.cluster_gcn_num_parts), sampler,
                                                batch_size=args.cluster_gcn_batch_size, shuffle=True,
                                                device=args.device)

    elif (load_type == "pretrain" and args.pretrain_sampling_method == "khop") or (
            load_type == "eval" and args.eval_sampling_method == "khop"):
        graph.ndata['feat'] = feats
        if load_type == "pretrain":
            if not args.no_verbose:
                print("loading data with khop sampler for pretraining")
            batch_size = args.batch_size
            shuffle = True
        elif load_type == "eval":  # lc for eval
            if not args.no_verbose:
                print("loading data with khop sampler for evaluating")
            batch_size = args.batch_size_f
            shuffle = False
        sampler = dgl.dataloading.ShaDowKHopSampler(args.khop_fanouts)
        dataloader = dgl.dataloading.DataLoader(graph, torch.arange(graph.num_nodes()), sampler,
                                                batch_size=batch_size, shuffle=shuffle, device=args.device)

    elif (load_type == "pretrain" and args.pretrain_sampling_method == "lc") or (
            load_type == "eval" and args.eval_sampling_method == "lc"):
        if load_type == "pretrain":
            if not args.no_verbose:
                print("loading data with LC sampler for pretraining")
            batch_size = args.batch_size
            drop_edge_rate = args.drop_edge_rate
            shuffle = True
            if dataset_name == "ogbn-papers100M":
                ego_graph_nodes = ego_graph_nodes[0] + ego_graph_nodes[1] + ego_graph_nodes[2] + ego_graph_nodes[3]
            else:
                ego_graph_nodes = ego_graph_nodes[0] + ego_graph_nodes[1] + ego_graph_nodes[2]
        elif load_type == "eval":  # lc for eval
            if not args.no_verbose:
                print("loading data with LC sampler for evaluating")
            batch_size = args.batch_size_f
            drop_edge_rate = 0
            shuffle = False
            ego_graph_nodes = ego_graph_nodes[0] + ego_graph_nodes[1] + ego_graph_nodes[2]
        dataloader = LocalClusteringLoader(
            root_nodes=ego_graph_nodes,
            graph=graph,
            feats=feats,
            drop_edge_rate=drop_edge_rate,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            persistent_workers=False,
            num_workers=0
        )
    else:
        raise NotImplementedError(f"{args.pretrain_sampling_method} doesn't support pretraining")

    if load_type == "pretrain":
        return feats.shape[1], dataloader
    else:
        num_train, num_val, num_test = [split_idx[k].shape[0] for k in ["train", "valid", "test"]]
        train_g_idx = np.arange(0, num_train)
        val_g_idx = np.arange(num_train, num_train + num_val)
        test_g_idx = np.arange(num_train + num_val, num_train + num_val + num_test)
        train_lbls, val_lbls, test_lbls = labels[train_g_idx], labels[val_g_idx], labels[test_g_idx]
        return (num_train, num_val, num_test), (train_lbls, val_lbls, test_lbls), dataloader


class LocalClusteringLoader(DataLoader):
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
