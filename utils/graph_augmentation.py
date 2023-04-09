import torch
import numpy as np
from dgl import DropEdge,FeatMask
import copy
import dgl
class Augmentation:
    # augmentation methods used in BGRL
    # implemented with DGL
    def __init__(self, p_f1=0.2, p_f2=0.1, p_e1=0.2, p_e2=0.3):
        """
        two simple graph augmentation functions --> "Node feature masking" and "Edge masking"
        Random binary node feature mask following Bernoulli distribution with parameter p_f
        Random binary edge mask following Bernoulli distribution with parameter p_e
        """
        self.p_f1 = p_f1
        self.p_f2 = p_f2
        self.p_e1 = p_e1
        self.p_e2 = p_e2
        self.method = "BGRL"

    def _feature_masking(self, data, device):
        edge_mask1 = DropEdge(p=self.p_e1)
        edge_mask2 = DropEdge(p=self.p_e2)

        # DGL.FeatMask will transform the origin graph as well
        # deepcopy first!
        graph_1 = copy.deepcopy(data)
        graph_2 = copy.deepcopy(data)

        graph_1 = edge_mask1(graph_1)
        graph_2 = edge_mask2(graph_2)
        feature_mask1 = FeatMask(p=self.p_f1, node_feat_names=['feat'])
        feature_mask2 = FeatMask(p=self.p_f2, node_feat_names=['feat'])
        graph_1 = feature_mask1(graph_1)
        graph_2 = feature_mask2(graph_2)
        return graph_1,graph_2
        # feat_mask1 = torch.FloatTensor(data.ndata['feat'].shape[1]).uniform_() > self.p_f1
        # feat_mask2 = torch.FloatTensor(data.ndata['feat'].shape[1]).uniform_() > self.p_f2
        # feat_mask1, feat_mask2 = feat_mask1.to(device), feat_mask2.to(device)
        # x1, x2 = data.ndata['feat'].clone(), data.ndata['feat'].clone()
        # x1, x2 = x1 * feat_mask1, x2 * feat_mask2
        #
        # edge_index1, edge_attr1 = dropout_adj(data.edge_index, data.edge_attr, p=self.p_e1)
        # edge_index2, edge_attr2 = dropout_adj(data.edge_index, data.edge_attr, p=self.p_e2)
        #
        # new_data1, new_data2 = data.clone(), data.clone()
        # new_data1.x, new_data2.x = x1, x2
        # new_data1.edge_index, new_data2.edge_index = edge_index1, edge_index2
        # new_data1.edge_attr, new_data2.edge_attr = edge_attr1, edge_attr2
        #
        # return new_data1, new_data2

    def __call__(self, data):
        return self._feature_masking(data)

def random_aug(graph, x, feat_drop_rate, edge_mask_rate):
    n_node = graph.number_of_nodes()
    edge_mask = mask_edge(graph, edge_mask_rate)
    feat = drop_feature(x, feat_drop_rate)

    ng = dgl.graph([])
    ng.add_nodes(n_node)
    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]
    ng.add_edges(nsrc, ndst)

    return ng, feat

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1),),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

def mask_edge(graph, mask_prob):
    E = graph.number_of_edges()

    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx
