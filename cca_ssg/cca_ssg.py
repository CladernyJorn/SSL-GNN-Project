import torch.nn as nn
import torch.nn.functional as F
#from dgl.nn import GraphConv
try:
    from ..gnn_modules import setup_module
except:
    from gnn_modules import setup_module

class CCA_SSG(nn.Module):
    def __init__(self, in_dim, encoder_type,hid_dim, out_dim, activation,num_layers, nhead=4):
        super().__init__()
        # if not use_mlp:
        #     self.backbone = GCN(in_dim, hid_dim, out_dim, n_layers)
        # else:
        #     self.backbone = MLP(in_dim, hid_dim, out_dim)
        self._encoder_type = encoder_type
        assert hid_dim % nhead == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = hid_dim // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = hid_dim
            enc_nhead = 1
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            allow_zero_degree=True
        )

    def embed(self, graph, feat):
        out = self.encoder(graph, feat)
        return out

    def forward(self, graph1, feat1, graph2, feat2):
        h1 = self.encoder(graph1, feat1)
        h2 = self.encoder(graph2, feat2)
        #print("h1",h1)
        z1 = (h1 - h1.mean(0)) / (h1.std(0)+1e-8)
        z2 = (h2 - h2.mean(0)) / (h2.std(0)+1e-8)

        #print("z1", z1)
        return z1, z2

