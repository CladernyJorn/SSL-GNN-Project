import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from ..gnn_modules import setup_module
except:
    from gnn_modules import setup_module
# PyG implemented encoder only GCN available
# class Encoder(torch.nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, activation,
#                  base_model=GCNConv, k: int = 2):
#         super(Encoder, self).__init__()
#         self.base_model = base_model
#
#         assert k >= 2
#         self.k = k
#         self.conv = [base_model(in_channels, 2 * out_channels)]
#         for _ in range(1, k-1):
#             self.conv.append(base_model(2 * out_channels, 2 * out_channels))
#         self.conv.append(base_model(2 * out_channels, out_channels))
#         self.conv = nn.ModuleList(self.conv)
#
#         self.activation = activation
#
#     def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
#         for i in range(self.k):
#             x = self.activation(self.conv[i](x, edge_index))
#         return x


class Model(torch.nn.Module):
    def __init__(self, in_feat,encoder_type, num_hidden: int, num_proj_hidden: int, activation,num_layers,
                 tau: float = 0.5,nhead: int = 4):
        super(Model, self).__init__()

        self._encoder_type = encoder_type
        assert num_hidden % nhead == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        #self.encoder=Encoder(in_feat,num_hidden,activation,encoder_type=encoder_type)

        # DGL implemented encoders, encoder_type in [gcn,gat,dotgat,gin] available
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_feat,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            allow_zero_degree=True
        )

        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, g,x: torch.Tensor):
        return self.encoder(g,x)

    def embed(self, g,x: torch.Tensor):
        return self.encoder(g,x)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret


