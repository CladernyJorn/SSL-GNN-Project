import torch.nn.functional as F
import torch.nn as nn
import torch
from typing import Optional
import numpy as np
import copy
from gnn_modules import setup_module
from utils.augmentation import random_aug

class model_bgrl(nn.Module):

    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            encoder_type: str = "gcn",
            pred_hid: int = 256,
            moving_average_decay: float = 0.99,
            epochs: int = 1000,
            drop_edge_rate_1: float = 0.5,
            drop_edge_rate_2: float = 0.5,
            drop_feature_rate_1: float = 0.5,
            drop_feature_rate_2: float = 0.5,
    ):
        super().__init__()
        self._encoder_type = encoder_type
        self.drop_edge_rate_1 = drop_edge_rate_1
        self.drop_edge_rate_2 = drop_edge_rate_2
        self.drop_feature_rate_1 = drop_feature_rate_1
        self.drop_feature_rate_2 = drop_feature_rate_2
        assert num_hidden % nhead == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1
        # DGL implemented encoders, encoder_type in [gcn,gat,dotgat,gin] available
        self.student_encoder = setup_module(
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
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm
        )
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        set_requires_grad(self.teacher_encoder, False)
        self.teacher_ema_updater = EMA(moving_average_decay, epochs)
        self.student_predictor = nn.Sequential(nn.Linear(num_hidden, pred_hid), nn.PReLU(),
                                               nn.Linear(pred_hid, num_hidden))
        self.student_predictor.apply(init_weights)

    def reset_moving_average(self):
        del self.teacher_encoder
        self.teacher_encoder = None

    def update_moving_average(self):
        assert self.teacher_encoder is not None, 'teacher encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)

    def forward(self, g, x):
        graph1, feat1 = random_aug(g.remove_self_loop(), x, self.drop_feature_rate_1, self.drop_edge_rate_1)
        graph2, feat2 = random_aug(g.remove_self_loop(), x, self.drop_feature_rate_2, self.drop_edge_rate_2)
        # encoding
        v1_student = self.student_encoder(graph1, feat1)
        v2_student = self.student_encoder(graph2, feat2)
        v1_pred = self.student_predictor(v1_student)
        v2_pred = self.student_predictor(v2_student)
        with torch.no_grad():
            v1_teacher = self.teacher_encoder(graph1, feat1)
            v2_teacher = self.teacher_encoder(graph2, feat2)
        loss1 = loss_fn(v1_pred, v2_teacher.detach())
        loss2 = loss_fn(v2_pred, v1_teacher.detach())
        loss = loss1 + loss2
        return loss.mean()
    def embed(self, g, x: torch.Tensor):
        return self.student_encoder(g, x)

class EMA:
    def __init__(self, beta, epochs):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.total_steps = epochs

    def update_average(self, old, new):
        if old is None:
            return new
        beta = 1 - (1 - self.beta) * (np.cos(np.pi * self.step / self.total_steps) + 1) / 2.0
        self.step += 1
        return old * beta + (1 - beta) * new

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
