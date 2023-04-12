import copy
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim as optim
import random
import numpy as np
import argparse
import yaml


def create_optimizer(opt, model, lr, weight_decay):
    opt_lower = opt.lower()
    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)
    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]

    if opt_lower == "adam":
        return optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        return optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        return optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        return optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"


def create_scheduler(args, optimizer):
    if not args.no_verbose:
        print("Use schedular")
    if args.model == "bgrl":
        scheduler = lambda epoch: epoch / 1000 if epoch < 1000 \
            else (1 + np.cos((epoch - 1000) * np.pi / (args.max_epoch - 1000))) * 0.5
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    else:
        scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / args.max_epoch)) * 0.5
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)


def get_current_lr(optimizer):
    return optimizer.state_dict()["param_groups"][0]["lr"]


def process_args(args):
    if args.use_cfg:
        path = args.use_cfg_path
        if path == "":
            if args.model == "graphmae":
                path = "./configs/best_GraphMAE_configs.yml"
            elif args.model == "grace":
                path = "./configs/best_Grace_configs.yml"
            elif args.model == "cca_ssg":
                path = "./configs/best_CCA_SSG_configs.yml"
            elif args.model == "bgrl":
                path = "./configs/best_BGRL_configs.yml"
        with open(path, "r") as f:
            configs = yaml.load(f, yaml.FullLoader)
        if args.dataset not in configs:
            return args
        configs = configs[args.dataset]
        for k, v in configs.items():
            if "lr" in k or "weight_decay" in k or "tau" in k or "lambd" in k:
                v = float(v)
            setattr(args, k, v)
    return args


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True


# training on down stream tasks
class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)

    def forward(self, x):
        logits = self.linear(x)
        return logits


def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)


# build args for all models
def build_args():
    parser = argparse.ArgumentParser(description="SSL-GNN-Train Settings")
    parser.add_argument("--model", type=str, default="graphmae")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--device", type=int, default=-1)

    # just for passing arguments, no need to set/change
    parser.add_argument("--num_features", type=int, default=-1)

    # encoder parameters
    parser.add_argument("--num_heads", type=int, default=4, help="number of hidden attention heads")
    parser.add_argument("--num_layers", type=int, default=2, help="number of hidden layers")
    parser.add_argument("--num_hidden", type=int, default=256, help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False, help="use residual connection")
    parser.add_argument("--in_drop", type=float, default=.2, help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=.1, help="attention dropout")
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--negative_slope", type=float, default=0.2, help="the negative slope of leaky relu for GAT")
    parser.add_argument("--activation", type=str, default="prelu")

    # parameters of GraphMAE
    parser.add_argument("--mask_rate", type=float, default=0.5)
    parser.add_argument("--drop_edge_rate", type=float, default=0.0)
    parser.add_argument("--replace_rate", type=float, default=0.0)
    parser.add_argument("--encoder", type=str, default="gat")
    parser.add_argument("--decoder", type=str, default="gat")
    parser.add_argument("--loss_fn", type=str, default="sce")
    parser.add_argument("--alpha_l", type=float, default=2, help="`pow`coefficient for `sce` loss")
    parser.add_argument("--concat_hidden", action="store_true", default=False)
    parser.add_argument("--num_out_heads", type=int, default=1, help="number of output attention heads")

    # parameters of Grace
    parser.add_argument("--num_proj_hidden", type=int, default=1, help="h->z linear in grace")
    parser.add_argument("--drop_edge_rate_1", type=float, default=0.5)
    parser.add_argument("--drop_edge_rate_2", type=float, default=0.5)
    parser.add_argument("--drop_feature_rate_1", type=float, default=0.5)
    parser.add_argument("--drop_feature_rate_2", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.5)

    # parameters of CCA-SSG
    parser.add_argument('--der', type=float, default=0.2, help='Drop edge ratio.')
    parser.add_argument('--dfr', type=float, default=0.2, help='Drop feature ratio.')
    parser.add_argument('--lambd', type=float, default=1e-3, help='trade-off ratio.')

    # parameters of BGRL (4 drop rate use Grace's arguments)
    parser.add_argument('--pred_hid', type=int, default=256)
    parser.add_argument("--moving_average_decay", type=float, default=0.99)

    # pretraining settings
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--scheduler", action="store_true", default=False)
    parser.add_argument("--warmup_steps", type=int, default=-1)
    parser.add_argument("--max_epoch", type=int, default=200, help="number of training epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")

    # eval settings
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument("--max_epoch_f", type=int, default=30)
    parser.add_argument("--lr_f", type=float, default=0.001, help="learning rate for evaluation")
    parser.add_argument("--weight_decay_f", type=float, default=0.0, help="weight decay for evaluation")

    # graph sampler settings
    parser.add_argument("--use_sampler", action="store_true", default=False)
    parser.add_argument("--budget", type=int, default=500, help="number of nodes sampled per batch with SaintSampler")
    parser.add_argument("--num_iters", type=int, default=0, help="iters to train (0: max_epoch*total_nodes_num/budget)")

    # other settings
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--load_model_path", type=str, default="")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--save_model_path", type=str, default="")
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--use_cfg_path", type=str, default="")
    parser.add_argument("--no_verbose", action="store_true", help="do not print process info")
    parser.add_argument("--eval_steps", type=int, default="200", help="epochs per evaluation during pretraining")
    parser.add_argument("--eval_nums", type=int, default="0", help="if set to non-zero, omit --eval_steps")
    args = parser.parse_args()
    return args
