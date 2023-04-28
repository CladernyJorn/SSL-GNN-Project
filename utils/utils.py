import torch
import torch.nn as nn
from torch import optim as optim
import random
import numpy as np
import argparse
import yaml
import psutil
import os
import sys


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
    if not args.supervised:
        if args.use_cfg:
            path = args.use_cfg_path
            if path == "":
                if args.model == "graphmae":
                    path = "./configs/GraphMAE_configs.yml"
                elif args.model == "graphmae2":
                    path = "./configs/GraphMAE2_configs.yml"
                elif args.model == "grace":
                    path = "./configs/Grace_configs.yml"
                elif args.model == "cca_ssg":
                    path = "./configs/CCA_SSG_configs.yml"
                elif args.model == "bgrl":
                    path = "./configs/BGRL_configs.yml"
                elif args.model == "ggd":
                    path = "./configs/GGD_configs.yml"
            with open(path, "r") as f:
                configs = yaml.load(f, yaml.FullLoader)
            if args.dataset not in configs:
                return args
            configs = configs[args.dataset]
            for k, v in configs.items():
                if "lr" in k or "weight_decay" in k or "tau" in k or "lambd" in k:
                    v = float(v)
                setattr(args, k, v)
    else:
        # supervised on experimental dataset
        print("load configs")
        if args.use_cfg:
            path = args.use_cfg_path
            if path == "":
                path = "./configs/exp_configs.yml"
            with open(path, "r") as f:
                configs = yaml.load(f, yaml.FullLoader)
            if args.model not in configs:
                raise NotImplementedError(f"Supervised {args.model} has no configs!")
            configs = configs[args.model]
            for k, v in configs.items():
                if "lr" in k or "weight_decay" in k or "tau" in k or "lambd" in k:
                    v = float(v)
                setattr(args, k, v)
    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        args.use_sampler = False
        if args.fast_result and len(args.seeds) != 1:
            args.seeds = range(1)  # train & eval only once
    else:
        args.use_sampler = True
        args.seeds = range(1) if len(args.seeds) != 1 else args.seeds  # only pretrain and infer once
        if args.fast_result and len(args.linear_prob_seeds) != 1:
            args.linear_prob_seeds = range(1)

    if not args.logging_path.endswith('.logs'):
        os.makedirs(args.logging_path, exist_ok=True)
        args.logging_path = os.path.join(args.logging_path,
                                         f"{args.model}_{args.dataset}_{args.pretrain_sampling_method}.log")
    if args.ego_graph_file_path == None:
        if args.dataset == "ogbn-arxiv":
            args.ego_graph_file_path = os.path.join("lc_ego_graphs", "ogbn-arxiv-lc-ego-graphs-256.pt")
        if args.dataset == "ogbn-products":
            args.ego_graph_file_path = os.path.join("lc_ego_graphs", "ogbn-products-lc-ego-graphs-256.pt")
    if args.pretrain_sampling_method == "clustergcn":
        if args.cluster_gcn_cache_path == "":
            args.cluster_gcn_cache_path = f"./tmp_{args.dataset}_{args.cluster_gcn_num_parts}_cluster_gcn.pkl"
    if args.pretrain_sampling_method == "khop":
        if isinstance(args.khop_fanouts, int):
            args.khop_fanouts = [args.khop_fanouts] * args.num_layers
        elif len(args.khop_fanouts) == 1:
            args.khop_fanouts = args.khop_fanouts * args.num_layers

    return args


class Logger(object):
    def __init__(self, filename="Default.log", no_verbose=False):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
        self.no_verbose = no_verbose

    def write(self, message):
        if not self.no_verbose:
            self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


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


def show_occupied_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 2


# build args for all models
def build_args():
    parser = argparse.ArgumentParser(description="SSL-GNN-Train Settings")
    parser.add_argument("--model", type=str, default="graphmae")
    parser.add_argument("--seeds", type=int, nargs="+", default=[i for i in range(20)],
                        help="for small: times of pretraining-infering-trainingclassifier (default 0-19); for large: times of infering-trainingclassifier(*n) (default 0)")
    parser.add_argument("--linear_prob_seeds", type=int, nargs="+", default=[i for i in range(10)],
                        help="only used in large dataset, training times of classifier per inference")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--device", type=int, default=-1)

    # supervised experimental training
    parser.add_argument("--supervised", action="store_true", default=False)
    parser.add_argument("--num_class", type=int, default=-1)


    # just for passing arguments, no need to set/change
    parser.add_argument("--num_features", type=int, default=-1)
    parser.add_argument("--logging_path", type=str, default='logs', help='path to save .log')
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
    # rest parameters in GraphMAE2
    parser.add_argument("--num_dec_layers", type=int, default=1)
    parser.add_argument("--num_remasking", type=int, default=3)
    parser.add_argument("--remask_rate", type=float, default=0.5)
    parser.add_argument("--remask_method", type=str, default="random")
    parser.add_argument("--mask_method", type=str, default="random")
    parser.add_argument("--mask_type", type=str, default="mask", help="`mask` or `drop`")
    parser.add_argument("--drop_edge_rate_f", type=float, default=0.0)
    parser.add_argument("--label_rate", type=float, default=1.0)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--delayed_ema_epoch", type=int, default=0)
    parser.add_argument("--momentum", type=float, default=0.996)

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

    # parameters of ggd
    parser.add_argument('--drop_feat', type=float, default=0.1)
    parser.add_argument("--num_proj_layers", type=int, default=1, help="number of project linear layers")

    # pretraining settings
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--scheduler", action="store_true", default=False)
    parser.add_argument("--warmup_steps", type=int, default=-1)
    parser.add_argument("--max_epoch", type=int, default=200, help="number of training epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")

    # eval settings
    parser.add_argument("--linear_prob", action="store_true", default=False)
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument("--max_epoch_f", type=int, default=30)
    parser.add_argument("--lr_f", type=float, default=0.001, help="learning rate for evaluation")
    parser.add_argument("--weight_decay_f", type=float, default=0.0, help="weight decay for evaluation")

    # graph batch training settings
    parser.add_argument("--use_sampler", action="store_true", default=False)
    parser.add_argument("--pretrain_sampling_method", type=str, default="lc", help="`lc` or `saint`")
    parser.add_argument("--eval_sampling_method", type=str, default="lc", help="`lc` or `saint`")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--batch_size_f", type=int, default=256)
    parser.add_argument("--batch_size_linear_prob", type=int, default=5120)
    parser.add_argument("--full_graph_forward", action="store_true", default=False)
    # cluster-gcn
    parser.add_argument("--cluster_gcn_batch_size", type=int, default=20, help="num of partitions per batch")
    parser.add_argument("--cluster_gcn_num_parts", type=int, default=1000, help="partition num of nodes in ClusterGCN")
    parser.add_argument("--cluster_gcn_cache_path", type=str, default="", help="path to cache cluster_gcn partitions")
    # saint
    parser.add_argument("--saint_budget", type=int, default=10000, help="saint sampled subgraph nodes")
    # khop
    parser.add_argument("--khop_fanouts", type=int, nargs="+", default=[10], help="num of neighbors for each gnn layer")
    # local clustering
    parser.add_argument("--ego_graph_file_path", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="data")

    # other settings
    parser.add_argument("--eval_first", action="store_true")
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--load_model_path", type=str, default="")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--save_model_path", type=str, default="checkpoints")
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--use_cfg_path", type=str, default="")
    parser.add_argument("--no_verbose", action="store_true", help="do not print process info")
    parser.add_argument("--eval_steps", type=int, default="200", help="epochs per evaluation during pretraining")
    parser.add_argument("--eval_nums", type=int, default="0", help="if set to non-zero, omit --eval_steps")
    parser.add_argument("--fast_result", action="store_true", help="if set, only pretrain-infer-trainclassifier once")

    args = parser.parse_args()
    return args
