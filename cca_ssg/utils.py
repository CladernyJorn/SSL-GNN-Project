import argparse
import random
import yaml
import logging
import numpy as np
import dgl
import torch
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True

def build_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--use_cfg", action="store_true")

    parser.add_argument("--encoder", type=str, default="gcn")
    parser.add_argument("--activation", type=str, default="relu")

    parser.add_argument("--lambd", type=float, default=0.001)
    parser.add_argument("--drop_edge_rate_1", type=float, default=0.2)
    parser.add_argument("--drop_edge_rate_2", type=float, default=0.2)
    parser.add_argument("--drop_feature_rate_1", type=float, default=0.2)
    parser.add_argument("--drop_feature_rate_2", type=float, default=0.2)


    parser.add_argument("--num_hidden", type=int, default=512,
                        help="number of hidden units")
    parser.add_argument("--num_proj_hidden", type=int, default=512,
                        help="number of hidden units")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="number of hidden layers")


    parser.add_argument("--num_heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--scheduler", action="store_true", default=False)
    parser.add_argument("--max_epoch", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--max_epoch_f", type=int, default=300)
    parser.add_argument("--lr_f", type=float, default=0.001, help="learning rate for evaluation")
    parser.add_argument("--weight_decay_f", type=float, default=0.0, help="weight decay for evaluation")
    parser.add_argument("--linear_prob", action="store_true", default=True)

    parser.add_argument("--logging", action="store_true")
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    # graph sample + mini-batch
    parser.add_argument("--use_sampler", action="store_true", default=False)
    parser.add_argument("--budget", type=int, default=500, help="number of nodes sampled per batch with SaintSampler")
    parser.add_argument("--num_iters", type=int, default=0,
                        help="number of iters to train ( default= max_epoch*total_nodes_num/budget )")

    args = parser.parse_args()
    return args

def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.dataset not in configs:
        logging.info("Best args not found")
        return args

    logging.info("Using best configs")
    configs = configs[args.dataset]

    for k, v in configs.items():
        if "learning_rate" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args