import copy
import logging
import numpy as np
from tqdm import tqdm
import torch

# self-made utils
from utils.dataset_utils import load_dataset,load_dataloader
from utils.train_evaluation_utils import node_classification_evaluation,create_optimizer

from dgl import DropEdge,FeatMask

# graphMAE only
from grace.utils import (
    build_args,
    set_random_seed,
    load_best_configs
)
from grace.grace import Model

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def build_model(args):
    model = Model(
        in_feat=args.num_features,
        encoder_type=args.encoder,
        num_hidden=args.num_hidden,
        num_proj_hidden=args.num_proj_hidden,
        activation=args.activation,
        num_layers=args.num_layers,
        tau=args.tau,
        nhead=args.num_heads
    )
    return model


def pretrain_whole_graph(model, graph, feat, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f,
             max_epoch_f, linear_prob,drop_edge_rate_1,drop_edge_rate_2,drop_feature_rate_1,drop_feature_rate_2):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)

    epoch_iter = tqdm(range(max_epoch))

    for epoch in epoch_iter:
        model.train()
        optimizer.zero_grad()
        edge_mask1=DropEdge(p=drop_edge_rate_1)
        edge_mask2 = DropEdge(p=drop_edge_rate_2)

        # DGL.FeatMask will transform the origin graph as well
        # deepcopy first!
        graph_1 = copy.deepcopy(graph)
        graph_2 = copy.deepcopy(graph)

        graph_1 = edge_mask1(graph_1)
        graph_2 = edge_mask2(graph_2)
        feature_mask1=FeatMask(p=drop_feature_rate_1,node_feat_names=['feat'])
        feature_mask2 = FeatMask(p=drop_feature_rate_2, node_feat_names=['feat'])
        graph_1=feature_mask1(graph_1)
        graph_2 = feature_mask2(graph_2)
        x_1 = graph_1.ndata['feat'].to(device)
        x_2 = graph_2.ndata['feat'].to(device)
        z1 = model(graph_1,x_1)
        z2 = model(graph_2,x_2)

        loss = model.loss(z1, z2)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")

        if (epoch + 1) % 200 == 0:
            node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device,
                                           linear_prob, mute=True)

    # return best_model
    return model

def pretrain_batch_sampler(model, graph,pretrain_dataloader, feat, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f,
             max_epoch_f, linear_prob,drop_edge_rate_1,drop_edge_rate_2,drop_feature_rate_1,drop_feature_rate_2):
    logging.info("start training..")
    # graph = graph.to(device)
    # x = feat.to(device)

    epoch_iter = tqdm(pretrain_dataloader)

    for i,subgraph in enumerate(epoch_iter):
        subgraph = subgraph.to(device)
        # x = subgraph.ndata['feat'].to(device)
        model.train()
        optimizer.zero_grad()
        edge_mask1=DropEdge(p=drop_edge_rate_1)
        edge_mask2 = DropEdge(p=drop_edge_rate_2)

        # DGL.FeatMask will transform the origin graph as well
        # deepcopy first!
        graph_1 = copy.deepcopy(subgraph)
        graph_2 = copy.deepcopy(subgraph)

        graph_1 = edge_mask1(graph_1)
        graph_2 = edge_mask2(graph_2)
        feature_mask1=FeatMask(p=drop_feature_rate_1,node_feat_names=['feat'])
        feature_mask2 = FeatMask(p=drop_feature_rate_2, node_feat_names=['feat'])
        graph_1=feature_mask1(graph_1)
        graph_2 = feature_mask2(graph_2)
        x_1 = graph_1.ndata['feat'].to(device)
        x_2 = graph_2.ndata['feat'].to(device)
        z1 = model(graph_1,x_1)
        z2 = model(graph_2,x_2)

        loss = model.loss(z1, z2)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Iters {i}: train_loss: {loss.item():.4f}")

        if (i + 1) % 200 == 0:
            node_classification_evaluation(model, graph, feat, num_classes, lr_f, weight_decay_f, max_epoch_f, device,
                                           linear_prob, mute=True)

    # return best_model
    return model


def train_eval_whole_graph(args):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch

    optim_type = args.optimizer
    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    max_epoch_f = args.max_epoch_f
    linear_prob = args.linear_prob
    load_model=args.load_model
    save_model=args.save_model
    use_scheduler = args.scheduler
    drop_edge_rate_1 = args.drop_edge_rate_1
    drop_edge_rate_2 = args.drop_edge_rate_2
    drop_feature_rate_1 = args.drop_feature_rate_1
    drop_feature_rate_2 = args.drop_feature_rate_2


    graph, (num_features, num_classes) = load_dataset(dataset_name)
    args.num_features = num_features

    acc_list = []
    estp_acc_list = []
    if isinstance(seeds,int):
        seeds=[seeds]
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)


        model = build_model(args)
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / max_epoch)) * 0.5
            # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
            # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None

        x = graph.ndata["feat"]
        if not load_model:
            model = pretrain_whole_graph(
                model, graph, x, optimizer, max_epoch, device, scheduler, num_classes, lr_f,
                weight_decay_f, max_epoch_f, linear_prob,
                drop_edge_rate_1=drop_edge_rate_1,
                drop_edge_rate_2 = drop_edge_rate_2,
                drop_feature_rate_1 = drop_feature_rate_1,
                drop_feature_rate_2 = drop_feature_rate_2
            )
            model = model.cpu()

        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("checkpoint.pt"))
        if save_model:
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")

        model = model.to(device)
        model.eval()

        final_acc, estp_acc = node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f,
                                                             max_epoch_f, device, linear_prob)
        acc_list.append(final_acc)
        estp_acc_list.append(estp_acc)


    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    estp_acc, estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    print(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")
    return final_acc
def train_eval_batch_sampler(args):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch

    optim_type = args.optimizer
    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    max_epoch_f = args.max_epoch_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    use_scheduler = args.scheduler
    drop_edge_rate_1 = args.drop_edge_rate_1
    drop_edge_rate_2 = args.drop_edge_rate_2
    drop_feature_rate_1 = args.drop_feature_rate_1
    drop_feature_rate_2 = args.drop_feature_rate_2

    use_scheduler = args.scheduler
    budget = args.budget

    graph, pretrain_dataloader, num_iters, (num_features, num_classes) = load_dataloader(dataset_name, budget,
                                                                                         args.num_iters, max_epoch,
                                                                                         device)
    args.num_features = num_features

    acc_list = []
    estp_acc_list = []
    if isinstance(seeds, int):
        seeds = [seeds]
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        model = build_model(args)
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / max_epoch)) * 0.5
            # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
            # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None

        x = graph.ndata["feat"]
        if not load_model:
            model = pretrain_batch_sampler(
                model, graph, pretrain_dataloader,x, optimizer, max_epoch, device, scheduler, num_classes, lr_f,
                weight_decay_f, max_epoch_f, linear_prob,
                drop_edge_rate_1=drop_edge_rate_1,
                drop_edge_rate_2=drop_edge_rate_2,
                drop_feature_rate_1=drop_feature_rate_1,
                drop_feature_rate_2=drop_feature_rate_2
            )
            model = model.cpu()

        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("checkpoint.pt"))
        if save_model:
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")

        model = model.to(device)
        model.eval()

        final_acc, estp_acc = node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f,
                                                             max_epoch_f, device, linear_prob)
        acc_list.append(final_acc)
        estp_acc_list.append(estp_acc)

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    estp_acc, estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    print(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")
    return final_acc

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "./configs/best_Grace_configs.yaml")
    print(args)
    if not args.use_sampler:
        train_eval_whole_graph(args)
    else:
        train_eval_batch_sampler(args)
