import logging
import numpy as np
from tqdm import tqdm
import torch

# self-made utils
from utils.dataset_utils import load_dataset,load_dataloader
from utils.train_evaluation_utils import node_classification_evaluation,create_optimizer


# graphMAE only
from graphmae.utils import (
    build_args,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs
)
from graphmae.graphmae import PreModel

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def build_model(args):
    model = PreModel(
        in_dim=args.num_features,
        num_hidden=args.num_hidden,
        num_layers=args.num_layers,
        nhead=args.num_heads,
        nhead_out=args.num_out_heads,
        activation=args.activation,
        feat_drop=args.in_drop,
        attn_drop=args.attn_drop,
        negative_slope=args.negative_slope,
        residual=args.residual,
        encoder_type=args.encoder,
        decoder_type=args.decoder,
        mask_rate=args.mask_rate,
        norm=args.norm,
        loss_fn=args.loss_fn,
        drop_edge_rate=args.drop_edge_rate,
        replace_rate=args.replace_rate,
        alpha_l=args.alpha_l,
        concat_hidden=args.concat_hidden,
    )
    return model


def pretrain_whole_graph(model, graph, feat, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f,
             max_epoch_f, linear_prob, logger=None):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)

    epoch_iter = tqdm(range(max_epoch))

    for epoch in epoch_iter:
        model.train()

        loss, loss_dict = model(graph, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)

        if (epoch + 1) % 200 == 0:
            node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device,
                                           linear_prob, mute=True)

    # return best_model
    return model

def pretrain_batch_sampler(model, graph,pretrain_dataloader, feat, optimizer, device, scheduler, num_classes, lr_f, weight_decay_f,
             max_epoch_f, linear_prob, logger=None):
    logging.info("start training..")
    # graph = graph.to(device)
    # x = feat.to(device)

    epoch_iter = tqdm(pretrain_dataloader)

    for i,subgraph in enumerate(epoch_iter):
        subgraph = subgraph.to(device)
        x=subgraph.ndata['feat'].to(device)
        model.train()

        loss, loss_dict = model(subgraph, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Iters {i}: train_loss: {loss.item():.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=i)

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
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate
    optim_type = args.optimizer
    loss_fn = args.loss_fn
    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler

    graph, (num_features, num_classes) = load_dataset(dataset_name)
    args.num_features = num_features

    acc_list = []
    estp_acc_list = []
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(
                name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None

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
            model = pretrain_whole_graph(model, graph, x, optimizer, max_epoch, device, scheduler, num_classes, lr_f,
                             weight_decay_f, max_epoch_f, linear_prob, logger)
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

        if logger is not None:
            logger.finish()

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
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate
    optim_type = args.optimizer
    loss_fn = args.loss_fn
    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler
    budget=args.budget

    graph,pretrain_dataloader, num_iters,(num_features, num_classes) = load_dataloader(dataset_name,budget,args.num_iters,max_epoch,device)
    args.num_features = num_features
    acc_list = []
    estp_acc_list = []
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(
                name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None

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
            model = pretrain_batch_sampler(model, graph, pretrain_dataloader, x,optimizer, device, scheduler, num_classes, lr_f,
                             weight_decay_f, max_epoch_f, linear_prob, logger)
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

        if logger is not None:
            logger.finish()

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    estp_acc, estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    print(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")
    return final_acc

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "./configs/best_GraphMAE_configs.yml")
    print(args)
    if not args.use_sampler:
        train_eval_whole_graph(args)
    else:
        train_eval_batch_sampler(args)
