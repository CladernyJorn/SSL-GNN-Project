import copy
import numpy as np
from tqdm import tqdm
import torch
import os
# models
from models import build_model

# self-made utils
from utils.load_data import load_dataloader, LinearProbingDataLoader
from utils.utils import (
    create_optimizer,
    set_random_seed,
    create_scheduler,
    LogisticRegression,
    accuracy,
    show_occupied_memory
)
import warnings

warnings.filterwarnings("ignore")


class ModelTrainer:

    def __init__(self, args):
        self._args = args
        self._device = args.device if args.device >= 0 else "cpu"
        self._use_sampler = args.use_sampler

    def train_eval(self):
        args = self._args
        set_random_seed(0)  # seed for pretrain
        self._full_graph_forward = hasattr(args,
                                           "full_graph_forward") and args.full_graph_forward and not args.linear_prob
        memory_before_load = show_occupied_memory()
        self._args.num_features, self._pretrain_dataloader = load_dataloader("pretrain", args.dataset, self._args)
        print(f"Data memory usage: {show_occupied_memory() - memory_before_load:.2f} MB")
        self.model = build_model(self._args)
        self.optimizer = create_optimizer(args.optimizer, self.model, args.lr, args.weight_decay)
        self.scheduler = None
        if args.scheduler:
            self.scheduler = create_scheduler(args, self.optimizer)

        # need to pretrain
        if not args.load_model:
            # get initial results
            if args.eval_first:
                self.infer_embeddings()
                print("initial test:")
                self.evaluate()
            self.pretrain()
            model = self.model.cpu()
            if args.save_model:
                os.makedirs(args.save_model_path, exist_ok=True)
                model_name = f"{args.model}_{args.dataset}_{args.pretrain_sampling_method}_checkpoint.pt"
                save_path = os.path.join(args.save_model_path, model_name)
                print(f"Saveing model to {save_path}")
                torch.save(model.state_dict(), save_path)

        # no need to pretrain, eval directly
        if args.load_model:
            print(f"Loading model from {args.load_model_path}")
            self.model.load_state_dict(torch.load(args.load_model_path))

        print("---- start evaluation ----")
        test_list = []
        for i, seed in enumerate(args.seeds):
            print(f"####### Run{i} for seed {seed} #######")
            set_random_seed(seed)
            self.infer_embeddings()
            test_acc = self.evaluate()
            test_list = test_list + test_acc
        final_test_acc, final_test_acc_std = np.mean(test_list), np.std(test_list)
        print(f"# final-test-acc: {final_test_acc:.4f}±{final_test_acc_std:.4f}", end="")

        return final_test_acc

    def pretrain(self):
        args = self._args
        print(f"\n--- Start pretraining {args.model} model on {args.dataset} using {args.pretrain_sampling_method} sampling ---")

        self.model.to(self._device)
        for epoch in range(args.max_epoch):
            epoch_iter = tqdm(self._pretrain_dataloader)
            losses = []
            for batch_g in epoch_iter:
                self.model.train()
                loss = self.get_loss(batch_g, epoch)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3)
                self.optimizer.step()
                epoch_iter.set_description(f"# Epochs {epoch}: train_loss: {loss.item():.4f}")
                losses.append(loss.item())
                if self.scheduler is not None:
                    self.scheduler.step()
                if args.model == "bgrl":
                    self.model.update_moving_average()  # bgrl uses EMA updating strategy
            print(f"# Epoch {epoch} | train_loss: {np.mean(losses):.4f}, Memory: {show_occupied_memory():.2f} MB")

    def get_loss(self, batch_g, epoch):
        args = self._args
        if args.model == "graphmae2" and args.pretrain_sampling_method == "lc":
            if args.drop_edge_rate > 0:
                batch_g, targets, _, node_idx, drop_g1, drop_g2 = batch_g
                batch_g = batch_g.to(self._device)
                drop_g1 = drop_g1.to(self._device)
                drop_g2 = drop_g2.to(self._device)
                x = batch_g.ndata.pop("feat")
                loss = self.model(batch_g, x, targets, epoch, drop_g1, drop_g2)
            else:
                batch_g, targets, _, node_idx = batch_g
                batch_g = batch_g.to(self._device)
                x = batch_g.ndata["feat"]
                loss = self.model(batch_g, x, targets, epoch)
        else:
            if args.pretrain_sampling_method == "lc":
                batch_g, targets, _, node_idx = batch_g
            elif args.pretrain_sampling_method == "saint":
                batch_g = batch_g
            elif args.pretrain_sampling_method == "khop":
                input_nodes, output_nodes, batch_g = batch_g
            batch_g = batch_g.to(self._device)
            x = batch_g.ndata["feat"]
            loss = self.model(batch_g, x)
        return loss

    def infer_embeddings(self):  # preparing embeddings and labels
        args = self._args
        num_info, label_info, self._eval_dataloader = load_dataloader("eval", args.dataset, args)
        self._num_train, self._num_val, self._num_test = num_info
        self._train_label, self._val_label, self._test_label = label_info
        with torch.no_grad():
            self.model.to(self._device)
            self.model.eval()
            embeddings = []
            for batch in tqdm(self._eval_dataloader, desc="Infering..."):
                if args.eval_sampling_method == "lc":
                    batch_g, targets, _, node_idx = batch
                    batch_g = batch_g.to(self._device)
                    x = batch_g.ndata.pop("feat").to(self._device)
                    targets = targets.to(self._device)
                elif args.eval_sampling_method == "khop":
                    input_nodes, output_nodes, batch_g = batch
                    batch_g = batch_g.to(self._device)
                    x = batch_g.ndata.pop("feat").to(self._device)
                    targets = output_nodes.to(self._device)
                batch_emb = self.model.embed(batch_g, x)[targets]
                embeddings.append(batch_emb.cpu())
        self._embeddings = torch.cat(embeddings, dim=0)
        self._train_emb = self._embeddings[:self._num_train]
        self._val_emb = self._embeddings[self._num_train:self._num_train + self._num_val]
        self._test_emb = self._embeddings[self._num_train + self._num_val:]
        print(f"train embeddings:{len(self._train_emb)}")
        print(f"val embeddings  :{len(self._val_emb)}")
        print(f"test embeddings :{len(self._test_emb)}")

    def evaluate(self):
        args = self._args
        train_emb, val_emb, test_emb = self._train_emb, self._val_emb, self._test_emb
        train_label = self._train_label.to(torch.long)
        val_label = self._val_label.to(torch.long)
        test_label = self._test_label.to(torch.long)
        acc = []
        for i, seed in enumerate(args.linear_prob_seeds):
            print(f"####### Run seed {seed} for LinearProbing...")
            set_random_seed(seed)
            criterion = torch.nn.CrossEntropyLoss()
            classifier = LogisticRegression(self._train_emb.shape[1], int(train_label.max().item() + 1)).to(
                self._device)
            optimizer = create_optimizer("adam", classifier, args.lr_f, args.weight_decay_f)
            train_loader = LinearProbingDataLoader(np.arange(len(train_emb)), train_emb, train_label,
                                                   batch_size=args.batch_size_linear_prob, num_workers=4,
                                                   persistent_workers=True, shuffle=True)
            val_loader = LinearProbingDataLoader(np.arange(len(val_emb)), val_emb, val_label,
                                                 batch_size=args.batch_size_linear_prob,
                                                 num_workers=4, persistent_workers=True, shuffle=False)
            test_loader = LinearProbingDataLoader(np.arange(len(test_emb)), test_emb, test_label,
                                                  batch_size=args.batch_size_linear_prob,
                                                  num_workers=4, persistent_workers=True, shuffle=False)
            best_val_acc = 0
            best_classifier = None
            epoch_iter = tqdm(range(args.max_epoch_f)) if not args.no_verbose else range(args.max_epoch_f)
            for epoch in epoch_iter:
                classifier.train()
                classifier.to(self._device)
                for batch_x, batch_label in train_loader:
                    batch_x = batch_x.to(self._device)
                    batch_label = batch_label.to(self._device)
                    pred = classifier(batch_x)
                    loss = criterion(pred, batch_label)
                    optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
                    optimizer.step()
                with torch.no_grad():
                    classifier.eval()
                    val_acc = self.eval_forward(classifier, val_loader, val_label)
                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    best_classifier = copy.deepcopy(classifier)
                if not args.no_verbose:
                    epoch_iter.set_description(
                        f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_acc:{val_acc:.4f}")

            best_classifier.eval()
            with torch.no_grad():
                test_acc = self.eval_forward(best_classifier, test_loader, test_label)
            print(f"# test_acc: {test_acc:.4f}")
            acc.append(test_acc)
        print(f"# test_acc: {np.mean(acc):.4f}±{np.std(acc):.4f}")
        return acc

    def eval_forward(self, classifier, loader, label):
        pred_all = []
        for batch_x, _ in loader:
            batch_x = batch_x.to(self._device)
            pred = classifier(batch_x)
            pred_all.append(pred.cpu())
        pred = torch.cat(pred_all, dim=0)
        acc = accuracy(pred, label)
        return acc
