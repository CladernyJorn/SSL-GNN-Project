import copy
import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import optim
from tensorboardX import SummaryWriter
import sys

# self-made utils
from utils.dataset_utils import load_dataset,load_dataloader
from utils.train_evaluation_utils import node_classification_evaluation,create_optimizer
from utils.graph_augmentation import Augmentation

# graphMAE only
from bgrl.utils import (
    build_args,
    load_best_configs,
)
from bgrl.bgrl import BGRL,LogisticRegression
torch.manual_seed(0)


class ModelTrainer_whole_graph:

    def __init__(self, args):
        self._args = args
        self._init()
        self.writer = SummaryWriter(log_dir="runs/BGRL_dataset({})".format(args.dataset))

    def _init(self):
        args = self._args
        self._device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
        self._dataset,(self.num_features, self.num_classes) = load_dataset(args.dataset)
        print(f"Data: {self._dataset}")
        hidden_layers = [int(l) for l in args.layers]
        layers = [self.num_features] + hidden_layers
        self._model = BGRL(layer_config=layers, pred_hid=args.pred_hid, dropout=args.dropout,
                                  epochs=args.epochs,encoder_type=args.encoder).to(self._device)
        #print(self._model)

        self._optimizer = optim.AdamW(params=self._model.parameters(), lr=args.lr, weight_decay=1e-5)
        # learning rate
        scheduler = lambda epoch: epoch / 1000 if epoch < 1000 \
            else (1 + np.cos((epoch - 1000) * np.pi / (self._args.epochs - 1000))) * 0.5
        self._scheduler = optim.lr_scheduler.LambdaLR(self._optimizer, lr_lambda=scheduler)

    def train(self):
        # get initial test results
        print("start training!")
        print("Initial Evaluation...")
        self.infer_embeddings()
        dev_best, dev_std_best, test_best, test_std_best = self.evaluate()
        self.writer.add_scalar("accs/val_acc", dev_best, 0)
        self.writer.add_scalar("accs/test_acc", test_best, 0)
        print("validation: {:.4f}, test: {:.4f}".format(dev_best, test_best))

        # start training
        self._model.train()
        for epoch in tqdm(range(self._args.epochs)):

            self._dataset.to(self._device)

            augmentation = Augmentation(float(self._args.aug_params[0]), float(self._args.aug_params[1]),
                                              float(self._args.aug_params[2]), float(self._args.aug_params[3]))
            view1, view2 = augmentation._feature_masking(self._dataset, self._device)

            v1_output, v2_output, loss = self._model(view1.to(self._device),view1.ndata['feat'].to(self._device),view2.to(self._device),view2.ndata['feat'].to(self._device))

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            self._scheduler.step()
            self._model.update_moving_average()
            sys.stdout.write('\rEpoch {}/{}, loss {:.4f}, lr {}'.format(epoch + 1, self._args.epochs, loss.data,
                                                                        self._optimizer.param_groups[0]['lr']))
            sys.stdout.flush()

            if (epoch + 1) % self._args.cache_step == 0:
                print("")
                print("\nEvaluating {}th epoch..".format(epoch + 1))

                self.infer_embeddings()
                dev_acc, dev_std, test_acc, test_std = self.evaluate()

                if dev_best < dev_acc:
                    dev_best = dev_acc
                    dev_std_best = dev_std
                    test_best = test_acc
                    test_std_best = test_std

                self.writer.add_scalar("stats/learning_rate", self._optimizer.param_groups[0]["lr"], epoch + 1)
                self.writer.add_scalar("accs/val_acc", dev_acc, epoch + 1)
                self.writer.add_scalar("accs/test_acc", test_acc, epoch + 1)
                print("validation: {:.4f}, test: {:.4f} \n".format(dev_acc, test_acc))

        print(f"validation: {dev_best:.4f}±{dev_std_best:.4f}, test: {test_best:.4f}±{test_std_best:.4f}\n")

        print()
        print("Training Done!")

    def infer_embeddings(self):

        self._model.train(False)
        self._embeddings = self._labels = None

        self._model.to(self._device)
        v1_output, v2_output, _ = self._model(
            graph1=self._dataset.to(self._device),x1=self._dataset.ndata['feat'].to(self._device),graph2=self._dataset.to(self._device), x2=self._dataset.ndata['feat'].to(self._device))
        emb = v1_output.detach()
        y = self._dataset.ndata['label'].detach()
        if self._embeddings is None:
            self._embeddings, self._labels = emb, y
        else:
            self._embeddings = torch.cat([self._embeddings, emb])
            self._labels = torch.cat([self._labels, y])
        print(len(self._embeddings))

    def evaluate(self):
        emb_dim, num_class = self._embeddings.shape[1], self._labels.unique().shape[0]

        dev_accs, test_accs = [], []

        for i in range(20):
            self._train_mask = self._dataset.ndata["train_mask"]
            self._dev_mask = self._dataset.ndata["val_mask"]
            self._test_mask = self._dataset.ndata["test_mask"]

            classifier = LogisticRegression(emb_dim, num_class).to(self._device)
            optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01, weight_decay=0.0)

            for epoch in range(2000):
                classifier.train()
                classifier.to(self._device)
                logits, loss = classifier(self._embeddings[self._train_mask].to(self._device), self._labels[self._train_mask].to(self._device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            dev_logits, _ = classifier(self._embeddings[self._dev_mask].to(self._device), self._labels[self._dev_mask].to(self._device))
            test_logits, _ = classifier(self._embeddings[self._test_mask].to(self._device), self._labels[self._test_mask].to(self._device))
            dev_preds = torch.argmax(dev_logits.cpu(), dim=1)
            test_preds = torch.argmax(test_logits.cpu(), dim=1)

            dev_acc = (torch.sum(dev_preds == self._labels[self._dev_mask]).float() /
                       self._labels[self._dev_mask].shape[0]).detach().cpu().numpy()
            test_acc = (torch.sum(test_preds == self._labels[self._test_mask]).float() /
                        self._labels[self._test_mask].shape[0]).detach().cpu().numpy()

            dev_accs.append(dev_acc * 100)
            test_accs.append(test_acc * 100)

        dev_accs = np.stack(dev_accs)
        test_accs = np.stack(test_accs)

        dev_acc, dev_std = dev_accs.mean(), dev_accs.std()
        test_acc, test_std = test_accs.mean(), test_accs.std()

        return dev_acc, dev_std, test_acc, test_std


class ModelTrainer_batch_sampler:

    def __init__(self, args):
        self._args = args
        self._init()
        self.writer = SummaryWriter(log_dir="runs/BGRL_dataset({})".format(args.dataset))

    def _init(self):
        args = self._args
        self._device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
        self._dataset,self.pretrain_dataloader,self.num_iters,(self.num_features, self.num_classes) = load_dataloader(args.dataset,args.budget,args.num_iters,args.epochs,args.device)
        print(f"Data: {self._dataset}")
        hidden_layers = [int(l) for l in args.layers]
        layers = [self.num_features] + hidden_layers
        self._model = BGRL(layer_config=layers, pred_hid=args.pred_hid, dropout=args.dropout,
                                  epochs=args.epochs,encoder_type=args.encoder).to(self._device)
        #print(self._model)

        self._optimizer = optim.AdamW(params=self._model.parameters(), lr=args.lr, weight_decay=1e-5)
        # learning rate
        scheduler = lambda epoch: epoch / 1000 if epoch < 1000 \
            else (1 + np.cos((epoch - 1000) * np.pi / (self._args.epochs - 1000))) * 0.5
        self._scheduler = optim.lr_scheduler.LambdaLR(self._optimizer, lr_lambda=scheduler)

    def train(self):
        # get initial test results
        print("start training!")
        print("Initial Evaluation...")
        self.infer_embeddings()
        dev_best, dev_std_best, test_best, test_std_best = self.evaluate()
        self.writer.add_scalar("accs/val_acc", dev_best, 0)
        self.writer.add_scalar("accs/test_acc", test_best, 0)
        print("validation: {:.4f}, test: {:.4f}".format(dev_best, test_best))

        # start training
        self._model.train()
        for i,subgraph in enumerate(tqdm(self.pretrain_dataloader)):

            self._dataset_ls=subgraph

            self._dataset_ls.to(self._device)

            augmentation = Augmentation(float(self._args.aug_params[0]), float(self._args.aug_params[1]),
                                              float(self._args.aug_params[2]), float(self._args.aug_params[3]))
            view1, view2 = augmentation._feature_masking(self._dataset_ls, self._device)

            v1_output, v2_output, loss = self._model(view1.to(self._device),view1.ndata['feat'].to(self._device),view2.to(self._device),view2.ndata['feat'].to(self._device))

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            self._scheduler.step()
            self._model.update_moving_average()
            sys.stdout.write('\rIter {}/{}, loss {:.4f}, lr {}'.format(i + 1, self.num_iters, loss.data,
                                                                        self._optimizer.param_groups[0]['lr']))
            sys.stdout.flush()

            if (i + 1) % self._args.cache_step == 0:
                print("")
                print("\nEvaluating {}th epoch..".format(i + 1))

                self.infer_embeddings()
                dev_acc, dev_std, test_acc, test_std = self.evaluate()

                if dev_best < dev_acc:
                    dev_best = dev_acc
                    dev_std_best = dev_std
                    test_best = test_acc
                    test_std_best = test_std

                self.writer.add_scalar("stats/learning_rate", self._optimizer.param_groups[0]["lr"], i + 1)
                self.writer.add_scalar("accs/val_acc", dev_acc, i + 1)
                self.writer.add_scalar("accs/test_acc", test_acc, i + 1)
                print("validation: {:.4f}, test: {:.4f} \n".format(dev_acc, test_acc))

        print(f"validation: {dev_best:.4f}±{dev_std_best:.4f}, test: {test_best:.4f}±{test_std_best:.4f}\n")

        print()
        print("Training Done!")

    def infer_embeddings(self):

        self._model.train(False)
        self._embeddings = self._labels = None

        self._model.to(self._device)
        v1_output, v2_output, _ = self._model(
            graph1=self._dataset.to(self._device),x1=self._dataset.ndata['feat'].to(self._device),graph2=self._dataset.to(self._device), x2=self._dataset.ndata['feat'].to(self._device))
        emb = v1_output.detach()
        y = self._dataset.ndata['label'].detach()
        if self._embeddings is None:
            self._embeddings, self._labels = emb, y
        else:
            self._embeddings = torch.cat([self._embeddings, emb])
            self._labels = torch.cat([self._labels, y])
        print(len(self._embeddings))

    def evaluate(self):
        emb_dim, num_class = self._embeddings.shape[1], self._labels.unique().shape[0]

        dev_accs, test_accs = [], []

        for i in range(20):
            self._train_mask = self._dataset.ndata["train_mask"]
            self._dev_mask = self._dataset.ndata["val_mask"]
            self._test_mask = self._dataset.ndata["test_mask"]

            classifier = LogisticRegression(emb_dim, num_class).to(self._device)
            optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01, weight_decay=0.0)

            for epoch in range(2000):
                classifier.train()
                classifier.to(self._device)
                logits, loss = classifier(self._embeddings[self._train_mask].to(self._device), self._labels[self._train_mask].to(self._device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            dev_logits, _ = classifier(self._embeddings[self._dev_mask].to(self._device), self._labels[self._dev_mask].to(self._device))
            test_logits, _ = classifier(self._embeddings[self._test_mask].to(self._device), self._labels[self._test_mask].to(self._device))
            dev_preds = torch.argmax(dev_logits.cpu(), dim=1)
            test_preds = torch.argmax(test_logits.cpu(), dim=1)

            dev_acc = (torch.sum(dev_preds == self._labels[self._dev_mask]).float() /
                       self._labels[self._dev_mask].shape[0]).detach().cpu().numpy()
            test_acc = (torch.sum(test_preds == self._labels[self._test_mask]).float() /
                        self._labels[self._test_mask].shape[0]).detach().cpu().numpy()

            dev_accs.append(dev_acc * 100)
            test_accs.append(test_acc * 100)

        dev_accs = np.stack(dev_accs)
        test_accs = np.stack(test_accs)

        dev_acc, dev_std = dev_accs.mean(), dev_accs.std()
        test_acc, test_std = test_accs.mean(), test_accs.std()

        return dev_acc, dev_std, test_acc, test_std

def train_eval_whole_graph(args):
    trainer = ModelTrainer_whole_graph(args)
    trainer.train()
    trainer.writer.close()

def train_eval_batch_sampler(args):
    trainer = ModelTrainer_batch_sampler(args)
    trainer.train()
    trainer.writer.close()

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = build_args()

    if args.use_cfg:
        args = load_best_configs(args, "./configs/best_BGRL_configs.yaml")
    print(args)
    if not args.use_sampler:
        train_eval_whole_graph(args)
    else:
        train_eval_batch_sampler(args)
