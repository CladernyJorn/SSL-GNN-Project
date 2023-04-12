import copy
import numpy as np
from tqdm import tqdm
import torch

# models
from models import build_model

# self-made utils
from utils.dataset import load_dataset, load_dataloader
from utils.utils import (
    create_optimizer,
    set_random_seed,
    build_args,
    process_args,
    create_scheduler,
    LogisticRegression,
    accuracy
)


class ModelTrainer:

    def __init__(self, args):
        self._args = args
        self._init()

    def _init(self):
        args = self._args
        self._verbose = not args.no_verbose
        self._device = args.device if args.device >= 0 else "cpu"
        self._use_sampler = args.use_sampler
        if not self._use_sampler:
            self._graph, (self.num_features, self.num_classes) = load_dataset(args.dataset)
            self._eval_steps = args.eval_steps if args.eval_nums == 0 else args.max_epoch // args.eval_nums
        else:
            self._graph, self._pretrain_dataloader, num_iters, (self.num_features, self.num_classes) = load_dataloader(
                args.dataset, args.budget, args.num_iters, args.max_epoch, args.device)
            self._eval_steps = args.eval_steps if args.eval_nums == 0 else num_iters // args.eval_nums
            self._args.max_epoch=num_iters
        self._args.num_features=self.num_features
        if self._verbose:
            print(f"Data: {self._graph}")

    def train_eval(self):
        args = self._args
        test_list = []
        estp_test_list = []
        dev_list = []
        estp_dev_list = []
        for i, seed in enumerate(args.seeds):
            if self._verbose:
                print(f"####### Run{i} for seed {seed} #######")
            set_random_seed(seed)
            self.model = build_model(args)
            self.optimizer = create_optimizer(args.optimizer, self.model, args.lr, args.weight_decay)
            self.scheduler = None
            if args.scheduler:
                self.scheduler = create_scheduler(args, self.optimizer)

            # need to pretrain
            if not args.load_model:
                # get initial results
                self.infer_embeddings()
                dev_acc, estp_dev_acc, test_acc, estp_test_acc = self.evaluate()
                if self._verbose:
                    print("initial test acc: {:.4f}".format(test_acc))
                self.pretrain()
                model = self.model.cpu()
                if args.save_model:
                    print(f"Saveing model to {args.save_model_path}")
                    torch.save(model.state_dict(), args.save_model_path)

            # no need to pretrain, eval directly
            if args.load_model:
                print(f"Loading model from {args.load_model_path}")
                self.model.load_state_dict(torch.load(args.load_model_path))

            # pretrain done, do evaluation
            self.infer_embeddings()
            dev_acc, estp_dev_acc, test_acc, estp_test_acc = self.evaluate()
            dev_list.append(dev_acc)
            estp_dev_list.append(estp_dev_acc)
            test_list.append(test_acc)
            estp_test_list.append(estp_test_acc)

        final_test_acc, final_test_acc_std = np.mean(test_list), np.std(test_list)
        estp_test_acc, estp_test_acc_std = np.mean(estp_test_list), np.std(estp_test_list)
        final_dev_acc, final_dev_acc_std = np.mean(dev_list), np.std(dev_list)
        estp_dev_acc, estp_dev_acc_std = np.mean(estp_dev_list), np.std(estp_dev_list)
        print(f"# final-dev-acc: {final_dev_acc:.4f}±{final_dev_acc_std:.4f}", end="")
        print(f"# early-stopping-dev-acc: {estp_dev_acc:.4f}±{estp_dev_acc_std:.4f}")
        print(f"# final-test-acc: {final_test_acc:.4f}±{final_test_acc_std:.4f}", end="")
        print(f"# early-stopping-test-acc: {estp_test_acc:.4f}±{estp_test_acc_std:.4f}")

        return final_test_acc

    def pretrain(self):
        args = self._args
        if self._verbose:
            print(f"\n--- Start pretraining {args.model} model on {args.dataset} ", end="")
            if self._use_sampler:
                print(f"using Saint ({args.budget} nodes per epoch)! ---")
            else:
                print(f"on full graph ({self._graph.number_of_nodes()} nodes per epoch)! ---")

        epoch_iter = tqdm(self._pretrain_dataloader) if self._use_sampler else tqdm(range(args.max_epoch))
        iters = enumerate(epoch_iter) if self._use_sampler else epoch_iter
        dev_best = 0
        test_best = 0
        self.model.to(self._device)
        for items in iters:
            if self._use_sampler:
                epoch, graph = items
            else:
                epoch = items
                graph = self._graph
            self.model.train()
            graph=graph.to(self._device)
            in_feature = graph.ndata['feat'].to(self._device)
            loss = self.model(graph, in_feature)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            if args.model == "bgrl":
                self.model.update_moving_average()  # bgrl uses EMA updating strategy

            epoch_iter.set_description(f"# Epochs {epoch}: train_loss: {loss.item():.4f}")

            if self._verbose:
                if (epoch + 1) % self._eval_steps == 0:
                    self.infer_embeddings()
                    dev_acc, estp_dev_acc, test_acc, estp_test_acc = self.evaluate()
                    if dev_best < dev_acc:
                        dev_best = dev_acc
                        test_best = test_acc
                    print("validation: {:.4f}, test: {:.4f} \n".format(dev_acc, test_acc))

        if self._verbose:
            print(f"validation: {dev_best:.4f}, test: {test_best:.4f}")

    def infer_embeddings(self):  # preparing embeddings and labels
        self.model.eval()
        self.model.to(self._device)
        self._labels = self._graph.ndata['label'].detach().to(self._device)
        self._graph = self._graph.remove_self_loop().add_self_loop()
        self._embeddings = self.model.embed(self._graph.to(self._device), self._graph.ndata['feat'].to(self._device))
        self._embeddings = self._embeddings.detach()

    def evaluate(self, mute=True):
        args = self._args
        in_feat = self._embeddings.shape[1]
        classifier = LogisticRegression(in_feat, self.num_classes).to(self._device)
        if not mute:
            num_finetune_params = [p.numel() for p in classifier.parameters() if p.requires_grad]
            print(f"num parameters for finetuning: {sum(num_finetune_params)}")
        optimizer = create_optimizer("adam", classifier, args.lr_f, args.weight_decay_f)
        criterion = torch.nn.CrossEntropyLoss()
        self._train_mask = self._graph.ndata["train_mask"]
        self._dev_mask = self._graph.ndata["val_mask"]
        self._test_mask = self._graph.ndata["test_mask"]
        best_val_acc = 0
        best_val_epoch = 0
        best_classifier = None
        for epoch in range(args.max_epoch_f):
            classifier.train()
            classifier.to(self._device)
            logits = classifier(self._embeddings[self._train_mask].to(self._device))
            loss = criterion(logits, self._labels[self._train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                classifier.eval()
                pred = classifier(self._embeddings)
                val_acc = accuracy(pred[self._dev_mask], self._labels[self._dev_mask])
                val_loss = criterion(pred[self._dev_mask], self._labels[self._dev_mask])
                test_acc = accuracy(pred[self._test_mask], self._labels[self._test_mask])
                test_loss = criterion(pred[self._test_mask], self._labels[self._test_mask])
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_val_epoch = epoch
                best_classifier = copy.deepcopy(classifier)
            if not mute:
                print(
                    f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_acc:{val_acc}, test_loss:{test_loss.item(): .4f}, test_acc:{test_acc: .4f}")
        best_classifier.eval()
        with torch.no_grad():
            pred = best_classifier(self._embeddings)
            estp_test_acc = accuracy(pred[self._test_mask], self._labels[self._test_mask])
        if self._verbose:
            print(
                f"--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")
        return val_acc, best_val_acc, test_acc, estp_test_acc

def train_eval(args):
    trainer = ModelTrainer(args)
    acc=trainer.train_eval()
    return acc

if __name__ == "__main__":
    args = build_args()
    args = process_args(args)
    if not args.no_verbose:
        print(args)
    train_eval(args)
