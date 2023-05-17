from utils.utils import build_args, process_args, Logger
import sys
import copy
import numpy as np
from tqdm import tqdm
import torch
from models import build_model
# self-made utils
from utils.load_data import load_dataset
from utils.utils import (
    create_optimizer,
    create_optimizer_learn_feature,
    set_random_seed,
    create_scheduler,
    accuracy
)


class ModelTrainer:

    def __init__(self, args):
        self._args = args
        self._device = args.device if args.device >= 0 else "cpu"
        self._use_sampler = args.use_sampler

    def train_eval(self):
        args = self._args
        self._graph, (self.num_features, self.num_class) = load_dataset(args.dataset)
        self.num_features = self.num_features
        self._eval_steps = args.eval_steps if args.eval_nums == 0 else args.max_epoch // args.eval_nums
        self._args.num_features = self.num_features
        print(f"{self._graph}")
        if args.classification:
            self._targets = self._graph.ndata['label'].detach().to(self._device)
            self._args.num_class = self.num_class
        elif args.regression:
            self._targets = self._graph.ndata['price'].detach().to(self._device)
            self._args.num_class = 1
        else:
            raise NotImplementedError("use --classification or --regression")
        self._train_mask = self._graph.ndata["train_mask"]
        self._dev_mask = self._graph.ndata["val_mask"]
        self._test_mask = self._graph.ndata["test_mask"]
        self.in_feature = self._graph.ndata['feat'].to(self._device)
        print(f"\n--- Start pretraining {args.model} model on {args.dataset} ---")
        test_list = []
        dev_list = []
        for i, seed in enumerate(args.seeds):
            print(f"####### Run{i} for seed {seed} #######")
            set_random_seed(seed)
            self.model = build_model(self._args)
            if not args.load_model:
                if args.eval_first:
                    print("Initial Test:")
                    self.evaluate()
                best_model = self.train(train_embeddings=False)
                model = best_model.cpu()
                self.model = best_model
                if args.save_model:
                    print(f"Saveing model to {args.save_model_path}")
                    torch.save(model.state_dict(), args.save_model_path)
            # no need to pretrain, eval directly
            if args.load_model:
                print(f"Loading model from {args.load_model_path}")
                self.model.load_state_dict(torch.load(args.load_model_path))
            train_result, dev_result, test_result = self.evaluate(mute=False, draw_acc_degree=True)
            dev_list.append(dev_result)
            test_list.append(test_result)
        final_test_acc, final_test_acc_std = np.mean(test_list), np.std(test_list)
        final_dev_acc, final_dev_acc_std = np.mean(dev_list), np.std(dev_list)
        print(f"# {'Classification Acc' if args.classification else 'Regression Loss'}:")
        print(f"# final-dev-acc: {final_dev_acc:.4f}±{final_dev_acc_std:.4f}")
        print(f"# final-test-acc: {final_test_acc:.4f}±{final_test_acc_std:.4f}")
        return final_test_acc

    def train(self, train_embeddings=False):
        args = self._args
        print(f"\n--- Start training {args.model} model on Amazon Experimental Dataset ---")
        if train_embeddings:
            self.in_feature.requires_grad = True
            self.optimizer = create_optimizer_learn_feature(args.optimizer, self.model, args.lr, args.weight_decay,
                                                            self.in_feature)
        else:
            self.optimizer = create_optimizer(args.optimizer, self.model, args.lr, args.weight_decay)
        self.in_feature.to(self._device)
        self.scheduler = None
        if args.scheduler:
            self.scheduler = create_scheduler(args, self.optimizer)
        epoch_iter = tqdm(range(args.max_epoch)) if not args.no_verbose else range(args.max_epoch)
        criterion = self.get_loss_func()
        self.model.to(self._device)
        dev_best = 0
        test_best = 0
        best_model = None
        for epoch in epoch_iter:
            graph = self._graph
            self.model.train()
            graph = graph.to(self._device)
            logits = self.model(graph, self.in_feature[self._train_mask])
            loss = criterion(logits, self._targets[self._train_mask])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            train_acc, dev_acc, test_acc = self.evaluate(mute=True)
            if dev_best < dev_acc:
                dev_best = dev_acc
                test_best = test_acc
                best_model = copy.deepcopy(self.model)
            if not args.no_verbose:
                epoch_iter.set_description(
                    f"# Epochs {epoch}: train_loss: {loss.item():.4f}， val_acc:{dev_acc:.4f}, test_acc:{test_acc:.4f}")
        print(f"validation: {dev_best:.4f}, test: {test_best:.4f}")
        return best_model

    def evaluate(self, mute=True, draw_acc_degree=False):
        model = self.model.to(self._device)
        if not mute:
            num_finetune_params = [p.numel() for p in model.parameters() if p.requires_grad]
            print(f"num parameters for finetuning: {sum(num_finetune_params)}")
        model.eval()
        graph = self._graph.to(self._device)
        # in_feature = graph.ndata['feat'].to(self._device)
        in_feature = self.in_feature.to(self._device)
        pred = model(graph, in_feature)
        criterion = self.get_loss_func()
        val_loss = criterion(pred[self._dev_mask], self._targets[self._dev_mask])
        test_loss = criterion(pred[self._test_mask], self._targets[self._test_mask])
        if args.classification:
            train_acc = accuracy(pred[self._train_mask], self._targets[self._train_mask])
            val_acc = accuracy(pred[self._dev_mask], self._targets[self._dev_mask])
            test_acc = accuracy(pred[self._test_mask], self._targets[self._test_mask])
            if draw_acc_degree:
                import matplotlib.pyplot as plt
                mask = self._dev_mask + self._test_mask
                y_true = self._targets[mask].squeeze().long()
                preds = pred[mask].max(1)[1].type_as(y_true)
                correct = preds.eq(y_true).double().cpu().numpy()
                degrees = ((graph.in_degrees() + graph.out_degrees()) / 2)[mask].int().cpu().numpy()
                degrees[degrees > 100] = 100
                print(f"Max-degree:{np.max(degrees)} Min-degree:{np.min(degrees)} Average-degree:{np.mean(degrees)}")

                correct_degree = {}
                for i, degree in enumerate(degrees):
                    if degree not in correct_degree.keys():
                        correct_degree[degree] = [correct[i]]
                    else:
                        correct_degree[degree].append(correct[i])
                count_degree = np.zeros(np.max(degrees) + 1)
                count_degree_correct = np.zeros(np.max(degrees) + 1)
                count_degree_acc = np.zeros(np.max(degrees) + 1)
                for degree, accs in correct_degree.items():
                    count_degree[degree] = len(accs)
                    count_degree_correct[degree] = np.array(accs).sum()
                    count_degree_acc[degree] = count_degree_correct[degree] / len(accs) if len(accs) != 0 else 0
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
                x = np.arange(np.max(degrees) + 1)
                ax1.bar(x=x, height=count_degree, label='Total samples', color='Coral', alpha=0.8)
                ax1.bar(x=x, height=count_degree_correct, label='Correct samples', color='LemonChiffon', alpha=0.8)
                ax1.legend(loc="upper left")
                # 设置标题
                ax1.set(xlabel='Degrees', ylabel='Number of nodes', title='Degree-Acc')
                # 画折线图
                ax2.plot(np.where(count_degree_acc > 0)[0], count_degree_acc[count_degree_acc > 0], "r", marker='.',
                         c='r', ms=5, linewidth='1', label="Acc")
                # 显示数字
                # for a, b in zip(x, count_degree_acc):
                #     plt.text(a, b, b, ha='center', va='bottom', fontsize=8)
                # 在右侧显示图例
                ax2.set(xlabel='Degrees', ylabel='Acc', title='Degree-Acc')
                fig.savefig("acc-degree.png")
                plt.show()
            return train_acc, val_acc, test_acc
        elif args.regression:
            train_loss = criterion(pred[self._train_mask], self._targets[self._train_mask])
            print(
                f"---train_loss: {train_loss.item(): .4f}, val_loss: {val_loss.item(): .4f}, test_loss: {test_loss.item()}---")
            return train_loss.item(), val_loss.item(), test_loss.item()

    def get_loss_func(self):
        args = self._args
        if args.classification:
            return torch.nn.CrossEntropyLoss()
        elif args.regression:
            return torch.nn.L1Loss()


def train_eval(args):
    trainer = ModelTrainer(args)
    acc = trainer.train_eval()
    return acc


if __name__ == "__main__":
    logger = Logger()
    # for wd in [0.1,0.05,0.01,0.005,0.001,0.0005,0.0001,5e-5,1e-5,1-6]:
    args = build_args()
    logger.set_log_path(filename=f"logs/{args.dataset}_{args.model}.log")
    sys.stdout = logger
    args = process_args(args)
    print(args)
    train_eval(args)
