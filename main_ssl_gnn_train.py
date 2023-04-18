import trainer_large
import trainer_small
from utils.utils import build_args, process_args


def train_eval(args):
    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        trainer = trainer_small.ModelTrainer(args)
        acc = trainer.train_eval()
        return acc
    elif args.dataset.startswith("ogbn"):
        trainer = trainer_large.ModelTrainer(args)
        acc = trainer.train_eval()
        return acc
    else:
        raise NotImplementedError(f"{args.dataset} is not supported yet!")


if __name__ == "__main__":
    args = build_args()
    args = process_args(args)
    if not args.no_verbose:
        print(args)
    train_eval(args)
