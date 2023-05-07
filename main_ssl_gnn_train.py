import trainer_large
import trainer_small
from utils.utils import build_args, process_args,Logger
import sys

def train_eval(args):

    if args.dataset.startswith("ogbn"):
        trainer = trainer_large.ModelTrainer(args)
        acc = trainer.train_eval()
        return acc
    elif args.dataset in ['cora', 'citeseer', 'pubmed','amazon_experimental_dataset']:
        trainer = trainer_small.ModelTrainer(args)
        acc = trainer.train_eval()
        return acc
    else:
        raise NotImplementedError(f"{args.dataset} is not supported yet!")


if __name__ == "__main__":
    args = build_args()
    args = process_args(args)
    logger = Logger(no_verbose=args.no_verbose)
    logger.set_log_path(filename=args.logging_path)
    sys.stdout = logger
    print(args)
    train_eval(args)
