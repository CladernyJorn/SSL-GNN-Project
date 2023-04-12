# SSL-GNN-Learning-Project
A coding project that integrates 4 recent SSL methods applied in GNN. 

- Supports **graphmae, grace, cca_ssg, bgrl** 

- Supports transductive node classification on  **cora, citeseer, pubmed** datasets

- Supports using **Saint Sampler** or pretraining on **full graph**

- Provides a simple and easy to use interface to **add other baseline methods**

- Provides **tuned parameter configs** for reproducing results on different baselines

  The reproduced results are as follows ( using config files in './configs/' and training on 1 3090GPU )

|          | Cora  | CiteSeer | PubMed |
| -------- | ----- | -------- | ------ |
| GraphMAE | 84.20 | 73.40    | 81.70  |
| Grace    | 81.80 | 70.20    | 81.40  |
| CCA-SSG  | 83.20 | 72.20    | 80.20  |
| BGRL     | 81.80 | 71.90    | 80.70  |

## Methods included

- **GraphMAE**: *Self-Supervised Masked Graph Autoencoders*

  ( https://arxiv.org/abs/2205.10803 )

- **Grace**: *Deep Graph Contrastive Representation Learning*

  ( https://arxiv.org/abs/2006.04131#)

- **BGRL**: *BOOTSTRAPPED REPRESENTATION LEARNING ON GRAPHS*

  ( https://openreview.net/forum?id=QrzVRAA49Ud )

- **CCA-SSG**: *From Canonical Correlation Analysis to Self-supervised Graph Neural Networks*

  ( https://proceedings.neurips.cc/paper/2021/hash/00ac8ed3b4327bdd4ebbebcb2ba10a00-Abstract.html )


## Dependencies

- Python >= 3.7
- [Pytorch](https://pytorch.org/) >= 1.9.0
- [dgl](https://www.dgl.ai/) >= 1.0.0 ( `dgl.nn.function.copy_src` is renamed as `dgl.nn.function.copy_u` )
- pyyaml == 5.4.1
- Numpy
- tqdm

## How to run?

### Example Commands

```shell
python main_ssl_gnn_train.py --model graphmae --dataset cora --encoder gat --decoder gat --device 0 --use_sampler --budget 500 --eval_nums 5 --num_iters 0

python main_ssl_gnn_train.py --model grace --dataset citeseer use_cfg

python main_ssl_gnn_train.py --model cca_ssg --dataset pubmed --encoder gcn --device 0 --use_sampler --budget 500 --eval_nums 5 --num_iters 0

python main_ssl_gnn_train.py --model bgrl --dataset cora --device 0 --use_cfg ----eval_nums 5 --no_verbose
```

### Important Arguments

Command line parameter used to control training ( for all 4 methods ):

- The `--model` argument should be one of [ **graphmae, grace, cca_ssg, bgrl** ].
- The `--dataset` argument should be one of [ **cora, citeseer, pubmed** ].
- The `--encoder`/ `--decoder` argument should be one of [ **gcn, gat, dotgat, gin, mlp** ].
- The `--device` argument should be an integer ( default -1 for 'cpu' ).
- The `--use_cfg` argument means using the configs to set training arguments (use `--use_cfg_path` to specify path).
- The `--eval_nums` argument indicates how often the evaluation is performed during pretraining.

If you want to use **Saint sampler** for mini-batch training on subgraph, you may add the arguments below. 

- ` --use_sampler`: use saint sampler for training instead of the default full graph training
- `--budget`: batch-size, which is the number of nodes in the subgraph sampled by the sampler at one time
- `--num_iters`:  The number of training iteration, if num_iters is 0, then use default $$ N(graph_{full})\times epochs/ budget$$



## How to add other baseline methods?

Follow the three steps below (do not modify *'main_ssl_gnn_train.py'* ):

1. add your **'model.py'** into **'./models/'**, in which your `nn.Module` class need to implement the following interfaces：
   - `model.__init__(self,...)`: Init parameters. You can use the basic gnn models provided in the *'./gnn_modules/'* ( use `setup_module()` to automatically select the encoder type )
   - `model.forward(self,g,x) -> loss`: calculate and return the final loss
     - g : (dgl.DGLGraph)  the input graph
     - x : (torch.tensor) feature of nodes
   - `model.embed(self,g,x) -> torch.tensor`: return the embedded representation calculated by encoder
     - g : (dgl.DGLGraph)  the input graph
     - x : (torch.tensor) feature of nodes
2. modify **`build_args()`** in *'utils/utils.py'* : add arguments of the new model
3. modify **`build_model(args)`** in *'models/\_\_init\_\_.py'*: use arguments in args to build your model 

## Directory description

A brief description of the code base structure

- `main_ssl_gnn_train.py`: Includes main functions for different methods ( using a trainer class to organize all training and evaluation functions ).
- `./models/`: *'.py'* models for different baseline models
- `./utils/` : Common functional tools
  - `dataset.py`: functions for preparing datasets and dataloaders
  - `augmentation.py`: functions for graph augmentations which are implemented with dgl
  - `utils.py`: common functions for training and evaluation ( like `build_args()`, `build_model(args)`)

- `./gnn_modules/`: common graph encoders implemented with dgl
  - including `gcn.py, gat.py, gin.py, dot_gat.py`, dot_gat is an equivalent implementations for gat
- `./configs/`: tuned model settings for different baselines on different dataset, use `--use_cfg` to load it 
- `Effect_of_Saint_Sampler.ipynb`: a presentation document that tests the effects of the Saint Sampler method on the cora dataset of GRACE and GraphMAE

## References

This repository is built according to the following articles: 

```ll
@inproceedings{hou2022graphmae,
  title={GraphMAE: Self-Supervised Masked Graph Autoencoders},
  author={Hou, Zhenyu and Liu, Xiao and Cen, Yukuo and Dong, Yuxiao and Yang, Hongxia and Wang, Chunjie and Tang, Jie},
  booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={594--604},
  year={2022}
}
@inproceedings{zhang2021canonical,
  title={From canonical correlation analysis to self-supervised graph neural networks},
  author={Zhang, Hengrui and Wu, Qitian and Yan, Junchi and Wipf, David and Philip, S Yu},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}
@inproceedings{Zhu:2020vf,
  author = {Zhu, Yanqiao and Xu, Yichen and Yu, Feng and Liu, Qiang and Wu, Shu and Wang, Liang},
  title = {{Deep Graph Contrastive Representation Learning}},
  booktitle = {ICML Workshop on Graph Representation Learning and Beyond},
  year = {2020},
  url = {http://arxiv.org/abs/2006.04131}
}
@inproceedings{thakoor2021bootstrapped,
  title={Bootstrapped representation learning on graphs},
  author={Thakoor, Shantanu and Tallec, Corentin and Azar, Mohammad Gheshlaghi and Munos, R{\'e}mi and Veli{\v{c}}kovi{\'c}, Petar and Valko, Michal},
  booktitle={ICLR 2021 Workshop on Geometrical and Topological Representation Learning},
  year={2021}
}
```

