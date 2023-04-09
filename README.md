# SSL-GNN-Learning-Project
A coding project that integrates 4 recent SSL methods applied in GNN. 

The code base also supports using Saint Sampler for pretraining on a small subgraph. 

(See more details in *GraphSAINT: Graph Sampling Based Inductive Learning Method* https://arxiv.org/abs/1907.04931v4)

## Methods included

- **GraphMAE**: *Self-Supervised Masked Graph Autoencoders*

  ( https://arxiv.org/abs/2205.10803 )

  ![image-20230401154756308](https://github.com/CladernyJorn/SSL-GNN-Learning-Project/blob/main/README.assets/image-20230401154756308.png)

- **Grace**: *Deep Graph Contrastive Representation Learning*

  ( https://arxiv.org/abs/2006.04131#)

  ![image-20230409113629844](https://github.com/CladernyJorn/SSL-GNN-Learning-Project/blob/main/\README.assets\image-20230409113629844.png)

- **BGRL**: *BOOTSTRAPPED REPRESENTATION LEARNING ON GRAPHS*

  ( https://openreview.net/forum?id=QrzVRAA49Ud )

  ![model](https://github.com/CladernyJorn/SSL-GNN-Learning-Project/blob/main/\README.assets\model.PNG)

- **CCA-SSG**: *From Canonical Correlation Analysis to Self-supervised Graph Neural Networks*

  ( https://proceedings.neurips.cc/paper/2021/hash/00ac8ed3b4327bdd4ebbebcb2ba10a00-Abstract.html )

  ![image-20230409113744297](https://github.com/CladernyJorn/SSL-GNN-Learning-Project/blob/main/\README.assets\image-20230409113744297.png)

## Dependencies

- Python >= 3.7
- [Pytorch](https://pytorch.org/) >= 1.9.0
- [dgl](https://www.dgl.ai/) >= 1.0.0 ( `dgl.nn.function.copy_src` is renamed as `dgl.nn.function.copy_u` )
- pyyaml == 5.4.1
- Scikit-learn
- Numpy
- tqdm

## Usage

### Example Commands

```shell
python main_GraphMAE.py --dataset cora --encoder gat --decoder gat --seeds 0 --device 0 --use_cfg
python main_Grace.py --dataset cora --encoder gin --use_cfg --seeds 0 --device 0
python main_CCA_SSG.py --dataset cora --encoder gcn --seeds 0 --device 0 --use_cfg 
python main_BGRL.py --dataset cora --encoder gat --device 0 --lr 0.0001 --epochs 20 --use_cfg
# --use_sampler --budget 500 --num_iters 1000
```

### Arguments

Command line parameter used to control training ( for all 4 methods ):

- The `--dataset` argument should be one of [ **cora, citeseer, pubmed** ].
- The `--encoder`/ `--decoder` argument should be one of [ **gcn, gat, dotgat, gin, mlp** ].
- The `--device` argument should be an integer ( default -1 for 'cpu' ).
- The `--use_cfg` argument means using the configs to set training arguments.

If you want to use **Saint sampler** for mini-batch training on subgraph, you may add the arguments below. 

- ` --use_sampler`: use saint sampler for training instead of the default full graph training
- `--budget`: batch-size, which is the number of nodes in the subgraph sampled by the sampler at one time
- `--num_iters`:  The number of training iteration, if num_iters is 0, then use default $ N(graph_{full})\times epochs/ budget$

## Directory description

A brief description of the code base structure

- `main_BGRL.py, main_CCA_SSG.py , main_Grace.py, main_GraphMAE.py`: Includes master functions for different methods
- `./bgrl/`,  `./cca_ssg/`, `./grace/`, `./graphmae/`: models for different methods and necessary auxiliary functions
- `./utils/` : Common functional tools
  - `dataset_utils.py`: functions for preparing datasets and dataloaders
  - `graph_augmentation.py`: functions for graph augmentations which are implemented with dgl
  - `train_evaluation_utils.py`: functions for evaluation ( linear evaluation for nodes classification )

- `./gnn_modules/`: common graph encoders implemented with dgl
  - including `gcn.py, gat.py, gin.py, dot_gat.py`, dot_gat is an equivalent implementations for gat
- `./configs/`: training settings for different methods ,use `--use_cfg` to load it 
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

