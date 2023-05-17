# SSL-GNN-Project
A coding project that integrates several recent SSL methods applied in GNN. 

- Supports **graphmae2, graphmae, grace, cca_ssg, bgrl, ggd**
- Supports transductive node classification on  **cora, citeseer, pubmed, ogbn (arxiv, products, papers100M)** datasets
- Supported Sampling methods:
  - **Saint** sampler for pretraining

  - **ClusterGCN** sampler for pretraining

  - **K-hop** sampler for pretraining and evaluation

  - **LocalClustering** sampler for pretraining and evaluation

- Provides a simple and easy to use interface to **add other baseline methods**
- Provides **tuned parameter configs** for reproducing results on different baselines. Reproduced results for reference: using config files in `./configs/` and training on 1 3090GPU. (If not mentioned, `gat` is the default encoder.)

| full-graph |          Cora           |        CiteSeer        |         PubMed          |
| :--------: | :---------------------: | :--------------------: | :---------------------: |
| GraphMAE2  |     $83.90\pm 0.58$     |    $73.28\pm 0.46$     |     $81.47\pm0.48$      |
|  GraphMAE  |     $84.06\pm0.60$      |     $73.11\pm0.33$     |     $80.99\pm 0.53$     |
|   Grace    | $82.53\pm0.62$ (`GCN`)  |     $69.66\pm1.21$     | $81.13\pm0.48$ (`GCN`)  |
|  CCA-SSG   | $83.77\pm 0.57$ (`GCN`) | $72.73\pm1.01$ (`GCN`) | $80.39\pm0.64$ (`GCN`)  |
|    BGRL    |     $81.23\pm 0.50$     |    $71.44\pm 0.76$     |     $79.10\pm0.42$      |
|    GGD     | $83.09\pm0.63$ (`GCN`)  | $73.03\pm0.60$ (`GCN`) | $80.29\pm 0.43$ (`GCN`) |

| sampling  | ogbn-arxiv (Saint) | ogbn-arxiv (ClusterGCN) | ogbn-arxiv (ShallowKhop) | ogbn-arxiv (LocalClustering) | ogbn-arxiv (Random-Init) |
| :-------: | :----------------: | :---------------------: | :----------------------: | :--------------------------: | :----------------------: |
| GraphMAE2 |   $70.43\pm0.09$   |     $69.77\pm0.21$      |      $70.96\pm0.10$      |        $71.79\pm0.12$        |      $70.43\pm0.21$      |
| GraphMAE  |   $70.54\pm0.20$   |     $71.52\pm0.19$      |      $70.77\pm0.35$      |        $70.42\pm0.11$        |      $70.43\pm0.21$      |
|   Grace   |   $67.65\pm0.16$   |     $65.58\pm0.20$      |      $67.84\pm0.15$      |        $71.03\pm0.13$        |      $70.43\pm0.21$      |
|  CCA-SSG  |                    |                         |                          |        $58.19\pm0.04$        |      $70.43\pm0.21$      |
|   BGRL    |                    |                         |                          |        $64.64\pm0.02$        |      $70.43\pm0.21$      |
|    GGD    |                    |                         |                          |        $46.20\pm0.14$        |      $65.86\pm0.01$      |

| sampling  | ogbn-products (LocalClustering) | ogbn-products (Random-Init) |
| :-------: | :-----------------------------: | :-------------------------: |
| GraphMAE2 |         $81.02\pm0.07$          |       $75.96\pm0.21$        |
| GraphMAE  |         $79.68\pm0.09$          |       $75.96\pm0.21$        |
|   Grace   |         $80.61\pm0.46$          |       $75.96\pm0.21$        |
|  CCA-SSG  |         $73.56\pm0.29$          |       $75.96\pm0.21$        |
|   BGRL    |         $67.16\pm0.07$          |       $75.96\pm0.21$        |
|    GGD    |         $73.41\pm0.31$          |       $76.51\pm0.02$        |

In order to make a fair comparison, all the results in the two tables (ogbn datasets) above are evaluated using Local Clustering sampling.

## Methods included

- **GraphMAE2**: A Decoding-Enhanced Masked Self-Supervised Graph Learner

  https://arxiv.org/abs/2304.04779

- **GraphMAE**: *Self-Supervised Masked Graph Autoencoders*

  ( https://arxiv.org/abs/2205.10803 )

- **Grace**: *Deep Graph Contrastive Representation Learning*

  ( https://arxiv.org/abs/2006.04131#)

- **BGRL**: *BOOTSTRAPPED REPRESENTATION LEARNING ON GRAPHS*

  ( https://openreview.net/forum?id=QrzVRAA49Ud )

- **CCA-SSG**: *From Canonical Correlation Analysis to Self-supervised Graph Neural Networks*

  ( https://proceedings.neurips.cc/paper/2021/hash/00ac8ed3b4327bdd4ebbebcb2ba10a00-Abstract.html )
  
- **GGD**: Rethinking and Scaling Up Graph Contrastive Learning: An Extremely Efficient Approach with Group Discrimination

  （ https://arxiv.org/abs/2206.01535 ）


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
python main_ssl_gnn_train.py --model graphmae --dataset cora --encoder gat --decoder gat --device 0 --eval_nums 5

python main_ssl_gnn_train.py --model grace --dataset citeseer use_cfg

python main_ssl_gnn_train.py --model graphmae2 --dataset ogbn-arxiv --use_cfg --device 0 --pretrain_sampling_method lc --eval_sampling_method lc

python main_ssl_gnn_train.py --model bgrl --dataset ogbn-arxiv --use_cfg --device 0 --pretrain_sampling_method saint --eval_sampling_method khop
```

### Important Arguments

Command line parameter used to control training ( for all 4 methods ):

- The `--model` argument should be one of [ **graphmae2, graphmae, grace, cca_ssg, bgrl** ].
- The `--dataset` argument should be one of [ **cora, citeseer, pubmed, ogbn-arxiv, ogbn-products, ogbn-papers100M** ].
- The `--encoder`/ `--decoder` argument should be one of [ **gcn, gat, dotgat, gin, mlp** ].
- The `--device` argument should be an integer ( default -1 for 'cpu' ).
- The `--use_cfg` argument means using the configs to set training arguments (use `--use_cfg_path` to specify path).

#### **For Large scale graphs** 

For ogbn-arxiv, ogbn-products, ogbn-papers100M, pretraining and evaluation can only be done with mini-batch samplers.

1. If you want to use **Saint sampler** for mini-batch pretraining on subgraph, you may add the arguments below. 

   - ` --pretrain_sampling_method saint`: use saint sampler for training instead of the default full graph training

   - `--saint_budget`: batch-size, which is the number of nodes in the subgraph sampled by the sampler at one time

2. If you want to use **ClusterGCN sampler** for mini-batch pretraining on subgraph, you may add the arguments below. 

   - ` --pretrain_sampling_method clustergcn`

   - `--cluster_gcn_num_parts`: partition num of nodes in ClusterGCN Algorithms
   - `--cluster_gcn_batch_size`:  denote how many partitions will be sampled per batch

   Notice that Saint and ClusterGCN sampler can only be used for pretraining sampling method, you may also set eval samplers by using 

   - `--eval_sampling_method`: should be one of [**lc, khop**], by default use **lc**.

3. If you want to use **K-hop sampler** for mini-batch pretraining or evaluation, you may add the arguments below. 

   - ` --pretrain_sampling_method khop`

   - `--khop_fanouts`: number of neighbors for each gnn layers

4. If you want to use **Local Clustering sampler** for mini-batch pretraining or evaluation, you may:

   1. Install [localclustering](https://github.com/kfoynt/LocalGraphClustering) 

   2. Generate local clusters before pretraining

      - `python ./datasets/localclustering.py --dataset <your_dataset> --data_dir <path_to_data>`

      By default, the program will load dataset from `./data/datasets` and save the generated local clusters to `./lc_ego_graphs`. 

   3. Add the arguments below:

      - ` --pretrain_sampling_method lc`

      - `--ego_graph_file_path`: path of ego graph file that is generated in step 2



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
- `./datasets/`: functions for preparing datasets from different graph data
- `./utils/` : Common functional tools
  - `load_data.py`: functions for preparing dataloaders
  - `augmentation.py`: functions for graph augmentations which are implemented with dgl
  - `utils.py`: common functions for training and evaluation ( like `build_args()`, `build_model(args)`)
  - `localclustering.py`:  generate ego graph files for Local Clustering sampler
- `./gnn_modules/`: common graph encoders implemented with dgl
  - including `gcn.py, gat.py, gin.py, dot_gat.py`, dot_gat is an equivalent implementations for gat
- `./configs/`: tuned model settings for different baselines on different dataset, use `--use_cfg` to load it 

## References

This repository is built according to the following articles: 

```latex
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
@inproceedings{hou2023graphmae2,
  title={GraphMAE2: A Decoding-Enhanced Masked Self-Supervised Graph Learner},
  author={Zhenyu Hou, Yufei He, Yukuo Cen, Xiao Liu, Yuxiao Dong, Evgeny Kharlamov, Jie Tang},
  booktitle={Proceedings of the ACM Web Conference 2023 (WWW’23)},
  year={2023}
}
@inproceeding{zheng2022rethinking,
  title={Rethinking and Scaling Up Graph Contrastive Learning: An Extremely Efficient Approach with Group Discrimination},
  author={Zheng, Yizhen and Pan, Shirui and Lee, Vincent Cs and Zheng, Yu and Yu, Philip S},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```

