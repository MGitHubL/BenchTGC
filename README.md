# BenchTGC

### Deep Temporal Graph Clustering: A Comprehensive benchmark and Datasets

This is the PyTorch version of BenchTGC. We want to provide you with as much usable code as possible.

If you find any problems, feel free to contact us: ```mengliuedu@163.com```.

## BenchTGC Datasets

You can download the datasets from [Data4TGC](https://github.com/MGitHubL/Data4TGC) and create "data" folder in the same directory as the "emb" and "code" folders.

## BenchTGC Framework

## Prepare

To run the code, you need download datasets first.

#### Pre-Training

In ```./code/pretrain/```, you need run the ```pretrain.py``` to generate pretrain embeddings.

Note that these embeddings are pre-trained embeddings, while the features in the dataset are positional encoding embeddings.

#### Training

You need create a folder for each dataset in ```./emb/``` to store generated node embeddings.

For example, after training with `School` dataset, the node embeddings will be stored in ```./emb/school/```


## Run

For each dataset, create a folder in ```emb``` folder with its corresponding name to store node embeddings, i.e., for arXivAI dataset, create ```./emb/arXivAI```.

For training, we give 5 improved methods, you can run them respectively.

All parameter settings have default values, you can adjust them.

## Test

For test, you have two ways:

(1) In the training process, we evaluate the clustering performance for each epoch. This evaluation is used for common-scale datasets, i.e., DBLP, Brain, Patent, and School.

(2) You can also run the ```clustering.py``` in the ```./code``` folder.

Note that the node embeddings in the ```./emb/school/school_ITREND.emb``` folder are just placeholders, you need to run the main code to generate them.

Note that the evaluation of the School dataset during training is not ideal, so we encourage the use of trained embeddings for clustering.


## Cite us

If you feel our work has been helpful, thank you for the citation.

```
@inproceedings{TGC_ML_ICLR,
  title={Deep Temporal Graph Clustering},
  author={Liu, Meng and Liu, Yue and Liang, Ke and Tu, Wenxuan and Wang, Siwei and Zhou, Sihang and Liu, Xinwang},
  booktitle={The 12th International Conference on Learning Representations},
  year={2024}
}
```
