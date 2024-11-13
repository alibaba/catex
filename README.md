# Category-Extensible Out-of-Distribution Detection

This repo is the official implementation of [CATegory-EXtensible Out-of-Distribution Detection via Hierarchical Context Descriptions](https://arxiv.org/abs/2407.16725) (CATEX)

## How to Install
This code is built on top of the awesome [CoOp](https://github.com/KaiyangZhou/CoOp) framework, and you need to install the `dassl` environment first (this has already prepared in this repo). 

```bash
cd Dassl.pytorch
pip install -r requirements.txt
python setup.py develop

cd ..
pip install -r requirements.txt
```

After that, run `pip install -r requirements.txt` under `catex/` to install a few more packages required by [CLIP](https://github.com/openai/CLIP) (this should be done when `dassl` is activated). Then, you are ready to go.

## How to Run

First, follow the [DATASETS.md](DATASETS.md) to install the datasets.

Then, check the [CATEX.md](CATEX.md) to see the detailed instructions on how to run the code to reproduce the results.

## Released models
- [ ] We are working on it. Please stay tuned.

## Citation
If you find this repo or paper useful in your research, please kindly star this repo and cite this paper:

```
@inproceedings{liu2023category,
  title={Category-Extensible Out-of-Distribution Detection via Hierarchical Context Descriptions},
  author={Liu, Kai and Fu, Zhihang and Chen, Chao and Jin, Sheng and Chen, Ze and Mingyuan Tao and Jiang, Rongxin and Ye, Jieping},
  booktitle={37th Conference on Neural Information Processing Systems (NeurIPS 2023)},
}
```
