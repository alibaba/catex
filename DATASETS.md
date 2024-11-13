# How to install datasets

We suggest putting all datasets under the same folder (say `$DATA`, defauled as `./data`) to ease management and following the instructions below to organize datasets to avoid modifying the source code. The file structure looks like

```
$DATA/
|–– imagenet/
|–– imagenet-a/
|–– imagenet-r/
|–– imagenet-sketch/
|–– imagenet100-MCM/
|–– imagenet100-NEW/
|–– imagenet200-UNION/
|–– imagenet-21k/
|–– LargeOOD/
```

If you have some datasets already installed somewhere else, you can create symbolic links in `$DATA/dataset_name` that point to the original data to avoid duplicate download.

Datasets list:
- [ImageNet](#imagenet)
- [ImageNet-A](#imagenet-a)
- [ImageNet-R](#imagenet-r)
- [ImageNet-Sketch](#imagenet-sketch)
- [ImageNet100-MCM](#imagenet100-mcm)
- [ImageNet100-NEW](#imagenet100-new)
- [ImageNet200-UNION](#imagenet200-union)
- [ImageNet21K](#imagenet21k)
- [LargeOOD](#largeood)

The instructions to prepare each dataset are detailed below. To ensure reproducibility and fair comparison for future work, we provide fixed train/val/test splits for all datasets except ImageNet where the validation set is used as test set. The fixed splits are either from the original datasets (if available) or created by us.

### ImageNet
- Create a folder named `imagenet/` under `$DATA`.
- Create `images/` under `imagenet/`.
- Download the dataset from the [official website](https://image-net.org/index.php) and extract the training and validation sets to `$DATA/imagenet/images`. The directory structure should look like
```
imagenet/
|–– images/
|   |–– train/ # contains 1,000 folders like n01440764, n01443537, etc.
|   |–– val/
```
- If you had downloaded the ImageNet dataset before, you can create symbolic links to map the training and validation sets to `$DATA/imagenet/images`.
- Download the `classnames.txt` to `$DATA/imagenet/` from this [link](https://drive.google.com/file/d/1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF/view?usp=sharing). The class names are copied from [CLIP](https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb).

### ImageNet-A
- Create a folder named `imagenet-adversarial/` under `$DATA`.
- Download the dataset from https://github.com/hendrycks/natural-adv-examples and extract it to `$DATA/imagenet-adversarial/`.
- Copy `$DATA/imagenet/classnames.txt` to `$DATA/imagenet-adversarial/`.

The directory structure should look like
```
imagenet-adversarial/
|–– imagenet-a/ # contains 200 folders whose names have the format of n*
|–– classnames.txt
```

### ImageNet-R
- Create a folder named `imagenet-rendition/` under `$DATA`.
- Download the dataset from https://github.com/hendrycks/imagenet-r and extract it to `$DATA/imagenet-rendition/`.
- Copy `$DATA/imagenet/classnames.txt` to `$DATA/imagenet-rendition/`.

The directory structure should look like
```
imagenet-rendition/
|–– imagenet-r/ # contains 200 folders whose names have the format of n*
|–– classnames.txt
```

### ImageNet-Sketch
- Download the dataset from https://github.com/HaohanWang/ImageNet-Sketch.
- Extract the dataset to `$DATA/imagenet-sketch`.
- Copy `$DATA/imagenet/classnames.txt` to `$DATA/imagenet-sketch/`.

The directory structure should look like
```
imagenet-sketch/
|–– images/ # contains 1,000 folders whose names have the format of n*
|–– classnames.txt
```

### ImageNet100-MCM
- The folder named `imagenet100-MCM/` has already been created under `$DATA` (`./data`).
- The subfolder `images/` has also been created under `imagenet-MCM/`, where the image files are symbolic links to the `imagenet` folder.
- The `classnames.txt` is adopted from [MCM](https://github.com/deeplearning-wisc/MCM/blob/main/data/ImageNet100/class_list.txt).

The directory structure should look like
```
imagenet100-MCM/
|–– images/
|   |–– train/ # contains 100 folders like n01498041, n01518878, etc.
|   |–– val/
|–– classnames.txt
```

### ImageNet100-NEW
- The folder named `imagenet100-NEW/` has already been created under `$DATA` (`./data`).
- The subfolder `images/` has also been created under `imagenet-NEW/`, where the image files are symbolic links to the `imagenet` folder.
- The `classnames.txt` is randomly sampled from the original `imagenet` categories, without overlapping against `imagenet100-MCM/classnames.txt`.

The directory structure should look like
```
imagenet100-NEW/
|–– images/
|   |–– train/ # contains 100 folders like n01629819, n01631663, etc.
|   |–– val/
|–– classnames.txt
```

### ImageNet200-UNION
- The folder named `imagenet200-UNION/` has already been created under `$DATA` (`./data`).
- The subfolder `images/` has also been created under `imagenet200-UNION/`, where the image files are symbolic links to the `imagenet` folder.
- The `classnames.txt` is merged from `imagenet100-MCM/classnames.txt` and `imagenet100-NEW/classnames.txt`. 

The directory structure should look like
```
imagenet200-UNION/
|–– images/
|   |–– train/ # contains 200 folders like n01498041, n01629819, etc.
|   |–– val/
|–– classnames.txt
```

### ImageNet21K
- The folder named `imagenet21k/` has already been created under `$DATA` (`./data`).
- Download the dataset witg the provided `imagenet21k/download.py` script and extract the images to `$DATA/imagenet/images`.
```bash
cd $DATA/imagenet21k
python download.py
```

The directory structure should look like
```
imagenet21k/
|–– images/ # contains 21,842 folders like n00004475, n00005787, etc.
|–– ImageNet21K_label.txt
|–– ImageNet21K_name.txt
```

### LargeOOD
- For large-scale ID (e.g. ImageNet-100, ImageNet), following [NPOS](https://github.com/deeplearning-wisc/npos) we use the curated 4 OOD datasets from [iNaturalist](https://arxiv.org/pdf/1707.06642.pdf), [SUN](https://vision.princeton.edu/projects/2010/SUN/paper.pdf), [Places](http://places2.csail.mit.edu/PAMI_places.pdf), and [Textures](https://arxiv.org/pdf/1311.3618.pdf), and de-duplicated concepts overlapped with ImageNet-1k. The datasets are created by  [Huang et al., 2021](https://github.com/deeplearning-wisc/large_scale_ood) .
- Create a folder named `LargeOOD/` under `$DATA`.
- Downloaded the subsampled iNaturalist, SUN, and Places via the following links and extract them to `$DATA/LargeOOD/`:

```bash
cd $DATA/LargeOOD/

wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz

tar -xvf iNaturalist.tar.gz
tar -xvf SUN.tar.gz
tar -xvf Places.tar.gz
tar -xvf dtd-r1.0.1.tar.gz
```

The directory structure should look like
```
LargeOOD/
|–– dtd/images/
|–– iNaturalist/images/
|–– Places/images/
|–– SUN/images/
```

