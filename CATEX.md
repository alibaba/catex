## How to Run

We provide the running scripts in `scripts/catex`, which allow you to reproduce the results on the NeurIPS'23 paper.

Make sure you change the path in `DATA` and run the commands under the main directory `CATEX/`.

### Training

**Training configration**: `CATEX/scripts/catex/ood_train.sh` contains all input arguments.
- `DATASET` takes as input a dataset name, like `imagenet` or `imagenet100_mcm`. The valid names are the files' names in `CATEX/configs/datasets/`.
- `CFG` means which config file to use, such as `rn50`, `vit_b16` or `vit_l14` (see `CATEX/configs/trainers/CATEX/`). Note that for `ImageNet*`, we use `CATEX/configs/trainers/CATEX/*_ep50.yaml` for all settings (please follow the implementation details shown in the paper).
- `ID_PERTUR_NUM` determines how much word to perturb to generate the spurious contexts from perceptual contexts. Default as `1`.
- `OOD_PROMPT_NUM` determines how much spurious contexts to learn for a given ID category. Default as `1`.
- `OOD_PROMPT_ORTH` indicates whether applying orthogonal constraints on multiple spurious contexts for a given ID category. It only works when `OOD_PROMPT_NUM > 1`. Default as `False`.

All the training command should follow:
```bash
bash ./scripts/catex/ood_train.sh DATASET CFG ID_PERTUR_NUM OOD_PROMPT_NUM OOD_PROMPT_ORTH
```

For instance, the default training setup on standard ImageNet benchmark is:
```bash
bash ./scripts/catex/ood_train.sh imagenet vit_b16_ep50 1 1 False
```


### Evaluation

**Evaluation configration**: `CATEX/scripts/catex/ood_test.sh` contains all input arguments.
- `DATASET` takes as input a dataset name, like `imagenet` or `imagenet100_mcm`. The valid names are the files' names in `CATEX/configs/datasets/`.
- `CFG` means which config file to use, such as `rn50`, `vit_b16` or `vit_l14` (see `CATEX/configs/trainers/CATEX/`). Note that for `ImageNet*`, we use `CATEX/configs/trainers/CATEX/*_ep50.yaml` for all settings (please follow the implementation details shown in the paper).
- `MODELDIR` assigns the checkpoints to evaluate, like the pre-trained `weights/imagenet`.
- `CTX_INIT` means the initially inherited contexts for each ID category. Default as `''`.
- `OOD_PROMPT_NUM` determins how much spurious contexts to learn for a given ID category. Default as `1`.
- `OOD_INFER_INTEGRATE` indicates whether applying integrated inference strategy as described in paper. Default as `True`.
- `OOD_INFER_OPTION` contains extra options for specific evaluation settings (such as cross-ID-task evaluation).


All the evaluation command should follow:
```bash
bash ./scripts/catex/ood_test.sh DATASET CFG MODELDIR CTX_INIT OOD_PROMPT_NUM OOD_INFER_INTEGRATE OOD_INFER_OPTION
```

Below we provide examples on how to evaluate CATEX on ImageNet datasets.

**Standard OOD Detection**:
- [x] ImageNet-1K: <br> `bash ./scripts/catex/ood_test.sh imagenet vit_b16_ep50 weights/imagenet '' 1 True`
- [x] ImageNet-100-MCM: <br> `bash ./scripts/catex/ood_test.sh imagenet100_mcm vit_b16_ep50 weights/imagenet100-MCM '' 1 True`
- [x] ImageNet-100-NEW: <br> `bash ./scripts/catex/ood_test.sh imagenet100_new vit_b16_ep50 weights/imagenet100-NEW '' 1 True`

**ID-Shifted OOD Detection**:
- [x] Transfer CATEX to ImageNet-A: <br> `bash ./scripts/catex/ood_test.sh imagenet_a vit_b16_ep50 weights/imagenet ensemble_learned 1 True`
- [x] Transfer CATEX to ImageNet-R: <br> `bash ./scripts/catex/ood_test.sh imagenet_r vit_b16_ep50 weights/imagenet ensemble_learned 1 True`
- [x] Transfer CATEX to ImageNet-Sketch: <br> `bash ./scripts/catex/ood_test.sh imagenet_sketch vit_b16_ep50 weights/imagenet ensemble_learned 1 True`

**Category-Extensible OOD Detection**:
- [x] Extract samples' features from ImageNet100-MCM and ImageNet100-NEW subsets, and perform OOD detection on the merged ImageNet200-UNION task: <br> 
```bash 
bash ./scripts/catex/ood_test.sh imagenet100_mcm vit_b16_ep50 weights/imagenet100-MCM '' 1 False save_res
bash ./scripts/catex/ood_test.sh imagenet100_new vit_b16_ep50 weights/imagenet100-NEW '' 1 False resume_res
```
- [ ] Extend CATEX to ImageNet-21K recognition: This involves sophisticated training and testing settings. Please stay tuned.


**Component Albation**:
- [x] Baseline CATEX: <br> `bash ./scripts/catex/ood_test.sh imagenet100_mcm vit_b16_ep50 weights/imagenet100-MCM '' 1 False`
- [x] Integrated inference: <br> `bash ./scripts/catex/ood_test.sh imagenet100_mcm vit_b16_ep50 weights/imagenet100-MCM '' 1 True`
- [x] Integrated inference with 8 Spurious-Contexts and orthogonal constraints: <br> `bash ./scripts/catex/ood_test.sh imagenet100_mcm vit_b16_ep50 weights/imagenet100-MCM-SCTX8-Orth '' 8True`


**Integration with Post-hoc OOD detection**:
- [x] Integrate CATEX with ReAct: <br> `bash ./scripts/catex/ood_test.sh imagenet vit_b16_ep50 weights/imagenet '' 1 True apply_react`
- [x] Integrate CATEX with BATS: <br> `bash ./scripts/catex/ood_test.sh imagenet vit_b16_ep50 weights/imagenet '' 1 True apply_bats`
- [x] Integrate CATEX with ASH: <br> `bash ./scripts/catex/ood_test.sh imagenet vit_b16_ep50 weights/imagenet '' 1 True apply_ash`


**Integration with Zero-Shot Recognition**:
- [ ] Integrate CATEX with [visual description](https://github.com/sachit-menon/classify_by_description_release): This involves cross-repository integration, and we are working on it. Please stay tuned.

