# Neural Architecture Search of Semantic Segmentation Models (in PyTorch)

This repository provides official models from two papers:
1. `Fast Neural Architecture Search of Compact Semantic Segmentation Models via Auxiliary Cells`, available [here](https://arxiv.org/abs/1810.10804);
2. `Template-Based Automatic Search of Compact Semantic Segmentation Architectures`, available [here](https://arxiv.org/abs/1904.02365).

For citations:
```
Fast Neural Architecture Search of Compact Semantic Segmentation Models via Auxiliary Cells
Vladimir Nekrasov, Hao Chen, Chunhua Shen, Ian Reid
CVPR, 2019
```
and

```
Template-Based Automatic Search of Compact Semantic Segmentation Architectures
Vladimir Nekrasov, Chunhua Shen, Ian Reid
WACV, 2020
```

## Updates

*22 May, 2020*: Added the search script for the WACV 2020 experiments on CityScapes.
*05 April, 2020* : Added decoder design and pre-trained segmentation models from the WACV 2020 paper ["Template-Based Automatic Search of Compact Semantic Segmentation Architectures"](https://arxiv.org/abs/1904.02365).

## Getting Started

For flawless reproduction of our results, the Ubuntu OS is recommended. The models have been tested using Python 3.6.

### Dependencies

```
pip3
Cython
cv2
jupyter-notebook
matplotlib
numpy
Pillow
torch>=1.0
torchvision
```

## Inference Examples

For the ease of reproduction, we have embedded all our examples inside Jupyter notebooks.

### CVPR 2019 Segmentation

Please refer to results on [PASCAL VOC](./examples/inference/VOC-segm.ipynb)

### CVPR 2019 Depth Estimation

Please refer to results on [NYUD-v2](./examples/inference/NYU-depth.ipynb)

### WACV 2020 Segmentation CityScapes

Please refer to results on [CityScapes](./examples/inference/WACV-CS-segm.ipynb)

### WACV 2020 Segmentation CamVid

Please refer to results on [CamVid](./examples/inference/WACV-CV-segm.ipynb)

## Note on the runtime of WACV 2020 models

In the paper we wrongly claimed that the latency of `arch0` and `arch1` on `2048x1048` inputs were `95.7±0.721` and `145±0.215`, correspondingly (on a single `1080Ti`). These numbers were incorrect as the model was in the training regime and not in the evaluation mode (`model.eval()` in PyTorch). Below are the characteristics of the discovered architectures with correct latency:

|| arch0 | arch1
| -------- |:-------------:| -----:|
|Number of parameters (19 output classes)|280147|268235
|Latency, ms on 2048x1024 inputs|52.25±0.03|97.11±0.24
|Latency, ms on 480x360 inputs|8.97±0.10|11.51±0.14


## Search

If you would like to search for architectures yourself, please follow the instructions below:

### CVPR 2019 (PASCAL VOC)

#### Prepare data

You would need to have PASCAL VOC segmentation dataset expanded with annotations from BSD.

After that, run in your terminal:

```bash
mkdir data/datasets
ln -s /path_to_voc/VOCdevkit data/datasets/
```

#### Running search

We rely on a Cython-based script for calculating mean IoU. In order to build the corresponding files, run the following:

```bash
cd src
python helpers/setup.py build_ext --build-lib=./helpers/
```

After that, you can execute `./examples/search/search.sh` that will start the search process.

***[!] Please note that all the hyperparameters were tuned for running the search process on 2 GPUs, each with at least 11GB of memory. In case when your setup differs, you would need to tune the hyperparameters accordingly.***

#### Checking the results

We output the log-file with all information on the search process. You can easily see top-performing architectures using the following command: `python src/helpers/num_uq.py path_to_genotypes.out`

### WACV 2020 (CityScapes)

#### Prepare data

You would need to have the CityScapes segmentation dataset.

After that, run in your terminal:

```bash
mkdir data/datasets
ln -s /path_to_cityscapes/cs/ data/datasets/
```

#### Running search

After that, you can execute `./examples/search/search_wacv.sh` that will start the search process.

***[!] Please note that all the hyperparameters were tuned for running the search process on 2 1080Ti GPUs. In case when your setup differs, you would need to tune the hyperparameters accordingly.***

## License

This project is free to use for non-commercial purposes - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

* University of Adelaide and Australian Centre for Robotic Vision (ACRV) for making this project happen
* HPC Phoenix cluster at the University of Adelaide for making the training of the models possible
* PyTorch developers
* Yerba mate tea
