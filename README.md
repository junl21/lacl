# lacl

This is a PyTorch implementation of the paper ["Lesion-Aware Contrastive Representation Learning for Histopathology Whole Slide Images Analysis"](https://arxiv.org/abs/2206.13115). 

```angular2html
@inproceedings{li2022lesion,
    author    = {Jun Li, Yushan Zheng*, Kun Wu, Jun Shi, Fengying Xie, Zhiguo Jiang},
    title     = {Lesion-Aware Contrastive Representation Learning For Histopathology Whole Slide Images Analysis},
    booktitle = {Medical Image Computing and Computer Assisted Intervention 
                 -- MICCAI 2022},
    year      = {2022}
}
```

Our code is modified from repository [moco](https://github.com/facebookresearch/moco).

### Data Preparation

This code use "train.txt" to store the path and pseudo-label of images. An example of "train.txt" file is described as follows:

```angular2html
<path>                         <pseudo-label>
[path to slide1]/0000_0000.jpg 0
[path to slide1]/0000_0001.jpg 0
...
[path to slide2]/0000_0000.jpg 1
...
```

Note: we assign the pseudo-label for the patches from a WSI as the same of the WSI.

### Training

Use "default" contrastive table to train the model by following command. This mode will construct the negative sample pair from all other classes of lesion queue.

```angular2html
python main_lacl.py \
  -a resnet50 \
  --moco-k [number of classes] \
  --mlp --aug-plus --cos \
  --dist-url 'tcp://localhost:10002' --multiprocessing-distributed \
  [your train.txt file folders]
```

Use "custom" contrastive table to train the model by following command. This mode will construct the negative sample pair from custom settings in [lacl/utils.py](https://github.com/junl21/lacl/blob/main/lacl/utils.py).

```
python main_lacl.py \
  -a resnet50 \
  --moco-k [number of classes] --contras-mode 'custom' \
  --mlp --aug-plus --cos \
  --dist-url 'tcp://localhost:10002' --multiprocessing-distributed \
  [your train.txt file folders]
```