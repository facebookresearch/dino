# Self-Supervised Vision Transformers with DINO

PyTorch implementation and pretrained models for DINO. For details, see **Emerging Properties in Self-Supervised Vision Transformers**.  
[[`blogpost`](https://ai.facebook.com/blog/dino-paws-computer-vision-with-self-supervised-transformers-and-10x-more-efficient-training)] [[`arXiv`](https://arxiv.org/abs/2104.14294)] [[`Yannic Kilcher's video`](https://www.youtube.com/watch?v=h3ij3F3cPIk)]

<div align="center">
  <img width="100%" alt="DINO illustration" src=".github/dino.gif">
</div>

## Pretrained models
You can choose to download only the weights of the pretrained backbone used for downstream tasks, or the full checkpoint which contains backbone and projection head weights for both student and teacher networks. We also provide the training and evaluation logs.

<table>
  <tr>
    <th>arch</th>
    <th>params</th>
    <th>k-nn</th>
    <th>linear</th>
    <th colspan="5">download</th>
  </tr>
  <tr>
    <td>DeiT-S/16</td>
    <td>21M</td>
    <td>74.5%</td>
    <td>77.0%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth">backbone only</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_full_checkpoint.pth">full checkpoint</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/args.txt">args</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_log.txt">logs</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_eval_linear_log.txt">eval logs</a></td>
  </tr>
  <tr>
    <td>DeiT-S/8</td>
    <td>21M</td>
    <td>78.3%</td>
    <td>79.7%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth">backbone only</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_full_checkpoint.pth">full checkpoint</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/args.txt">args</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_log.txt">logs</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_eval_linear_log.txt">eval logs</a></td>
  </tr>
  <tr>
    <td>ViT-B/16</td>
    <td>85M</td>
    <td>76.1%</td>
    <td>78.2%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth">backbone only</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain_full_checkpoint.pth">full checkpoint</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/args.txt">args</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain_log.txt">logs</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain_eval_linear_log.txt">eval logs</a></td>
  </tr>
  <tr>
    <td>ViT-B/8</td>
    <td>85M</td>
    <td>77.4%</td>
    <td>80.1%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth">backbone only</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain_full_checkpoint.pth">full checkpoint</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/args.txt">args</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain_log.txt">logs</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain_eval_linear_log.txt">eval logs</a></td>
  </tr>
  <tr>
    <td>ResNet-50</td>
    <td>23M</td>
    <td>67.5%</td>
    <td>75.3%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth">backbone only</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain_full_checkpoint.pth">full checkpoint</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/args.txt">args</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain_log.txt">logs</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain_eval_linear_log.txt">eval logs</a></td>
  </tr>
</table>

The pretrained models are available on PyTorch Hub.
```python
import torch
deits16 = torch.hub.load('facebookresearch/dino:main', 'dino_deits16')
deits8 = torch.hub.load('facebookresearch/dino:main', 'dino_deits8')
vitb16 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
vitb8 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
resnet50 = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
```

## Training

### Documentation
Please install [PyTorch](https://pytorch.org/) and download the [ImageNet](https://imagenet.stanford.edu/) dataset. This codebase has been developed with python version 3.6, PyTorch version 1.7.1, CUDA 11.0 and torchvision 0.8.2. The exact arguments to reproduce the models presented in our paper can be found in the `args` column of the [pretrained models section](https://github.com/facebookresearch/dino#pretrained-models). For a glimpse at the full documentation of DINO training please run:
```
python main_dino.py --help
```

### Vanilla DINO training :sauropod:
Run DINO with DeiT-small network on a single node with 8 GPUs for 100 epochs with the following command. Training time is 1.75 day and the resulting checkpoint should reach 69.3% on k-NN eval and 74.0% on linear eval. We provide [training](https://dl.fbaipublicfiles.com/dino/example_runs_logs/dino_vanilla_deitsmall16_log.txt) and [linear evaluation](https://dl.fbaipublicfiles.com/dino/example_runs_logs/dino_vanilla_deitsmall16_eval.txt) logs (with batch size 256 at evaluation time) for this run to help reproducibility.
```
python -m torch.distributed.launch --nproc_per_node=8 main_dino.py --arch deit_small --data_path /path/to/imagenet/train --output_dir /path/to/saving_dir
```

### Multi-node training
We use Slurm and [submitit](https://github.com/facebookincubator/submitit) (`pip install submitit`). To train on 2 nodes with 8 GPUs each (total 16 GPUs):
```
python run_with_submitit.py --nodes 2 --ngpus 8 --arch deit_small --data_path /path/to/imagenet/train --output_dir /path/to/saving_dir
```

<details>
<summary>
DINO with ViT-base network.
</summary>

```
python run_with_submitit.py --nodes 2 --ngpus 8 --use_volta32 --arch vit_base  --data_path /path/to/imagenet/train --output_dir /path/to/saving_dir
```

</details>

### Boosting DINO performance :t-rex:
You can improve the performance of the vanilla run by:
- training for more epochs: `--epochs 300`,
- increasing the teacher temperature: `--teacher_temp 0.07 --warmup_teacher_temp_epochs 30`.
- removing last layer normalization (only safe with `--arch deit_small`): `--norm_last_layer false`,

<details>
<summary>
Full command.
</summary>

```
python run_with_submitit.py --arch deit_small --epochs 300 --teacher_temp 0.07 --warmup_teacher_temp_epochs 30 --norm_last_layer false --data_path /path/to/imagenet/train --output_dir /path/to/saving_dir
```

</details>

The resulting pretrained model should reach 73.3% on k-NN eval and 76.0% on linear eval. Training time is 2.6 days with 16 GPUs. We provide [training](https://dl.fbaipublicfiles.com/dino/example_runs_logs/dino_boost_deitsmall16_log.txt) and [linear evaluation](https://dl.fbaipublicfiles.com/dino/example_runs_logs/dino_boost_deitsmall16_eval.txt) logs (with batch size 256 at evaluation time) for this run to help reproducibility.

### ResNet-50 and other convnets trainings
This code also works for training DINO on convolutional networks, like ResNet-50 for example. We highly recommend to adapt some optimization arguments in this case. For example following is a command to train DINO on ResNet-50 on a single node with 8 GPUs for 100 epochs. We provide [training](https://dl.fbaipublicfiles.com/dino/example_runs_logs/dino_rn50_log.txt) logs for this run.
```
python -m torch.distributed.launch --nproc_per_node=8 main_dino.py --arch resnet50 --optimizer sgd --weight_decay 1e-4 --weight_decay_end 1e-4 --global_crops_scale 0.14 1 --local_crops_scale 0.05 0.14 --data_path /path/to/imagenet/train --output_dir /path/to/saving_dir
```

## Self-attention visualization
You can look at the self-attention of the [CLS] token on the different heads of the last layer by running:
```
python visualize_attention.py
```

## Self-attention video generation
You can generate videos like the one on the blog post with `video_generation.py`.

https://user-images.githubusercontent.com/46140458/116817761-47885e80-ab68-11eb-9975-d61d5a919e13.mp4

Extract frames from input video and generate attention video:
```
python video_generation.py  --pretrained_weights dino_deitsmall8_pretrain.pth \
    --input_path input/video.mp4 \
    --output_path output/ \
    --fps 25
```

Use folder of frames already extracted and generate attention video:
```
python video_generation.py  --pretrained_weights dino_deitsmall8_pretrain.pth \
    --input_path output/frames/ \
    --output_path output/ \
    --resize 256 \
```

Only generate video from folder of attention maps images:
```
python video_generation.py --input_path output/attention \
    --output_path output/ \
    --video_only \
    --video_format avi
```

Also, check out [this colab](https://gist.github.com/aquadzn/32ac53aa6e485e7c3e09b1a0914f7422) for a video inference notebook.

<div align="center">
  <img width="100%" alt="Self-attention from a Vision Transformer with 8x8 patches trained with DINO" src=".github/attention_maps.png">
</div>


## Evaluation: k-NN classification on ImageNet
To evaluate a simple k-NN classifier with a single GPU on a pre-trained model, run:
```
python -m torch.distributed.launch --nproc_per_node=1 eval_knn.py --data_path /path/to/imagenet
```
If you choose not to specify `--pretrained_weights`, then DINO reference weights are used by default. If you want instead to evaluate checkpoints from a run of your own, you can run for example:
```
python -m torch.distributed.launch --nproc_per_node=1 eval_knn.py --pretrained_weights /path/to/checkpoint.pth --checkpoint_key teacher --data_path /path/to/imagenet 
```

## Evaluation: Linear classification on ImageNet
To train a supervised linear classifier on frozen weights on a single node with 8 gpus, run:
```
python -m torch.distributed.launch --nproc_per_node=8 eval_linear.py --data_path /path/to/imagenet
```

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## Citation
If you find this repository useful, please consider giving a star :star: and citation :t-rex::
```
@article{caron2021emerging,
  title={Emerging Properties in Self-Supervised Vision Transformers},
  author={Caron, Mathilde and Touvron, Hugo and Misra, Ishan and J\'egou, Herv\'e  and Mairal, Julien and Bojanowski, Piotr and Joulin, Armand},
  journal={arXiv preprint arXiv:2104.14294},
  year={2021}
}
```
