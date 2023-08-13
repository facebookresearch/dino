# Use this code to predict embeddings and then visualize them in tensorboard.
import argparse
import os

import numpy as np
import torch
from torch import nn
import sys
from torch.utils.tensorboard import SummaryWriter
import vision_transformer as vits
import utils
from nifti_datahandling import NumpyDatasetEval
from torchvision import models as torchvision_models



def prepare_tensorboard(backbone, data_loader, log_dir="./logs"):
    """
    Compute CLS embedding and prepare for TensorBoard
    :param backbone: Trained vision transformer
    :param data_loader: A dataloader that applies basic augmentations
    :param log_dir: Directory to save TensorBoard logs
    :return:
    embeddings: torch.Tensor: embeddings of shape (n_samples, out_dim)
    labels: List of strings representing the class (being what folder the image originates from)
    """
    device = next(backbone.parameters()).device
    embs_l = []
    imgs_l = []
    imgs_display = []
    labels = []

    for img, label, original_data in data_loader:
        img = img.to(device)
        embeddings = backbone(img).detach().cpu()
        embs_l.append(embeddings)
        imgs_l.append(img)
        labels.extend(label)

        # for display take a single slice, min max normalize and append

        # For ct clip then normalize, do not mulitply by 255 for visualization purposes
        original_data = np.clip(original_data[:, 1, :, :].cpu().numpy(), -150, 150)
        normalized_img = (original_data - np.min(original_data)) / (np.max(original_data) - np.min(original_data) + 0.001)
        imgs_display.append(torch.from_numpy(np.expand_dims(normalized_img, axis=1)))


    embeddings = torch.cat(embs_l, dim=0)
    imgs_final = torch.cat(imgs_display, dim=0)

    # Save the embeddings and labels to a file for TensorBoard
    writer = SummaryWriter(log_dir)
    writer.add_embedding(embeddings, metadata=labels, label_img=imgs_final, global_step=0)
    writer.close()

    return embeddings, labels


def run_visualization(args):
    # ============ building network ... ============
    if "vit" in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch](num_classes=0)
        model.fc = nn.Identity()
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()

    dataset = NumpyDatasetEval(args.data_path, paths_text=args.dataset_file)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    prepare_tensorboard(model, data_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation for embeddings in tensorboard')
    parser.add_argument('--batch_size_per_gpu', default=10, type=int, help='Per-GPU batch-size')
    parser.add_argument('--pretrained_weights', default='./checkpoint0080.pth', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
                        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--data_path', default='/home/edan/HighRad/Data/DicomClassifier/large_lungs_and_liver_test_set/', type=str)
    parser.add_argument('--dataset_file', default='test_files.txt', type=str)
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
    args = parser.parse_args()
    run_visualization(args)

