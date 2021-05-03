# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import glob
import sys
import argparse
import cv2

from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

import utils
import vision_transformer as vits


def extract_frames_from_video():
    vidcap = cv2.VideoCapture(args.input_path)
    args.fps = vidcap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {args.input_path} ({args.fps} fps)")
    print("Extracting frames...")
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(os.path.join(args.output_dir, f"frame-{count:04}.jpg"), image)
        success, image = vidcap.read()
        count += 1


def generate_video_from_images(format="mp4"):
    print("Generating video...")
    img_array = []
    # Change format to png if needed
    for filename in tqdm(sorted(glob.glob(os.path.join(args.output_dir, "attn-*.jpg")))):
        with open(filename, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            size = (img.width, img.height)

            img_array.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

    if args.video_format == "avi":
        out = cv2.VideoWriter(
            "video.avi", cv2.VideoWriter_fourcc(*"XVID"), args.fps, size
        )
    else:
        out = cv2.VideoWriter(
            "video.mp4", cv2.VideoWriter_fourcc(*"MP4V"), args.fps, size
        )

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print("Done")


def inference(images_folder_list: str):
    for img_path in tqdm(images_folder_list):
        with open(img_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")

        if args.resize is not None:
            transform = pth_transforms.Compose(
                [
                    pth_transforms.ToTensor(),
                    pth_transforms.Resize(args.resize),
                    pth_transforms.Normalize(
                        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                    ),
                ]
            )
        else:
            transform = pth_transforms.Compose(
                [
                    pth_transforms.ToTensor(),
                    pth_transforms.Normalize(
                        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                    ),
                ]
            )

        img = transform(img)

        # make the image divisible by the patch size
        w, h = (
            img.shape[1] - img.shape[1] % args.patch_size,
            img.shape[2] - img.shape[2] % args.patch_size,
        )
        img = img[:, :w, :h].unsqueeze(0)

        w_featmap = img.shape[-2] // args.patch_size
        h_featmap = img.shape[-1] // args.patch_size

        attentions = model.forward_selfattention(img.to(device))

        nh = attentions.shape[1]  # number of head

        # we keep only the output patch attention
        attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - args.threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        # interpolate
        th_attn = (
            nn.functional.interpolate(
                th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest"
            )[0]
            .cpu()
            .numpy()
        )

        attentions = attentions.reshape(nh, w_featmap, h_featmap)
        attentions = (
            nn.functional.interpolate(
                attentions.unsqueeze(0), scale_factor=args.patch_size, mode="nearest"
            )[0]
            .cpu()
            .numpy()
        )

        # save attentions heatmaps
        os.makedirs(args.output_dir, exist_ok=True)
        fname = os.path.join(args.output_dir, "attn-" + os.path.basename(img_path))
        plt.imsave(
            fname=fname,
            arr=sum(
                attentions[i] * 1 / attentions.shape[0]
                for i in range(attentions.shape[0])
            ),
            cmap="inferno",
            format="jpg",
        )

    generate_video_from_images(args.video_format)


def load_model():
    # build model
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(
            "Pretrained weights found at {} and loaded with msg: {}".format(
                args.pretrained_weights, msg
            )
        )
    else:
        print(
            "Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate."
        )
        url = None
        if args.arch == "deit_small" and args.patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif args.arch == "deit_small" and args.patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
        elif args.arch == "vit_base" and args.patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif args.arch == "vit_base" and args.patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print(
                "Since no pretrained weights have been provided, we load the reference pretrained DINO weights."
            )
            state_dict = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/dino/" + url
            )
            model.load_state_dict(state_dict, strict=True)
        else:
            print(
                "There is no reference weights available for this model => We use random weights."
            )
    return model


def parse_args():
    parser = argparse.ArgumentParser("Visualize Self-Attention maps")
    parser.add_argument(
        "--arch",
        default="deit_small",
        type=str,
        choices=["deit_tiny", "deit_small", "vit_base"],
        help="Architecture (support only ViT atm).",
    )
    parser.add_argument(
        "--patch_size", default=8, type=int, help="Patch resolution of the model."
    )
    parser.add_argument(
        "--pretrained_weights",
        default="",
        type=str,
        help="Path to pretrained weights to load.",
    )
    parser.add_argument(
        "--checkpoint_key",
        default="teacher",
        type=str,
        help='Key to use in the checkpoint (example: "teacher")',
    )
    parser.add_argument(
        "--input_path",
        default=None,
        type=str,
        help="""Path to a video file if you want to extract frames
            or to a folder of images already extracted by yourself.""",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Path where to save visualizations and / or video.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx percent of the mass.""",
    )
    parser.add_argument(
        "--resize",
        default=None,
        type=int,
        nargs="+",
        help="""Apply a resize transformation to input image(s). Use if OOM error.
        Usage (single or W H): --resize 512, --resize 720 1280""",
    )
    parser.add_argument(
        "--fps",
        default=30.0,
        type=float,
        help="FPS of input / output video. Automatically set if you extract frames from a video.",
    )
    parser.add_argument(
        "--video_only",
        action="store_true",
        help="""Use this flag if you only want to generate a video and not all attention images.
            If used, --output_dir must be set to the folder containing attention images.""",
    )
    parser.add_argument(
        "--video_format",
        default="mp4",
        type=str,
        choices=["mp4", "avi"],
        help="Format of generated video (mp4 or avi).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = load_model()

    # If you only want a video
    if args.video_only:
        generate_video_from_images(args.video_format)
    else:
        # If input path isn't set
        if args.input_path is None:
            print(f"Provided input path {args.input_path} is non valid.")
            sys.exit(1)
        else:
            # If input path exists
            if os.path.exists(args.input_path):
                # If input is a video file
                if os.path.isfile(args.input_path):
                    extract_frames_from_video()
                    imgs_list = [
                        os.path.join(args.output_dir, i)
                        for i in sorted(os.listdir(args.output_dir))
                    ]
                    inference(imgs_list)
                # If input is an images folder
                if os.path.isdir(args.input_path):
                    imgs_list = [
                        os.path.join(args.input_path, i)
                        for i in sorted(os.listdir(args.input_path))
                    ]
                    inference(imgs_list)
            # If input path doesn't exists
            else:
                print(f"Provided video file path {args.input_path} is non valid.")
                sys.exit(1)
