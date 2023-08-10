# This class is to be used both for evaluation during training and as a metric for testing
import argparse
import numpy as np
import torch
from nifti_datahandling import NumpyDatasetEval
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score
from torch import nn
import sys
from torch.utils.tensorboard import SummaryWriter
import vision_transformer as vits
import utils
from torchvision import models as torchvision_models
from sklearn.preprocessing import LabelEncoder



class DinoKNN:

    def __init__(self, model, data: str, k: int, kfold: int):
        """

        :param model: Trained torch model for embedding
        :param data: path to text file containing all data or file containing embeddings.
        :param k: k nearest neighbors to use
        :param kfold: how many folds to use during cross fold validation
        """

        self.model = model
        self.model.cuda()
        self.model.eval()
        self.data = data
        self.k = k
        self.kfold = kfold

    def predict_knn(self, args) -> list:
        """
        This method is given either a list of strings that are all numpy arrays representing images
        or a path to a text file containing paths to the test samples
        each path is preprocessed, embedded by the model and then is given a label according to its knn
        :return: log of the results
        """

        device = next(self.model.parameters()).device
        # For each data point, preprocess and get an embedding
        dataset = NumpyDatasetEval(self.data, paths_text=args.eval_file)
        print(f'testing {args.eval_file}')
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        embeddings = []
        labels = []
        for X, y, _ in data_loader:
            X = X.to(device)
            cur_embeddings = self.model(X).detach().cpu()
            embeddings.append(cur_embeddings)
            labels.extend(y)
        embeddings = np.concatenate(embeddings, axis=0)
        # labels are strings, this transforms the labels into unique digits so we can work with sklearn
        labels = LabelEncoder().fit_transform(labels)

        # split data into kfold folds.
        # For each fold evaluate precision and recall
        scores = []
        kf = KFold(n_splits=self.kfold, shuffle=True, random_state=8)
        for i, (train_indices, val_indices) in enumerate(kf.split(embeddings)):
            neighbors = KNeighborsClassifier(n_neighbors=self.k).fit(embeddings[train_indices], labels[train_indices])
            predictions = neighbors.predict(embeddings[val_indices])
            precision = precision_score(labels[val_indices], predictions, average="macro")
            recall = recall_score(labels[val_indices], predictions, average="macro")
            scores.append((recall, precision))

        # Return a log of the results as a dictionary
        return scores


def run_knn(args):
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
    dinoKNN = DinoKNN(model=model, data=args.data_path, k=5, kfold=5)
    logs = dinoKNN.predict_knn(args)
    print(logs)

    total_recall = np.average([tup[0] for tup in logs])
    total_precision = np.average([tup[1] for tup in logs])

    print(f'average recall over 5 fold validation: {total_recall}')
    print(f'average precision over 5 fold validation: {total_precision}')

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
    parser.add_argument('--data_path', default='/home/edan/HighRad/Data/DicomClassifier/val_dataset', type=str)
    parser.add_argument('--eval_file', default='val_dataset.txt', type=str)
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
    args = parser.parse_args()
    run_knn(args)


