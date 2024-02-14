#!/usr/bin/env python3

from my_transformer import IBVSTransformerTraining
import torch
import torch.nn as nn
import os
import argparse
import copy

def initial_parser(parser):
    parser.add_argument('--model', type=str, default='vit_small', help='Name of Model')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch-Size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--flattened_tensor', type=bool, default=True, help='Use if using only 1 MLP on flattened image tensor')
    parser.add_argument('--use_given_its', type=bool, default=False, help='If true, stop iterations after it_per_epoch')
    parser.add_argument('--perc_train_traj', type=int, default=90, help='Percantage of trajectories to train on')
    parser.add_argument('--perc_vel_traj', type=int, default=10, help='Percantage of trajectories for validation')
    parser.add_argument('--pretrained', type=bool, default=True, help='Choose if pretrained model')
    parser.add_argument('--train_norm', type=bool, default=False, help='Choose if normalization is trained')
    parser.add_argument('--layer_norm', type=bool, default=True, help='Choose if using normalization layer')
    parser.add_argument('--activation', type=str, default='tanh', help='Choose activation function')
    parser.add_argument('--use_scheduler', type=bool, default=True, help='Use scheduler for learning rate adaption')
    parser.add_argument('--use_seed', type=bool, default=True, help='Choose if seed value is used for repeatability')
    parser.add_argument('--seed', type=int, default=9757937385320587333, help='Seed value')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for L2 regularization during training. Used for optimizer') # default_value is 1e-2

    # GPU-Optionen
    parser.add_argument('--cuda', action='store_true', help='Verwenden Sie CUDA (GPU), wenn verfügbar')
    return parser

def set_parser_args(parser, name, type_name, default_value, help_str):
    parser.add_argument('--' + name, type=type_name, default=default_value, help=help_str)
    return parser

def get_args(parser):
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    torch.cuda.empty_cache()
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:200" #512
    parser = argparse.ArgumentParser(description='Train a neural network')
    init_parser = initial_parser(parser)
    
    num_traj = 600
    num_train_traj = 550
    start_case = 8
    num_test_cases = 1
    it_per_epoch = 10000

    output_features = [8]
    flattened_tensor = True
    normalized=False
    use_loc_tok = True
    use_cls_tok = True
    without_scale = False
    pretrained=True
    use_depth = False
    beta=1000
    
    loss_func = nn.SmoothL1Loss(reduction='mean', beta=beta)
    parser = set_parser_args(parser=copy.deepcopy(init_parser), 
                             name='blurred', 
                             type_name=bool, 
                             default_value=False,
                             help_str='Choose if rgb images have to be blurred')
    parser.add_argument('--random_blur', type=bool, default=False, help='Choose if random images get blurred')

    for test_case in range(start_case, start_case + num_test_cases):
        if test_case == 1:
            output_features = [8]
            use_loc_tok = False
            use_cls_tok = True
        elif test_case == 2:
            output_features = [8]
            use_loc_tok = True
            use_cls_tok = False
        elif test_case == 3:
            output_features = [8]
            use_loc_tok = True
            use_cls_tok = True
        elif test_case == 4:
            output_features = [4,4]
            use_loc_tok = False
            use_cls_tok = True
        elif test_case == 5:
            output_features = [4,4]
            use_loc_tok = True
            use_cls_tok = False
        elif test_case == 6:
            output_features = [4,4]
            use_loc_tok = True
            use_cls_tok = True
        elif test_case == 7:
            output_features = [3,1,3,1]
            use_loc_tok = False
            use_cls_tok = True
        elif test_case == 8:
            output_features = [3,1,3,1]
            use_loc_tok = True
            use_cls_tok = False
        elif test_case == 9:
            output_features = [3,1,3,1]
            use_loc_tok = True
            use_cls_tok = True
        """
        output_features = [8]
        flattened_tensor = True
        normalized=False
        use_loc_tok = True
        use_cls_tok = True
        without_scale = False
        pretrained=True
        use_depth = False
        beta=1000
        if test_case == 1:
            loss_func = nn.SmoothL1Loss(reduction='mean', beta=beta)
            parser = set_parser_args(parser=copy.deepcopy(init_parser), 
                            name='blurred', 
                            type_name=bool, 
                            default_value=False,
                            help_str='Choose if rgb images have to be blurred')
            parser.add_argument('--random_blur', type=bool, default=False, help='Choose if random images get blurred')
        elif test_case == 2:
            without_scale = True
            loss_func = nn.SmoothL1Loss(reduction='mean', beta=beta)
            parser = set_parser_args(parser=copy.deepcopy(init_parser), 
                            name='blurred', 
                            type_name=bool, 
                            default_value=True,
                            help_str='Choose if rgb images have to be blurred')
            parser.add_argument('--random_blur', type=bool, default=True, help='Choose if random images get blurred')
        elif test_case == 3:
            without_scale = True
            loss_func = nn.SmoothL1Loss(reduction='mean', beta=beta)
            parser = set_parser_args(parser=copy.deepcopy(init_parser), 
                            name='blurred', 
                            type_name=bool, 
                            default_value=True,
                            help_str='Choose if rgb images have to be blurred')
            parser.add_argument('--random_blur', type=bool, default=False, help='Choose if random images get blurred')
        else:
            print(f'All Cases for {pretrained=} finished')
        """
        if without_scale and len(output_features)==1:
            output_features = [6]
        args = get_args(parser)
        trans_trainer = IBVSTransformerTraining(args=args, pretrained=pretrained, use_loc_tok=use_loc_tok, 
                                                use_cls_tok=use_cls_tok, flattened_tensor=flattened_tensor,
                                                loss_func=loss_func, beta=beta, use_depth=use_depth, output_features=output_features)
        trans_trainer.train_transformer(args=args, 
                                        test_case=test_case, it=it_per_epoch, pretrained=pretrained, normalize_imgs=normalized,
                                        without_scale=without_scale)
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""