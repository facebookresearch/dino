#Define Path 
load_path = "/a/yu-yamaoka/Scientific_reports/Crop_Data/TEST256clear_tuning_aug64/epo80"
save_path = load_path
data_num = 24000

import os
import torch
import numpy as np

import umap
import umap.plot
import matplotlib.pyplot as plt

# Visualization

def umap_visualization(data, label, filename):

    mapper = umap.UMAP(n_components=2).fit(data) #default nearest neighbour=15
    umap.plot.points(mapper, labels=label)
    plt.savefig(filename+".png")
    plt.show()

os.makedirs(save_path, exist_ok=True)


train_features = torch.load(os.path.join(load_path, "trainfeat.pth"))
train_labels = torch.load(os.path.join(load_path, "trainlabels.pth"))

test_features = torch.load(os.path.join(load_path, "testfeat.pth"))
test_labels = torch.load(os.path.join(load_path, "testlabels.pth"))
test_labels += 4

new_features = torch.cat((train_features, test_features), dim=0)

    
# Low memeory対策：Random Sampling
rand_idx = np.random.permutation(len(train_features))
umap_visualization(train_features[rand_idx][0:data_num].detach().cpu(), train_labels[rand_idx][0:data_num].detach().cpu(),
                    os.path.join(save_path, "train"))


test_features = torch.load(os.path.join(load_path, "testfeat.pth"))
test_labels = torch.load(os.path.join(load_path, "testlabels.pth"))

# Low memeory対策：Random Sampling
rand_idx = np.random.permutation(len(test_features))
umap_visualization(test_features[rand_idx][0:data_num].detach().cpu(), test_labels[rand_idx][0:data_num].detach().cpu(),
                    os.path.join(save_path, "test"))

