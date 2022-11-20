import os
import numpy as np
import pandas as pd
import seaborn as sns

from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE



def tsne_graph(df, model_path, title, epoch = ""):
    X = np.array(df['feat'].to_list())
    X = np.reshape(X, (X.shape[0], -1))

    X_tsne = TSNE(verbose = 1).fit_transform(X)

    df["comp-1"] = X_tsne[:,0]
    df["comp-2"] = X_tsne[:,1]

    plt.figure(figsize = (15, 15), dpi = 80, facecolor = 'silver', edgecolor = 'gray')

    title  = "TSNE_{}".format(title)
    title += ", Epoch_{}".format(epoch) if epoch else "" 

    sns.scatterplot(x = "comp-1", y = "comp-2", 
                    hue = "label", s = 50,
                    data = df).set(title = title)
    
    save_path = os.path.join(model_path, title)
    if not os.path.isdir(save_path): os.makedirs(save_path)
    fig_path  = os.path.join(save_path, "epoch_{}.png".format(epoch))
    plt.savefig(fig_path)



def eval_model(model, dataloader, model_path, title = "", epoch = ""):
    classes = dataloader.dataset.classes

    dataset_list = list()
    process = tqdm(dataloader, total = len(dataloader), ncols = 200)
    for samples, labels in process:
        samples = samples.cuda(non_blocking=True)
        labels  = labels.cuda(non_blocking=True)
        
        feats = model(samples)
        
        for i in range(len(labels)):
            label = labels[i]
            feat  = feats[i]
            dataset_list.append({
                "label" : classes[label.item()],
                "feat"  : feat.cpu().detach().numpy()})

    df = pd.DataFrame(dataset_list)
    
    tsne_graph(df, model_path, title, epoch)

