import os
import numpy as np
import pandas as pd
import seaborn as sns

from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from torch.nn.functional import normalize

def tsne_graph(df, output_dir, epoch, name):
    X = np.array(df['feat'].to_list())
    X = np.reshape(X, (X.shape[0], -1))

    X_tsne = TSNE(verbose = 1).fit_transform(X)

    df["comp-1"] = X_tsne[:,0]
    df["comp-2"] = X_tsne[:,1]

    plt.figure(figsize = (15, 15), dpi = 80, facecolor = 'silver', edgecolor = 'gray')

    folder_name  = "TSNE_{}".format(name)

    save_path = os.path.join(output_dir, folder_name)
    if not os.path.isdir(save_path): os.makedirs(save_path)
    fig_path  = os.path.join(save_path, "epoch_{}.png".format(epoch))

    sns.scatterplot(x = "comp-1", y = "comp-2", 
                    hue = "label", s = 50,
                    palette = "deep",
                    data = df).set(title = "{}, Epoch: {}".format(name, epoch))
    plt.xlim([-60, 60])
    plt.ylim([-60, 60])
    plt.savefig(fig_path)



def evaluation(teacher_model, student_model, dataloader, output_dir, epoch = ""):
    classes = dataloader.dataset.classes

    student_results, teacher_results = list(), list()
    process = tqdm(dataloader, total = len(dataloader), ncols = 200)
    for samples, labels in process:
        samples = samples.cuda(non_blocking=True)
        labels  = labels.cuda(non_blocking=True)
        
        feats_teacher = teacher_model(samples)
        feats_student = student_model(samples)
        
        feats_student = normalize(feats_teacher)
        feats_student = normalize(feats_student)

        for i in range(len(labels)):
            label = labels[i]
            feat_t = feats_teacher[i]
            feat_s = feats_student[i]

            teacher_results.append({
                "label" : classes[label.item()],
                "feat"  : feat_t.cpu().detach().numpy()})

            student_results.append({
                "label" : classes[label.item()],
                "feat"  : feat_s.cpu().detach().numpy()})

    teacher_df = pd.DataFrame(teacher_results)
    student_df = pd.DataFrame(student_results)

    tsne_graph(teacher_df, output_dir, epoch, name = "teacher")
    tsne_graph(student_df, output_dir, epoch, name = "student")

