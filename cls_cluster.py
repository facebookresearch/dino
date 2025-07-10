"""
cluster_dbscan_hdbscan_plot.py
在“DBSCAN ➜ 递归 HDBSCAN”基础上，生成 3D 散点图：
- 颜色 = 第一层 DBSCAN (大类)
- marker 形状 = 子类 (细分 label)
"""

import argparse, json, sys
from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import (silhouette_score,
                             davies_bouldin_score,
                             calinski_harabasz_score,
                             adjusted_rand_score)
import umap.umap_ as umap
import random
import hdbscan
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message=".*force_all_finite.*")


# ---------- 第一层：DBSCAN ---------------------------------
def run_dbscan(X, eps, min_samples, metric):
    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=-1)
    return model.fit_predict(X)

# ---------- 递归：HDBSCAN ----------------------------------
def run_hdbscan(X, min_cluster_size, min_samples, metric):
    if metric == "cosine":
        # HDBSCAN 不支持 cosine 距离，需要转换为欧氏距离
       
        clst = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                            min_samples=min_samples,
                            algorithm='brute',
                            metric=metric)
    else:
        clst = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                            min_samples=min_samples,
                            metric=metric)
    return clst.fit_predict(X)

def recursive_hdbscan(X, depth, max_depth, min_cluster_size, min_samples,
                      split_factor, metric, hierarchy, parent):
    labels = run_hdbscan(X, min_cluster_size, min_samples, metric)
    uniq, cnts = np.unique(labels, return_counts=True)
    hierarchy[parent] = {int(u): int(c) for u, c in zip(uniq, cnts)}

    global_labels = np.full(X.shape[0], -1, dtype=int)
    next_label = 0
    for lab, n in zip(uniq, cnts):
        if lab == -1:
            continue
        mask = labels == lab
        need_split = (depth < max_depth) and (n >= min_cluster_size * split_factor)
        if need_split:
            sub = recursive_hdbscan(
                X[mask], depth+1, max_depth,
                min_cluster_size, min_samples, split_factor,
                metric, hierarchy, f"{parent}/{lab}"
            )
            sub_uniq = np.unique(sub[sub != -1])
            mapping = {s: next_label+i for i, s in enumerate(sub_uniq)}
            for s, new_s in mapping.items():
                sub[sub == s] = new_s
            next_label += len(mapping)
            global_labels[mask] = sub
        else:
            global_labels[mask] = next_label
            next_label += 1
    return global_labels

### NEW ###
def plot_clusters_3d(emb, db_labels, final_labels, out_path, elev=30, azim=45):
    """
    emb : (N, 3)  –  三维嵌入
    db_labels : (N,)  –  第一层 DBSCAN label
    final_labels : (N,) –  细分后扁平 label
    """
    if emb.shape[1] != 3:
        raise ValueError("embedding 维度必须为 3，才能直接绘 3D 图。")

    # ---- 颜色映射（大类） ----
    uniq_db = sorted([u for u in np.unique(db_labels) if u != -1])
    cmap = plt.colormaps.get_cmap('tab20')
    # 色差足够大的离散 colormap
    db2color = {lab: cmap(i) for i, lab in enumerate(uniq_db)}
    db2color[-1] = (0.7, 0.7, 0.7, 0.3)          # 噪声灰 + 透明

    # ---- marker 映射（子类） ----
    base_markers = ['o', '^', 's', 'P', 'X', 'D', 'v', '<', '>', '*']
    # 对每个“大类”单独循环 marker，保证同一大类颜色一致
    sub_marker = {}
    for lab in uniq_db:
        sub_ids = np.unique(final_labels[(db_labels == lab) & (final_labels != -1)])
        for idx, sid in enumerate(sub_ids):
            sub_marker[sid] = base_markers[idx % len(base_markers)]
    sub_marker[-1] = '.'   # 噪声

    # ---- 绘图 ----
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for sid in np.unique(final_labels):
        mask = final_labels == sid
        parent = db_labels[mask][0] if sid != -1 else -1
        ax.scatter(emb[mask, 0], emb[mask, 1], emb[mask, 2],
                   c=[db2color[parent]], marker=sub_marker[sid],
                   s=12 if sid != -1 else 8, alpha=0.9 if sid != -1 else 0.3,
                   edgecolors='none')

    ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2'); ax.set_zlabel('UMAP-3')
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.show()
    plt.close(fig)
    print(f"✅ Saved 3D cluster plot ➜ {out_path}")

# ---------- CLI & 主流程 -----------------------------------
def main(args):
    X = np.load(args.emb_path)        # ← 可能是 3 维或更高
    print(f"✅ Loaded embedding: {X.shape}")

    # ------ Step-1：DBSCAN ------
    db_labels = run_dbscan(X, args.db_eps, args.db_min_samples, args.metric)

    # ------ Step-1: DBSCAN ------
    db_labels = run_dbscan(X, args.db_eps, args.db_min_samples, args.metric)

    uniq, cnts = np.unique(db_labels, return_counts=True)
    print(f"[DBSCAN] clusters={len(uniq) - (1 if -1 in uniq else 0)}, noise={np.mean(db_labels == -1):.2%}")

    # ------ Step-2: 递归 HDBSCAN ------
    hierarchy = {"root": {int(u): int(c) for u, c in zip(uniq, cnts)}}
    final_labels = np.full(X.shape[0], -1, dtype=int)
    next_label = 0
    for lab, n in zip(uniq, cnts):
        if lab == -1:
            continue
        mask = db_labels == lab
        need_split = n >= args.hdb_min_cluster_size * args.split_factor
        if need_split:
            sub = recursive_hdbscan(
                X[mask], depth=1, max_depth=args.max_depth,
                min_cluster_size=args.hdb_min_cluster_size,
                min_samples=args.hdb_min_samples,
                split_factor=args.split_factor,
                metric=args.metric,
                hierarchy=hierarchy,
                parent=f"root/{lab}",
            )
            sub_uniq = np.unique(sub[sub != -1])
            mapping = {s: next_label+i for i, s in enumerate(sub_uniq)}
            for s, new_s in mapping.items():
                sub[sub == s] = new_s
            next_label += len(mapping)
            final_labels[mask] = sub
        else:
            final_labels[mask] = next_label
            next_label += 1

        # ==================== ① 数值指标 ==================== #
    ### METRIC ###
    mask = final_labels != -1
    if len(np.unique(final_labels[mask])) > 1:
        sil  = silhouette_score(X[mask], final_labels[mask])
        dbi  = davies_bouldin_score(X[mask], final_labels[mask])
        ch   = calinski_harabasz_score(X[mask], final_labels[mask])
        print(f"\n📊 Internal metrics (噪声已剔除):")
        print(f"   Silhouette          = {sil:.4f}  (↑)")
        print(f"   Davies-Bouldin      = {dbi:.4f}  (↓)")
        print(f"   Calinski-Harabasz   = {ch:.2f}   (↑)")
    else:
        print("\n📊 Too few clusters (≤1) after removing noise; metrics skipped.")

    # ================== ② 稳定性 / 交叉验证 ================= #
    ### STABILITY ###
    if args.n_stability > 0:
        aris = []
        N = len(X)
        idx_all = np.arange(N)
        for _ in range(args.n_stability):
            sub_idx = np.random.choice(N, int(args.stab_ratio * N), replace=False)
            sub_labels = final_labels.copy()
            # —— 只在子样本上重新聚类（重跑 DBSCAN+HDBSCAN）—— #
            sub_X = X[sub_idx]
            tmp_db  = run_dbscan(sub_X, args.db_eps, args.db_min_samples, args.metric)
            tmp_fin = np.full(len(sub_X), -1)
            next_lab = 0
            for lab in np.unique(tmp_db):
                if lab == -1: continue
                m = tmp_db == lab
                tmp_fin[m] = run_hdbscan(
                    sub_X[m], args.hdb_min_cluster_size,
                    args.hdb_min_samples, args.metric
                ) + next_lab
                next_lab = tmp_fin.max() + 1
            # —— 计算 ARI —— #
            aris.append(adjusted_rand_score(final_labels[sub_idx], tmp_fin))
        print(f"\n🔁 Stability (n={args.n_stability}, ratio={args.stab_ratio}): "
              f"ARI = {np.mean(aris):.4f} ± {np.std(aris):.4f}")

    # =================== ③ 再降 3 维可视化 ================= #
    ### VIS-UMAP ###
    if args.save_fig:
        emb3 = X[:, :3] if X.shape[1] == 3 else \
            umap.UMAP(n_components=3, random_state=42).fit_transform(X)
        # ① 大类配色 + ② 同大类内子簇换 marker
        fig_path = Path(args.output_dir) / args.fig_name
        plot_clusters_3d(emb3, db_labels, final_labels, fig_path,
                         elev=args.elev, azim=args.azim)


    # ------ 保存 ------
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "labels.npy", final_labels)
    with open(out / "cluster_hierarchy.json", "w") as f:
        json.dump(hierarchy, f, indent=2)
    print(f"✅ Saved labels & hierarchy ➜ {out.resolve()}")
    print(f"Total clusters (excl. noise): {len(np.unique(final_labels[final_labels != -1]))}")

    # # ------ 绘图 ------
    # if args.save_fig:
    #     fig_path = out / args.fig_name
    #     plot_clusters_3d(X[:, :3], db_labels, final_labels, fig_path,
    #                      elev=args.elev, azim=args.azim)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_path",  required=True, help="PCA/UMAP 输出的 .npy（3 维）")
    ap.add_argument("--output_dir", default="clusters", help="结果保存目录")
    # DBSCAN 超参
    ap.add_argument("--db_eps", type=float, default=0.6)
    ap.add_argument("--db_min_samples", type=int, default=10)
    # HDBSCAN 超参
    ap.add_argument("--hdb_min_cluster_size", type=int, default=50)
    ap.add_argument("--hdb_min_samples", type=int, default=None)
    ap.add_argument("--split_factor", type=float, default=2.0)
    ap.add_argument("--max_depth", type=int, default=2)
    # metric
    ap.add_argument("--metric", type=str, default="euclidean")
    # 绘图选项
    ap.add_argument("--save_fig", action="store_true", help="保存 3D 图")
    ap.add_argument("--fig_name", type=str, default="clusters_3d.png")
    ap.add_argument("--elev", type=float, default=30, help="视角 elev")
    ap.add_argument("--azim", type=float, default=45, help="视角 azim")
    ap.add_argument("--n_stability", type=int, default=5,
                    help="随机子采样次数 (0 = 不做稳定性分析)")
    ap.add_argument("--stab_ratio", type=float, default=0.8,
                    help="每次子样本比例")

    args = ap.parse_args(
        [
            '--emb_path', '../dino_data/output_dino/cls_vis_output/umap_output.npy',
            '--output_dir', '../dino_data/output_dino/clusters',
            '--db_eps', '0.8',
            '--db_min_samples', '300',
            '--hdb_min_cluster_size', '200',
            '--split_factor', '3.0',
            '--max_depth', '1',
            '--metric', 'euclidean',
            '--save_fig',
            # '--metric', 'cosine',
            '--n_stability', '10',
            '--stab_ratio', '0.9',
        ]
    )
    main(args)