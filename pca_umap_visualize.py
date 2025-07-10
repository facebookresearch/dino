import pickle
import numpy as np
from sklearn.decomposition import PCA
import umap.umap_ as umap
import pyvista as pv
from pathlib import Path
import argparse

def load_cls_data(cls_path):
    with open(cls_path, "rb") as f:
        data = pickle.load(f)
    return data

def collect_cls_and_labels(cls_data):
    all_cls = []
    person_labels = []
    for person, cls_list in cls_data.items():
        for cls in cls_list:
            all_cls.append(cls)
            person_labels.append(person)
    all_cls = np.array(all_cls)
    return all_cls, person_labels

def save_reducers(pca, umap_model, output_dir):
    with open(Path(output_dir) / "pca_model.pkl", "wb") as f:
        pickle.dump(pca, f)
    with open(Path(output_dir) / "umap_model.pkl", "wb") as f:
        pickle.dump(umap_model, f)
    print(f"✅ Saved PCA and UMAP models to {output_dir}")

def load_reducers(output_dir):
    with open(Path(output_dir) / "pca_model.pkl", "rb") as f:
        pca = pickle.load(f)
    with open(Path(output_dir) / "umap_model.pkl", "rb") as f:
        umap_model = pickle.load(f)
    print(f"✅ Loaded PCA and UMAP models from {output_dir}")
    return pca, umap_model

def apply_pca_umap(all_cls, pca_dim, umap_dim):
    print("Running PCA...")
    pca = PCA(n_components=pca_dim)
    pca_out = pca.fit_transform(all_cls)

    print("Running UMAP...")
    reducer = umap.UMAP(n_components=umap_dim, random_state=42)
    umap_out = reducer.fit_transform(pca_out)

    return umap_out, pca, reducer

def generate_colors(person_labels):
    unique_persons = sorted(set(person_labels))
    colormap = np.random.rand(len(unique_persons), 3)  # RGB colors
    person_to_color = {p: colormap[i] for i, p in enumerate(unique_persons)}
    color_array = np.array([person_to_color[p] for p in person_labels])
    return color_array, person_to_color

def plot_with_pyvista(umap_out, color_array, out_path, title="Person CLS Point Clouds"):
    cloud = pv.PolyData(umap_out)
    cloud["colors"] = color_array
    plotter = pv.Plotter(off_screen=True)
    plotter.add_points(cloud, scalars="colors", rgb=True, point_size=5)
    plotter.add_axes()
    plotter.show(screenshot=str(out_path))
    print(f"✅ Saved plot to {out_path}")

def plot_each_person(umap_out, person_labels, person_to_color, output_dir):
    unique_persons = sorted(set(person_labels))
    for person in unique_persons:
        mask = np.array([p == person for p in person_labels])
        points = umap_out[mask]
        cloud = pv.PolyData(points)
        color = person_to_color[person]
        plotter = pv.Plotter(off_screen=True)
        plotter.add_points(cloud, color=color, point_size=5)
        plotter.add_axes()
        out_path = output_dir / f"person_{person}_point_cloud.png"
        plotter.show(screenshot=str(out_path))
        print(f"✅ Saved plot to {out_path}")

def plot_person_in_context(umap_out, person_labels, person_to_color, output_dir):
    """
    在总体灰色点云背景上高亮每个 person 自己的点。
    """
    unique_persons = sorted(set(person_labels))
    grey = (0.7, 0.7, 0.7)          # 背景灰，取一个中性亮度

    # 预先把全体云图准备好（提升一点效率，避免每次都复制）
    background_cloud = pv.PolyData(umap_out)

    for person in unique_persons:
        # —— 选中该 person 的点 —— #
        mask = np.array([p == person for p in person_labels])
        person_points = umap_out[mask]
        person_cloud = pv.PolyData(person_points)
        color = person_to_color[person]

        # —— 画图 —— #
        plotter = pv.Plotter(off_screen=True)
        plotter.add_points(background_cloud, color=grey, point_size=4)
        plotter.add_points(person_cloud, color=color, point_size=6)
        plotter.add_axes()

        out_path = Path(output_dir) / f"person_{person}_in_context.png"
        plotter.show(screenshot=str(out_path))
        print(f"✅ Saved context plot to {out_path}")

def main(args):
    cls_data = load_cls_data(args.cls_path)
    print(f"✅ Loaded {len(cls_data)} CLS entries from {args.cls_path}")

    all_cls, person_labels = collect_cls_and_labels(cls_data)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(args.load_model)
    if args.load_model:
        # 加载模型并转换
        pca, umap_model = load_reducers(args.output_dir)
        pca_out = pca.transform(all_cls)
        umap_out = umap_model.transform(pca_out)
    else:
        # 训练新模型
        umap_out, pca, umap_model = apply_pca_umap(all_cls, args.pca_dim, args.umap_dim)
        save_reducers(pca, umap_model, args.output_dir)

    color_array, person_to_color = generate_colors(person_labels)

    all_out_path = Path(args.output_dir) / "all_persons_point_cloud.png"
    plot_with_pyvista(umap_out, color_array, all_out_path)

    plot_each_person(umap_out, person_labels, person_to_color, Path(args.output_dir))

    plot_person_in_context(umap_out, person_labels, person_to_color, Path(args.output_dir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls_path", type=str, required=True, help="Path to cls_by_person.pkl")
    parser.add_argument("--output_dir", type=str, default="../dino_data/output_dino/cls_vis_output", help="Output directory for plots")
    parser.add_argument("--pca_dim", type=int, default=20, help="PCA target dimension before UMAP")
    parser.add_argument("--umap_dim", type=int, default=3, help="UMAP target dimension (default 3)")
    parser.add_argument("--load_model", action="store_true", help="If set, load PCA/UMAP models from output_dir")
    args = parser.parse_args(
        [
            '--cls_path', '../dino_data/output_dino/cls_by_person.pkl',
            '--output_dir', '../dino_data/output_dino/cls_vis_output',
            '--pca_dim', '30',
            '--umap_dim', '3',
            '--load_model',
        ]
    )
    main(args)
