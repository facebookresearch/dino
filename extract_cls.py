import torch
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import pickle
from tqdm import tqdm

from sequence_transformer import ASDTransformer
from dino_sequence_dataset import DinoSequenceDataset
from dino_pretrain_main import my_collate_fn

def extract_cls(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # === 数据加载 ===
    dataset = DinoSequenceDataset(args.data_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=my_collate_fn)

    # === 模型 ===
    model = ASDTransformer(mode="pretrain").to(device)
    state_dict = torch.load(args.weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"✅ Loaded weights from {args.weights_path}")

    # === 提取并存储 CLS ===
    all_cls = []  # 存储每个序列的 CLS + person_id + exp_id + ts

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting CLS"):
            s_a = batch['s_a'].to(device)
            s_s = batch['s_s'].to(device)
            s_d = batch['s_d'].to(device)

            s_a_idx = batch['s_a_idx'].to(device)
            s_s_idx = batch['s_s_idx'].to(device)
            s_d_idx = batch['s_d_idx'].to(device)

            s_a_idx_batch = batch['s_a_idx_batch'].to(device)
            s_s_idx_batch = batch['s_s_idx_batch'].to(device)
            s_d_idx_batch = batch['s_d_idx_batch'].to(device)

            student_mask = batch['student_mask'].to(device).bool()

            cls_out = model(s_a, s_s, s_d,
                             s_a_idx, s_s_idx, s_d_idx, 
                             student_mask,
                             s_a_idx_batch, s_s_idx_batch, s_d_idx_batch,
                             mask_d=args.maskd)
            # 如果返回的是 dict 或 tuple，取 cls 部分
            if isinstance(cls_out, dict):
                cls_out = cls_out['cls_output']
            elif isinstance(cls_out, tuple):
                cls_out = cls_out[0]

            for i in range(cls_out.shape[0]):
                all_cls.append({
                    "person_id": batch["person_id"][i],
                    "exp_id": batch["exp_id"][i],
                    "ts": batch["ts"][i],
                    "cls": cls_out[i].cpu().numpy()
                })

    # === 保存 ===
    output_path = Path(args.output_path) / "cls_features.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(all_cls, f)

    print(f"✅ Saved CLS features to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to pretrain.pt")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to pretrained model weights")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save extracted CLS (pkl)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--maskd", type=bool, default=False)
    args = parser.parse_args([
    '--data_path', "../dino_data/dino_sequence_data/pretrain.pt",
    '--pretrained_weights', '../dino_data/weights/20:100epoch pretrain/student_epoch20.pth',
    '--output_path', '../dino_data/output_dino',
    '--batch_size', '32',
    '--maskd', 'True'
    ])

    extract_cls(args)

