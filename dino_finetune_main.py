import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,random_split
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import csv
import argparse
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns

from sequence_transformer import ASDTransformer
from dino_sequence_dataset import DinoSequenceDataset

def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def my_collate_fn(batch):
    batch_num = len(batch)
    s_a_idx= torch.tensor(batch[0]["student_type_idx"]["A_idx"], dtype=torch.long)
    s_s_idx= torch.tensor(batch[0]["student_type_idx"]["S_idx"], dtype=torch.long)
    s_d_idx= torch.tensor(batch[0]["student_type_idx"]["D_idx"], dtype=torch.long)
    s_a_idx_batch = torch.stack([s_a_idx for i in range(batch_num)]).long()
    s_s_idx_batch = torch.stack([s_s_idx for i in range(batch_num)]).long()
    s_d_idx_batch = torch.stack([s_d_idx for i in range(batch_num)]).long()
    return {
        "s_a": torch.stack([item['student_a_tensor'] for item in batch]),
        "s_s": torch.stack([item['student_s_tensor'] for item in batch]), 
        "s_d": torch.stack([item['student_d_tensor'] for item in batch]),  
        "s_a_idx": s_a_idx,
        "s_s_idx": s_s_idx,
        "s_d_idx": s_d_idx,
        "s_a_idx_batch": s_a_idx_batch,
        "s_s_idx_batch": s_s_idx_batch,
        "s_d_idx_batch": s_d_idx_batch,
        "student_mask": torch.stack([item['student_mask'] for item in batch]),
        "target_d": torch.stack([item['target_d_tensor'] for item in batch]),
        "target_a": torch.stack([item['target_a_tensor'] for item in batch]),
        "person_id": [item['person_id'] for item in batch],
        "exp_id": [item['exp_id'] for item in batch],
        "ts": [item['ts'] for item in batch],
    }

def plot_confusion_matrix(all_targets, all_preds, epoch, figures_dir):
    # ------------- 定义标签顺序 ----------------
    # 保证顺序与 model 输出 / all_targets 的整数 id 完全一致！
    labels = ["passive", "backward", "synchronized", "forward"]

    # ------------- 计数字矩阵 ----------------
    cm = confusion_matrix(all_targets, all_preds, labels=range(len(labels)))

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm,
                annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel('Prediction');  plt.ylabel('Ground Truth')
    plt.tight_layout()
    plt.savefig(figures_dir / f"confusion_matrix_epoch{epoch+1}.png")
    plt.close()

    # ------------- 百分比矩阵 ----------------
    cm_perc = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)   # 行归一化
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_perc * 100,
                annot=True, fmt='.1f', cmap='Greens',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Percentage (%)'})
    plt.title("Confusion Matrix")
    plt.xlabel('Prediction');  plt.ylabel('Ground Truth')
    plt.tight_layout()
    plt.savefig(figures_dir / f"confusion_matrix_norm_epoch{epoch+1}.png")
    plt.close()


def train_finetune(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据加载
    train_dataset = DinoSequenceDataset(args.train_data_path)
    val_dataset = DinoSequenceDataset(args.val_data_path)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=my_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=my_collate_fn)

    student = ASDTransformer(mode="finetune").to(device)
    if args.pretrained_weights:
        state_dict = torch.load(args.pretrained_weights, map_location=device)
        student.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded pretrained weights from {args.pretrained_weights}")

    # === 模式：linear probing / full finetune ===
    if args.train_mode == "linear":
        print("🔒 Linear Probing: Only training prediction heads...")
        for name, param in student.named_parameters():
            if "decision_head" in name or "action_head" in name or "finetune_transformer" in name or 'finetune_norm' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif args.train_mode == "finetune":
        print("🔓 Full Fine-tuning: Training entire model...")
        for param in student.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown train_mode: {args.train_mode}")

    params_groups = [
        {"params": [p for n, p in student.named_parameters() if p.requires_grad and ("bias" not in n) and ("norm" not in n)], "weight_decay": 0.05},
        {"params": [p for n, p in student.named_parameters() if p.requires_grad and ("bias" in n or "norm" in n)], "weight_decay": 0.0},
    ]
    optimizer = optim.AdamW(params_groups, lr=args.lr)

    # === 路径和日志 ===
    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    finetune_dir = Path(args.output_dir) / f"finetune_{args.train_mode}_{time_tag}"
    weights_dir = finetune_dir / "weights"
    figures_dir = finetune_dir / "figures"
    finetune_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    csv_path = finetune_dir / f"finetune_metrics.csv"

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch",'phase',"decision_loss", "action_loss", "accuracy", "mae", "precision", "recall", "f1"])

    history = {"train":  { "d_loss":[], "a_loss":[], "acc":[], "mae":[], "f1":[], "precision":[], "recall":[]}, "val": { "d_loss":[], "a_loss":[], "acc":[], "mae":[], "f1":[], "precision":[], "recall":[]}}


    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()

    start_time = time.time()
    for epoch in range(args.epochs):
        for phase, loader in [("train", train_loader), ("val", val_loader)]:
            student.train() if phase == "train" else student.eval()

            total_d_loss = total_a_loss = 0.0
            total_correct = total_samples = total_mae = 0.0
            all_targets, all_preds = [], []

            pbar = tqdm(loader, desc=f"{phase.capitalize()} Epoch {epoch+1}/{args.epochs}")
            for batch in pbar:
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
                target_d = batch['target_d'].to(device)
                target_a = batch['target_a'].to(device).float()

                with torch.set_grad_enabled(phase == "train"):
                    d_out, a_out = student(s_a, s_s, s_d,
                                            s_a_idx, s_s_idx, s_d_idx, 
                                            student_mask,
                                            s_a_idx_batch, s_s_idx_batch, s_d_idx_batch,
                                            mask_d=args.maskd, 
                                            finetune_type=args.finetune_type)
                    # === 主任务 loss ===
                    loss_decision = ce_loss_fn(d_out, target_d.squeeze(-1).long())
                    loss_action = mse_loss_fn(a_out, target_a)

                    if phase == "train":
                        loss = loss_decision + loss_action
                        loss.backward()
                        clip_gradients(student, args.clip_grad)
                        optimizer.step()
                        optimizer.zero_grad()

                total_d_loss += loss_decision.item()
                total_a_loss += loss_action.item()
                pred_class = torch.argmax(d_out, dim=1)
                correct = (pred_class == target_d.squeeze(-1)).sum().item()
                total_correct += correct
                total_samples += target_d.size(0)
                mae = torch.mean(torch.abs(a_out.squeeze() - target_a.squeeze())).item()
                total_mae += mae * target_d.size(0)
                all_targets.extend(target_d.cpu().numpy())
                all_preds.extend(pred_class.cpu().numpy())
                pbar.set_postfix(d_loss=loss_decision.item(), a_loss=loss_action.item(), acc=correct/target_d.size(0), mae=mae)

            avg_d_loss = total_d_loss / len(loader)
            avg_a_loss = total_a_loss / len(loader)
            history[phase]["d_loss"].append(avg_d_loss)
            history[phase]["a_loss"].append(avg_a_loss)

            avg_acc = total_correct / total_samples
            avg_mae = total_mae / total_samples
            history[phase]["acc"].append(avg_acc)
            history[phase]["mae"].append(avg_mae)

            if phase == "val":
                plot_confusion_matrix(all_targets, all_preds, epoch, figures_dir)
                ckpt_path = weights_dir / f"student_finetune_epoch{epoch+1}.pth"
                torch.save(student.state_dict(), ckpt_path)

            precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro")
            history[phase]["f1"].append(f1)
            history[phase]["precision"] = precision
            history[phase]["recall"] = recall
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, phase, avg_d_loss, avg_a_loss, avg_acc, avg_mae, precision, recall, f1])

        # acc 和 mae 曲线，train和val都画
        plt.figure()
        plt.plot(history['train']['f1'], '--', label='Train Decision F1')
        plt.plot(history['train']['mae'], '--', label='Train Action MAE')
        plt.plot(history['val']['f1'], ':', label='Validation Decision F1')
        plt.plot(history['val']['mae'], ':', label='Validation Action MAE')
        plt.xlabel('Epoch')
        plt.ylabel('Metric')
        plt.legend()
        plt.grid(True)
        plt.title("Decision F1 and Action MAE Curve")
        plt.savefig(figures_dir / f"acc_mae_curve_epoch{epoch+1}.png")
        plt.close()

        # 主 loss 曲线
        plt.figure()
        plt.plot(history['train']['d_loss'], '--', label='Train Decision Loss')
        plt.plot(history['val']['d_loss'], '--', label='Validation Decision Loss')
        plt.plot(history['train']['a_loss'], ':', label='Train Action Loss')
        plt.plot(history['val']['a_loss'], ':', label='Validation Action Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.title("Finetune Loss Curves")
        plt.savefig(figures_dir / f"loss_curve_epoch{epoch+1}.png")
        plt.close()

        # loss acc 和 mae 曲线画一起
        plt.figure(figsize=(10, 6))
        plt.plot(history['train']['d_loss'], '--', label='Train Decision Loss')
        plt.plot(history['val']['d_loss'], '--', label='Validation Decision Loss')
        plt.plot(history['train']['a_loss'], ':', label='Train Action Loss')
        plt.plot(history['val']['a_loss'], ':', label='Validation Action Loss')
        plt.plot(history['train']['f1'], label='Train Decision F1')
        plt.plot(history['val']['f1'], label='Validation Decision F1')
        plt.plot(history['train']['mae'], label='Train Action MAE')
        plt.plot(history['val']['mae'], label='Validation Action MAE')
        plt.xlabel('Epoch')
        plt.ylabel('Metrics')
        plt.legend()
        plt.grid(True)
        plt.title("Finetune Performance Curves")
        plt.savefig(figures_dir / f"loss_acc_mae_curve_epoch{epoch+1}.png")
        plt.close()



        # 两两画图
        def save_pair_plot(train_list, val_list, name):
            plt.figure()
            plt.plot(train_list, label=f'Train {name}')
            plt.plot(val_list, label=f'Val {name}')
            plt.xlabel('Epoch')
            plt.ylabel(name)
            plt.legend()
            plt.grid(True)
            plt.title(f'Finetune {name} Curve')
            plt.savefig(figures_dir / f"{name.lower()}_curve_epoch{epoch+1}.png")
            plt.close()
        save_pair_plot(history['train']['d_loss'], history['val']['d_loss'], 'Decision Loss')
        save_pair_plot(history['train']['a_loss'], history['val']['a_loss'], 'Action Loss')
        save_pair_plot(history['train']['f1'], history['val']['f1'], 'Decision F1')
        save_pair_plot(history['train']['mae'], history['val']['mae'], 'Action MAE')

    total_time = str(timedelta(seconds=int(time.time() - start_time)))
    print(f"\nFinetuning complete in: {total_time}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, required=True, help='Path to finetune train dataset')
    parser.add_argument('--val_data_path', type=str, required=True, help='Path to finetune validation dataset')
    parser.add_argument('--pretrained_weights', type=str, default=None, help="Path to pretrained weights")
    parser.add_argument('--output_dir', type=str, help="Directory to save outputs")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--clip_grad', type=float, default=1.0)
    parser.add_argument('--train_mode', type=str, choices=['linear', 'finetune'], default='finetune',
                        help="Training mode: linear (linear probing) or finetune (full fine-tuning)")
    parser.add_argument('--finetune_type', type=int, default=0, choices=[0, 1,2,3]) # 0: 全局 cls, 1: 最后几个 A/S
    parser.add_argument("--maskd", action="store_true", help="Enable D masking.")


    # args = parser.parse_args([
    # '--train_data_path', '../dino_data/dino_sequence_data/finetune_train.pt',
    # '--val_data_path', '../dino_data/dino_sequence_data/finetune_val.pt',
    # '--pretrained_weights', '../dino_data/weights/20:100epoch pretrain/student_epoch20.pth',
    # '--output_dir', '../dino_data/output_dino',
    # '--finetune_type', '0'
    # ])
    # 7折交叉验证
    for i in range(7):
        print(f"Starting finetune fold {i+1}...")
        args = parser.parse_args([
            '--train_data_path', f'../dino_data/dino_sequence_data/finetune_fold{i}_train_.pt',
            '--val_data_path', f'../dino_data/dino_sequence_data/finetune_fold{i}_val.pt',
            '--pretrained_weights', '../dino_data/output_dino/pretrain_maskd/weights/student_epoch100.pth',
            '--output_dir', '../dino_data/output_dino',
            '--finetune_type', '0',
            '--maskd',
            '--train_mode', 'finetune',
        ])
        train_finetune(args)
    print("All folds finetuned successfully!")