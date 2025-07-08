import os
import csv
import time
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm

from sequence_transformer import ASDTransformer
from dino_sequence_dataset import DinoSequenceDataset

# 兼容 AMP 支持 (PyTorch >=1.8)
try:
    from torch.cuda.amp import autocast, GradScaler
except ImportError:
    autocast = None
    GradScaler = None

# cudnn 加速（仅在 CUDA 可用时）
if torch.cuda.is_available():
    cudnn.benchmark = True


# ============ VICRegLoss 内联定义 ============
class VICRegLoss(nn.Module):
    def __init__(self, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, x, y):
        """
        x, y: [B, D] 特征向量
        """
        # === Invariance loss ===
        sim_loss = F.mse_loss(x, y)

        # === Variance loss ===
        std_x = torch.sqrt(x.var(dim=0) + 1e-04)
        std_y = torch.sqrt(y.var(dim=0) + 1e-04)
        std_loss = torch.mean(F.relu(1 - std_x)) + torch.mean(F.relu(1 - std_y))

        # === Covariance loss ===
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        cov_x = (x.T @ x) / (x.shape[0] - 1)
        cov_y = (y.T @ y) / (y.shape[0] - 1)

        cov_loss = (
            self.off_diagonal(cov_x).pow(2).sum() / x.shape[1]
            + self.off_diagonal(cov_y).pow(2).sum() / y.shape[1]
        )

        loss = (
            self.sim_coeff * sim_loss +
            self.std_coeff * std_loss +
            self.cov_coeff * cov_loss
        )
        return loss

    @staticmethod
    def off_diagonal(x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


# ============ DINOLoss 内联定义 ============
class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, center_momentum=0.9,
                 use_cls_token_only=False):
        super().__init__()
        self.student_temp = 0.18
        self.center_momentum = center_momentum
        self.use_cls_token_only = use_cls_token_only
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = torch.cat([
            torch.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            torch.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ])
        self.ncrops = ncrops

    def forward(self, student_output, teacher_output, epoch):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)
        temp = self.teacher_temp_schedule[epoch].to(teacher_output.device)
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss, n_loss_terms = 0, 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


def cancel_gradients_last_layer(epoch, model, freeze_epoch):
    if epoch >= freeze_epoch:
        return
    for name, param in model.named_parameters():
        if "last_layer" in name:
            param.grad = None

def clip_gradients(model, clip):
    norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = param.grad.data.norm(2)
            if torch.isnan(norm):
                continue
            norms.append(norm.item())
            torch.nn.utils.clip_grad_norm_(param, clip)
    return norms

def my_collate_fn(batch):
    return {
        "s_a": torch.stack([item['student_a_tensor'] for item in batch]),
        "s_s": torch.stack([item['student_s_tensor'] for item in batch]), 
        "s_d": torch.stack([item['student_d_tensor'] for item in batch]),  
        "t_a": torch.stack([item['teacher_a_tensor'] for item in batch]) ,
        "t_s": torch.stack([item['teacher_s_tensor'] for item in batch]) ,
        "t_d": torch.stack([item['teacher_d_tensor'] for item in batch]) ,
        "s_a_idx": torch.tensor(batch[0]["student_type_idx"]["A_idx"], dtype=torch.long),
        "s_s_idx": torch.tensor(batch[0]["student_type_idx"]["S_idx"], dtype=torch.long),
        "s_d_idx": torch.tensor(batch[0]["student_type_idx"]["D_idx"], dtype=torch.long),
        "t_a_idx": torch.tensor(batch[0]["teacher_type_idx"]["A_idx"], dtype=torch.long) ,
        "t_s_idx": torch.tensor(batch[0]["teacher_type_idx"]["S_idx"], dtype=torch.long) ,
        "t_d_idx": torch.tensor(batch[0]["teacher_type_idx"]["D_idx"], dtype=torch.long) ,
        "student_mask": torch.stack([item['student_mask'] for item in batch]),
        "teacher_mask": torch.stack([item['teacher_mask'] for item in batch])
    }
def cosine_fpd_loss(stu_list, tea_list):
    """
    stu_list / tea_list : 由若干 (B,D) tensor 组成的 tuple
    返回标量 FPD 损失
    """
    V = len(stu_list)
    if V == 1:                                     # 单视角特例
        return 1.0 - (stu_list[0] * tea_list[0].detach()).sum(dim=-1).mean()

    total = 0.0
    pairs = 0
    for i, q in enumerate(tea_list):
        for j, p in enumerate(stu_list):
            if i == j:
                continue
            total += (1.0 - (p * q.detach()).sum(dim=-1).mean())  # ← 已是标量 tensor
            pairs += 1
    return total / pairs
# ============ 主训练函数 ============
def train_dino(args):
    # 生成时间戳
    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 创建子目录
    pretrain_dir = Path(args.output_dir) / f"pretrain_{time_tag}"
    weights_dir = Path(pretrain_dir) / "weights"
    figures_dir = Path(pretrain_dir) / "figures"
    pretrain_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 初始化 CSV 路径
    csv_path = Path(pretrain_dir) / f"train_metrics.csv"
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "student_variance", "teacher_variance"])

    # 初始化曲线数据列表
    loss_curve = []
    student_variance_epoch_curve = []
    teacher_variance_epoch_curve = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 数据加载 ----
    dataset = DinoSequenceDataset(args.data_path)

    # 1) 统计每条样本的 target-D 标签
    labels = []
    for i in range(len(dataset)):
        t = dataset[i]["target_d_tensor"]
        labels.append(int(t.item()) if t is not None else -1)  # 用 -1 标记无标签

    labels = torch.tensor(labels)
    valid  = labels >= 0                        # 过滤掉 -1 的样本
    freq   = torch.bincount(labels[valid], minlength=4).float()

    # 2) 构建权重：权重 ∝ 1 / 频次
    weights = torch.zeros_like(labels, dtype=torch.float)
    weights[valid] = 1.0 / freq[labels[valid]]

    # 3) WeightedRandomSampler
    sampler = torch.utils.data.WeightedRandomSampler(
        weights      = weights,
        num_samples  = len(weights),    # 一个 epoch 看一遍原始样本数
        replacement  = True             # 必须 True 才能按权重放回采样
    )

    # 4) DataLoader 用 sampler，shuffle 要关掉
    dataloader = DataLoader(
        dataset,
        batch_size   = args.batch_size,
        sampler      = sampler,
        shuffle      = False,           # sampler 已经决定顺序
        num_workers  = 0,
        collate_fn   = my_collate_fn
    )


    # ---- 模型 ----
    student = ASDTransformer(mode="pretrain").to(device)
    teacher = ASDTransformer(mode="pretrain").to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    # ---- DINOLoss ----
    dino_loss = DINOLoss(
        args.out_dim,
        ncrops=2,
        warmup_teacher_temp=args.warmup_teacher_temp,
        teacher_temp=args.teacher_temp,
        warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs,
        nepochs=args.epochs,
    ).to(device)

    vicreg_loss_fn = VICRegLoss().to(device)

    # ---- 优化器 ----
    params_groups = [
        {"params": [p for n, p in student.named_parameters() if p.requires_grad and ("bias" not in n) and ("norm" not in n)], "weight_decay": 0.05},
        {"params": [p for n, p in student.named_parameters() if p.requires_grad and ("bias" in n or "norm" in n)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params_groups, lr=args.lr)

    momentum_schedule = [args.momentum_teacher] * args.epochs * len(dataloader)

    start_time = time.time()
    for epoch in range(args.epochs):
        student.train()
        total_loss = 0.0

        student_variances = []
        teacher_variances = []

        epoch_loss = []

        pbar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{args.epochs}]", leave=False)
        for it, batch in enumerate(pbar):
            s_a = batch['s_a'].to(device)
            s_s = batch['s_s'].to(device)
            s_d = batch['s_d'].to(device)

            t_a = batch['t_a'].to(device)
            t_s = batch['t_s'].to(device) 
            t_d = batch['t_d'].to(device) 

            s_a_idx = batch['s_a_idx'].to(device)
            s_s_idx = batch['s_s_idx'].to(device)
            s_d_idx = batch['s_d_idx'].to(device)

            t_a_idx = batch['t_a_idx'].to(device) 
            t_s_idx = batch['t_s_idx'].to(device) 
            t_d_idx = batch['t_d_idx'].to(device) 

            student_mask = batch['student_mask'].to(device).bool()
            teacher_mask = batch['teacher_mask'].to(device).bool()


            s_out = student(s_a, s_s, s_d, s_a_idx, s_s_idx, s_d_idx, student_mask)
            t_out = teacher(t_a, t_s, t_d, t_a_idx, t_s_idx, t_d_idx, teacher_mask)

            s_nor = F.normalize(s_out, dim=-1)   # (B, D)
            t_nor = F.normalize(t_out, dim=-1)   # (B, D)
            # print("s_out:", s_out.shape, "t_out:", t_out.shape)
            # dinoloss = dino_loss(s_out, t_out, epoch)
            # vicreg_loss = vicreg_loss_fn(s_out, t_out)
            # loss = dinoloss + vicreg_loss*0
            loss  = cosine_fpd_loss(s_nor, t_nor)
            student_var = s_out.var(dim=1).mean().item()  # [B, D] → scalar
            teacher_var = t_out.var(dim=1).mean().item()
            student_variances.append(student_var)
            teacher_variances.append(teacher_var)
            loss.backward()
            if args.clip_grad:
                clip_gradients(student, args.clip_grad)
            cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            optimizer.step()
            optimizer.zero_grad()

            # === EMA update teacher ===
            with torch.no_grad():
                m = momentum_schedule[it]
                for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                    param_k.data.mul_(m).add_((1. - m) * param_q.data)

            total_loss += loss.item()
            epoch_loss.append(loss.item())
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)

        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}")
        loss_curve.append(avg_loss)

        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = weights_dir / f"student_epoch{epoch+1}.pth"
            torch.save(student.state_dict(), ckpt_path)
            ckpt_path = weights_dir / f"teacher_epoch{epoch+1}.pth"
            torch.save(teacher.state_dict(), ckpt_path)

        # === 记录 loss 和 variance ===
        avg_student_var = sum(student_variances) / len(student_variances)
        avg_teacher_var = sum(teacher_variances) / len(teacher_variances)
        student_variance_epoch_curve.append(avg_student_var)
        teacher_variance_epoch_curve.append(avg_teacher_var)

        # === 写入 CSV ===
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_loss, avg_student_var, avg_teacher_var])

        # === 每轮都画图保存 ===
        # 1. Loss 曲线
        plt.figure()
        plt.plot(loss_curve, label='Train Loss', color='tab:blue')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve".format(epoch + 1))
        plt.grid(True)
        plt.legend()
        plt.savefig(figures_dir / f"loss_until_epoch{epoch+1}.png")
        plt.close()

        # 2. Variance 曲线
        plt.figure()
        plt.plot(student_variance_epoch_curve, label='Student Variance', color='tab:orange')
        plt.plot(teacher_variance_epoch_curve, label='Teacher Variance', color='tab:green')
        plt.xlabel("Epoch")
        plt.ylabel("Variance")
        plt.title("Variance Curve".format(epoch + 1))
        plt.grid(True)
        plt.legend()
        plt.savefig(figures_dir / f"variance_until_epoch{epoch+1}.png")
        plt.close()

        # 3.loss in one epoch
        plt.figure()
        plt.plot(epoch_loss, label='Loss in one epoch', color='tab:red')
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.title(f"Loss in Epoch {epoch + 1}")
        plt.grid(True)
        plt.legend()
        plt.savefig(figures_dir / f"loss_in_epoch_{epoch+1}.png")
        plt.close()

        # 3. variances in one epoch
        plt.figure()
        plt.plot(student_variances, label='Student Variance in this epoch', color='tab:purple')
        plt.plot(teacher_variances, label='Teacher Variance in this epoch', color='tab:cyan')
        plt.xlabel("Batch")
        plt.ylabel("Variance")
        plt.title(f"Variance in one Epoch {epoch + 1}")
        plt.grid(True)
        plt.legend()
        plt.savefig(figures_dir / f"variance_in_epoch_{epoch+1}.png")
        plt.close()

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print("\nTraining complete in:", total_time)
    writer.close()


# ============ 主函数入口 ============
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out_dim", type=int, default=256)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--clip_grad", type=float, default=3.0)
    parser.add_argument("--freeze_last_layer", type=int, default=1)
    parser.add_argument("--momentum_teacher", type=float, default=0.996)
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float, help="Initial value for the teacher temperature.")
    parser.add_argument('--teacher_temp', default=0.07, type=float, help="Final value (after linear warmup) of the teacher temperature.")
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help="Number of warmup epochs for the teacher temperature.")

    args = parser.parse_args([
        "--data_path", "../dino_data/dino_sequence_data/pretrain.pt",
        "--output_dir", "../dino_data/output_dino",
        "--epochs", "100",
        "--batch_size", "32",
        "--lr", "1e-4",
        "--out_dim", "256",
        "--save_interval", "10",
        "--momentum_teacher", "0.99",
        "--warmup_teacher_temp", "0.08",
        "--teacher_temp", "0.10",
        "--warmup_teacher_temp_epochs", "25"
    ])
    train_dino(args)
    