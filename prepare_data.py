import json
import torch
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from parameter import IGN_LEN, TEACHER_TOKEN_LIMIT, VAL_IDS

# ==== 输入输出路径 ====
INPUT_PATH = Path("../processed_data/paper2/transformer_input.jsonl")
OUTPUT_DIR = Path("../dino_data/dino_sequence_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==== 读取 JSONL 数据 ====
data = []
with INPUT_PATH.open("r") as f:
    for line in f:
        sample = json.loads(line)
        data.append(sample)

# ==== 构造序列数据（不 padding，延迟到 Dataset 内部处理）====
sequences = []
sequences_finetune_2 = []
print(len(data), "条数据")
for i, sample in tqdm(enumerate(data), desc="Processing samples", total=len(data)):
    pid = sample['person_id']
    exp_id = sample['exp_type']
    ts = sample['ts']
    dis_seq = sample['dis_state_seq']
    v_seq = sample['v_state_seq']
    d_seq = sample['decision_seq']
    a_seq = sample['action_seq']

    min_len = min(len(dis_seq), len(v_seq), len(d_seq), len(a_seq))
    if min_len < 6:
        # print(len(dis_seq), f"跳过 {pid} 的数据，长度不足{IGN_LEN}")
        continue

    merged_seq = []
    for i in range(min(min_len, int(TEACHER_TOKEN_LIMIT / 3))):
        merged_seq.append({"type": "A", "value": a_seq[i]})
        merged_seq.append({"type": "S", "value": {"dis": dis_seq[i], "v": v_seq[i]}})
        merged_seq.append({"type": "D", "value": d_seq[i]})

    sample_dict = {
    "person_id": pid,
    "exp_id": exp_id,
    "ts": ts,
    "seq": merged_seq
    }

    if min_len < IGN_LEN:
        sequences_finetune_2.append(sample_dict)
    else:
        sequences.append(sample_dict)


"""方法1"""
# # ==== 数据划分 ====
# # 预训练：大部分人（42人）和他们的所有实验
# # 微调： 预训练没有见过的人共18人，其中训练集、验证集共计14人，可以出现重复的人，但是同一人在验证集中不能出现训练集中出现过的场景，测试集全新的4个人和他们的所有实验

# # ==== 确定 pretrain 数据 ====
# all_person_ids = list(set(s['person_id'] for s in sequences if s['person_id'] not in VAL_IDS))
# random.seed(42)
# finetune_person_ids = random.sample(all_person_ids, 14)

# pretrain = [s for s in sequences if s['person_id'] not in finetune_person_ids and s['person_id'] not in VAL_IDS]

# # ==== 微调 train/val 数据 ====
# finetune_candidates = [s for s in sequences if s['person_id'] in finetune_person_ids]
# # 合并 sequences_finetune_2 非 VAL_IDS 样本
# finetune_candidates += [s for s in sequences_finetune_2 if s['person_id'] not in VAL_IDS]

# finetune_train = []
# finetune_val = []

# # 按人划分，每人按场景分 7:3
# for pid in set(s['person_id'] for s in finetune_candidates):
#     person_samples = [s for s in finetune_candidates if s['person_id'] == pid]

#     # 按 exp_id 分组
#     exp_groups = {}
#     for sample in person_samples:
#         exp_id = sample['exp_id']
#         exp_groups.setdefault(exp_id, []).append(sample)

#     exp_ids = list(exp_groups.keys())
#     random.shuffle(exp_ids)

#     split_idx = int(len(exp_ids) * 0.7)
#     train_exp_ids = exp_ids[:split_idx]
#     val_exp_ids = exp_ids[split_idx:]

#     for eid in train_exp_ids:
#         finetune_train.extend(exp_groups[eid])
#     for eid in val_exp_ids:
#         finetune_val.extend(exp_groups[eid])

# # ==== 微调测试集 ====
# finetune_test = [s for s in sequences + sequences_finetune_2 if s['person_id'] in VAL_IDS]

"""方法2"""
# ==== 数据划分：方法2 以场景为单位打乱，其中验证集全新的人和他们的所有场景====
# 测试集：严格规定的4个人的所有数据（含短数据）
finetune_test = [s for s in sequences + sequences_finetune_2 if s['person_id'] in VAL_IDS]

# 剩余数据：用于预训练和微调训练
remaining = [s for s in sequences if s['person_id'] not in VAL_IDS]

# 用 (person_id, exp_id) 作为唯一 ID
unique_ids = list(set((s['person_id'], s['exp_id']) for s in remaining))
random.shuffle(unique_ids)
split_idx = int(len(unique_ids) * 0.7)
pretrain_ids = set(unique_ids[:split_idx])
finetune_train_ids = set(unique_ids[split_idx:])

pretrain = [s for s in remaining if (s['person_id'], s['exp_id']) in pretrain_ids]
finetune_train = [s for s in remaining if (s['person_id'], s['exp_id']) in finetune_train_ids]

# 把短数据（非测试集的）都作为微调训练集
finetune_train += [s for s in sequences_finetune_2 if s['person_id'] not in VAL_IDS]

# 从微调训练集中分出 15% 作为验证集
random.shuffle(finetune_train)
split_idx_val = int(len(finetune_train) * 0.75)
finetune_val = finetune_train[split_idx_val:]
finetune_train = finetune_train[:split_idx_val]

# ==== 保存数据 ====
torch.save(pretrain, OUTPUT_DIR / "pretrain.pt")
torch.save(finetune_train, OUTPUT_DIR / "finetune_train.pt")
torch.save(finetune_val, OUTPUT_DIR / "finetune_val.pt")
torch.save(finetune_test, OUTPUT_DIR / "finetune_test.pt")

# ==== 打印信息 ====
print("数据处理完成")
print(f"预训练集样本数: {len(pretrain)}")
print(f"微调训练集样本数: {len(finetune_train)}")
print(f"微调验证集样本数: {len(finetune_val)}")
print(f"微调测试集样本数: {len(finetune_test)}")
