import torch
from torch.utils.data import Dataset
from parameter import TEACHER_TOKEN_LIMIT, STUDENT_TOKEN_LIMIT

class DinoSequenceDataset(Dataset):
    def __init__(self, data_path, mode='pretrain', 
                 max_teacher_tokens=TEACHER_TOKEN_LIMIT, 
                 max_student_tokens=STUDENT_TOKEN_LIMIT - 1):
        self.data = torch.load(data_path)
        assert mode in ['pretrain', 'finetune', 'val']
        self.mode = mode
        self.max_teacher_tokens = max_teacher_tokens
        self.max_student_tokens = max_student_tokens

    def __len__(self):
        return len(self.data)
    
    def encode_type_id(self, seq):
        type_id_map = {"A": 0, "S": 1, "D": 2}
        return [type_id_map.get(token["type"], -1) for token in seq]  # -1 表示 PAD，用 mask 区分

    def pad_left(self, seq, target_len):
        # print('seq',len(seq),'seq[-6:]',seq[-6:])
        if target_len == len(seq):
            return seq, [1] * len(seq)
        pad_len = target_len - len(seq)
        # print(f"[Padding] Padding {pad_len} tokens to the left of the sequence, total length: {target_len}")
        pad_token = [{'type': 'A', 'value': 0.0},
                     {'type': 'S', 'value': {'dis': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'v': [0, 0, 0]}},
                     {'type': 'D', 'value': 0}]
        padded = pad_token * int(pad_len/3) + seq
        mask = [0] * pad_len + [1] * len(seq)
        # print(f"[Padding] Padded sequence length: {len(padded)}, mask length: {len(mask)}")
        return padded[:], mask[:]
    
    def value_to_tensor(self, seq):
        value_list = []
        A_idx, S_idx, D_idx = [], [], []
        for i, token in enumerate(seq):
            t = token['type']
            v = token['value']

            if t == 'A':
                val_tensor = torch.tensor([v], dtype=torch.float32)  # (1,)
                A_idx.append(i)

            elif t == 'S':
                dis = torch.tensor(v['dis'], dtype=torch.float32)  # (10,)
                vel = torch.tensor(v['v'], dtype=torch.float32)    # (3,)
                val_tensor = torch.cat([dis, vel]).view(-1)       # (13,)
                S_idx.append(i)

            elif t == 'D':
                val_tensor = torch.tensor(v, dtype=torch.long)   # (1,)
                D_idx.append(i)

            else:
                print(f"[Warning] Unknown token type: {t}, using zero tensor as fallback")
                val_tensor = torch.zeros(1, dtype=torch.float32)

            value_list.append(val_tensor)

        type_idx = {
            'A_idx': A_idx,
            'S_idx': S_idx,
            'D_idx': D_idx
        }

        return value_list, type_idx

    def __getitem__(self, idx):
        item = self.data[idx]
        # print(f"[Processing] Sample {idx}")
        token_seq = item['seq']
        person_id = item['person_id']
        exp_id = item['exp_id'].split('-')[1]  # 提取实验 ID 的后缀
        ts = item['ts']

        max_tokens = self.max_teacher_tokens if self.mode != 'finetune' else self.max_student_tokens
        # print('max_tokens:', max_tokens, 'len(token_seq):', len(token_seq))
        # print('token_seq:', token_seq[-6:])
        if self.mode == "pretrain":
            padded_seq, mask = self.pad_left(token_seq[-max_tokens:], max_tokens)
        else:
            padded_seq, mask = self.pad_left(token_seq[-max_tokens-4:-4], max_tokens)


        # 学生
        student_seq = padded_seq[:self.max_student_tokens]
        student_mask = torch.tensor(mask[:self.max_student_tokens], dtype=torch.bool)
        student_type_id = torch.tensor(self.encode_type_id(student_seq), dtype=torch.long)
        student_values, student_type_idx = self.value_to_tensor(student_seq)

        #分离 student A/S/D tensor
        student_a_tensor = torch.stack([student_values[i] for i in student_type_idx['A_idx']]) if student_type_idx['A_idx'] else None
        student_s_tensor = torch.stack([student_values[i] for i in student_type_idx['S_idx']]) if student_type_idx['S_idx'] else None
        student_d_tensor = torch.stack([student_values[i] for i in student_type_idx['D_idx']]) if student_type_idx['D_idx'] else None

        # 教师
        if self.mode == 'pretrain':
            teacher_seq = padded_seq
            teacher_mask = torch.tensor(mask, dtype=torch.bool)
            teacher_type_id = torch.tensor(self.encode_type_id(teacher_seq), dtype=torch.long)
            teacher_values, teacher_type_idx = self.value_to_tensor(teacher_seq)

            teacher_a_tensor = torch.stack([teacher_values[i] for i in teacher_type_idx['A_idx']]) if teacher_type_idx['A_idx'] else None
            teacher_s_tensor = torch.stack([teacher_values[i] for i in teacher_type_idx['S_idx']]) if teacher_type_idx['S_idx'] else None
            teacher_d_tensor = torch.stack([teacher_values[i] for i in teacher_type_idx['D_idx']]) if teacher_type_idx['D_idx'] else None
        else:
            teacher_seq = teacher_mask = teacher_type_id = teacher_values = teacher_type_idx = None
            teacher_a_tensor = teacher_s_tensor = teacher_d_tensor = None

        # 提取目标 D/A
        d_index = self.max_student_tokens if self.mode != 'finetune' else len(token_seq) - 4
        a_index = d_index + 1
        full_seq = padded_seq if self.mode != 'finetune' else token_seq

        target_d, target_a, target_d_tensor, target_a_tensor = None, None, None, None
        if d_index < len(full_seq) and full_seq[d_index]['type'] == 'D':
            target_d = full_seq[d_index]
            target_d_tensor = torch.tensor(target_d['value'], dtype=torch.long).view(1)

        if a_index < len(full_seq) and full_seq[a_index]['type'] == 'A':
            target_a = full_seq[a_index]
            target_a_tensor = torch.tensor(target_a['value'], dtype=torch.float32).view(1)

        output = {
            "student_seq": student_seq,
            "student_tensor": student_values,
            "student_mask": student_mask,
            "student_type_id": student_type_id,
            "student_type_idx": student_type_idx,
            "student_a_tensor": student_a_tensor,
            "student_s_tensor": student_s_tensor,
            "student_d_tensor": student_d_tensor,
            "target_d": target_d,
            "target_a": target_a,
            "target_d_tensor": target_d_tensor,
            "target_a_tensor": target_a_tensor,
            "person_id": person_id,
            "exp_id": exp_id,
            "ts": ts
        }

        if self.mode == 'pretrain':
            output.update({
                "teacher_seq": teacher_seq,
                "teacher_tensor": teacher_values,
                "teacher_mask": teacher_mask,
                "teacher_type_id": teacher_type_id,
                "teacher_type_idx": teacher_type_idx,
                "teacher_a_tensor": teacher_a_tensor,
                "teacher_s_tensor": teacher_s_tensor,
                "teacher_d_tensor": teacher_d_tensor
            })

        return output


if __name__ == "__main__":
    # 用法示例：
    # 加载 pretrain 数据

    # pretrain_dataset = DinoSequenceDataset("../dino_data/dino_sequence_data/pretrain.pt", mode='pretrain')
    # for i in range(100):
    #     sample = pretrain_dataset[i]
    #     print(sample['target_d'], sample['target_a'])
    # sample = pretrain_dataset[7]
    # print("--------------len",len(sample['teacher_seq']),len(sample['student_seq']))
    # print("--------------teacher_seq:", sample['teacher_seq'])
    # print("----------------student_seq:", sample['student_seq'])
    # print("---------------target_d:", sample['target_d'])
    # print("----------------target_a:", sample['target_a'])
    # print("----------------student student_type_idx:", sample['student_type_idx'])
    # print("----------------person_id:", sample['person_id'])
    # print("----------------exp_id:", sample['exp_id'])
    # print("----------------ts:", sample['ts'])

    # 加载 finetune 数据

    finetune_dataset = DinoSequenceDataset("../dino_data/dino_sequence_data/finetune_train.pt", mode='finetune')
    for i in range(100):
        sample = finetune_dataset[i]
        print(sample['target_d'], sample['target_a'])
    
    # sample = finetune_dataset[0]
    # print(sample['student_seq'][-1], sample['target_d'], sample['target_a'])
    # print("--------------len",len(sample['student_seq']))
    # print("----------------student_seq:", sample['student_seq'])
    # print("---------------target_d:", sample['target_d'])
    # print("----------------target_d_tensor:", sample['target_d_tensor'])
    # print("----------------target_a:", sample['target_a'])
    # print("----------------target_a_tensor:", sample['target_a_tensor'])
    # print("----------------person_id:", sample['person_id'])
    # print("----------------exp_id:", sample['exp_id'])
    # print("----------------ts:", sample['ts'])


