import torch

# ========= 成对遮盖函数（每条序列单独 Bernoulli） =========
def mask_pair_as(mask: torch.Tensor,
                             a_idx: torch.Tensor,
                             s_idx: torch.Tensor,
                             ratio: float) -> torch.Tensor:
    """
    对每条序列单独计算可见 A/S 对数，随机遮掉相同比例（向下取整）。

    mask   (B, T)  bool  True=可见
    a_idx  (N, 2)  long  A token 索引 [batch, col]
    s_idx  (N, 2)  long  S token 索引 [batch, col]
    ratio  float   遮盖比例 0~1
    """
    assert a_idx.size(0) == s_idx.size(0), "A、S 数量需一致"

    # 当前仍可见且成对存在
    keep = mask[a_idx[:,0], a_idx[:,1]] & mask[s_idx[:,0], s_idx[:,1]]
    a_idx, s_idx = a_idx[keep], s_idx[keep]

    # ---- 按序列逐条处理 ----
    for b in torch.unique(a_idx[:,0]):
        sel = (a_idx[:,0] == b)             # 属于该 batch 的行
        n_pairs = sel.sum().item()
        if n_pairs == 0:
            continue
        k = int(n_pairs * ratio)            # 固定要遮掉多少对
        if k == 0:
            continue
        rand = torch.randperm(n_pairs, device=mask.device)[:k]
        rows_a, cols_a = a_idx[sel][rand].T
        rows_s, cols_s = s_idx[sel][rand].T
        mask[rows_a, cols_a] = False
        mask[rows_s, cols_s] = False
    return mask


# ========= 2.  测  试 =========
def main():
    B, steps = 3, 4              # 3 条序列，每条 4 个时间步
    # token 布局：CLS | A S D | A S D | ...
    T = 1 + 3 * steps            # 总 token 列数
    mask = torch.ones(B, T, dtype=torch.bool)

    a_pos, s_pos, d_pos = [], [], []
    for b in range(B):
        for t in range(steps):
            base = 1 + 3 * t     # CLS 后第一个位置
            a_pos.append([b, base    ])   # A
            s_pos.append([b, base + 1])   # S
            d_pos.append([b, base + 2])   # D

    a_idx = torch.tensor(a_pos, dtype=torch.long)
    s_idx = torch.tensor(s_pos, dtype=torch.long)
    d_idx = torch.tensor(d_pos, dtype=torch.long)

    # ---- 1) 遮 A/S 成对 (ratio=0.5) ----
    ratio = 0.5
    mask = mask_pair_as(mask, a_idx, s_idx, ratio)

    # ---- 2) 遮所有 D token ----
    mask[d_idx[:,0], d_idx[:,1]] = True      # 若仅想遮部分，可改成随机

    # ---- 打印结果 ----
    print("Mask matrix  (1=visible, 0=masked):")
    print(mask.int(), "\n")

    for b in range(B):
        cols_a = a_idx[a_idx[:,0]==b][:,1]
        cols_d = d_idx[d_idx[:,0]==b][:,1]
        vis_a  = mask[b, cols_a].sum().item()
        vis_d  = mask[b, cols_d].sum().item()
        tot_a  = cols_a.numel()
        print(f"batch {b}:  A/S masked {((1-vis_a/tot_a)*100):.0f}%"
              f"   D masked {((1-vis_d/cols_d.numel())*100):.0f}%")

    # 断言：每对 A/S 同步遮 & D 全遮
    for a, s, d in zip(a_idx, s_idx, d_idx):
        b, ca = a.tolist()
        _, cs = s.tolist()
        _, cd = d.tolist()
        assert mask[b, ca] == mask[b, cs], "A/S 对遮盖不同步"
        assert mask[b, cd] == False,       "D token 未遮掉"
    print("\n✅  测试通过 —— 每序列按固定比例遮 A/S，对应 A/S 成对同步，所有 D 已遮。")

if __name__ == "__main__":
    main()