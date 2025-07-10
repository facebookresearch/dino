import torch

# ---------- 固定布局下标 ----------
def build_fixed_idx(B: int, steps: int = 10):
    """
    返回 a_idx_batch (B,10), s_idx_batch (B,10), d_idx_batch (B,9)
    """
    # A、S 一定有 10 组
    cols_a = torch.arange(0, 3*steps, 3)[:10]      # 0,3,6,...,27
    cols_s = cols_a + 1                            # 1,4,7,...,28
    # D 只有 9 组（缺最后 1 个）
    cols_d = cols_a[:-1] + 2                       # 2,5,8,...,26

    a_idx = cols_a.unsqueeze(0).repeat(B, 1)       # (B,10)
    s_idx = cols_s.unsqueeze(0).repeat(B, 1)       # (B,10)
    d_idx = cols_d.unsqueeze(0).repeat(B, 1)       # (B,9)
    return a_idx, s_idx, d_idx


# ---------- 按 (B,L) 形式遮 A/S + 全遮 D ----------
def mask_pair_as_by_row(mask: torch.Tensor,
                        a_idx: torch.Tensor,
                        s_idx: torch.Tensor,
                        ratio: float) -> torch.Tensor:
    B, T = mask.shape
    assert a_idx.shape == s_idx.shape
    L = a_idx.size(1)

    for b in range(B):
        cols_a = a_idx[b]                # (L,)
        cols_s = s_idx[b]
        valid   = mask[b, cols_a] & mask[b, cols_s]   # 仍可见的成对位置
        cols_a, cols_s = cols_a[valid], cols_s[valid]

        k = int(cols_a.numel() * ratio)               # 要遮多少对
        if k:
            rand = torch.randperm(cols_a.numel(), device=mask.device)[:k]
            mask[b, cols_a[rand]] = False
            mask[b, cols_s[rand]] = False
    return mask


# ---------- 一键测试 ----------
def test():
    B, T = 32, 29                    # 32 条序列, 每条 29 token (不含 CLS)
    mask = torch.ones(B, T, dtype=torch.bool)

    a_idx, s_idx, d_idx = build_fixed_idx(B)         # (B,10) / (B,9)
    print('mask.shape', mask.shape,
          d_idx.shape, a_idx.shape, s_idx.shape)

    # 1) 遮掉 30 % 的 A/S 对
    mask = mask_pair_as_by_row(mask, a_idx, s_idx, ratio=0.3)
    # 2) 全遮 D
    rows = torch.arange(B).unsqueeze(1).expand_as(d_idx)
    mask[rows, d_idx] = False

    print('剩余可见 token 数:', mask.sum().item())    # 应明显 < B*T=928
    print('mask:\n', mask)

test()