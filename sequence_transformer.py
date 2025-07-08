import torch
import torch.nn as nn

class ASDTransformer(nn.Module):
    def __init__(self, 
                 embed_dim=256, 
                 num_heads=8, 
                 depth=6, 
                 mlp_ratio=4.0, 
                 dropout=0.1,
                 num_decision_classes=4,
                 max_seq_len=150,
                 mode='pretrain'  # 'pretrain' or 'finetune'
                ):
        super().__init__()

        self.embed_dim = embed_dim
        self.mode = mode

        # === Embedding layers ===
        self.action_embed = nn.Linear(1, embed_dim)
        self.dis_embed = nn.Linear(10, embed_dim)
        self.v_embed = nn.Linear(3, embed_dim)
        self.decision_embed = nn.Embedding(num_embeddings=num_decision_classes, embedding_dim=embed_dim)

        # === Positional embedding ===
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len + 1, embed_dim))  # +1 for cls_token
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # === CLS token ===
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # === Transformer encoder ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # === Finetune aggregator ===
        finetune_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dropout=dropout,
            dim_feedforward=embed_dim * 2,  # 保持轻量
        )
        self.finetune_transformer = nn.TransformerEncoder(finetune_encoder_layer, num_layers=3)
        self.finetune_norm = nn.LayerNorm(embed_dim)

        # === Output heads ===
        self.decision_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_decision_classes)  # 分类
        )

        self.action_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1)  # 回归
        )
        self.decision_head_combined = nn.Sequential(
            nn.LayerNorm(embed_dim*4),
            nn.Linear(embed_dim*4, num_decision_classes)  # 分类
        )

        self.action_head_combined = nn.Sequential(
            nn.LayerNorm(embed_dim*4),
            nn.Linear(embed_dim*4, 1)  # 回归
        )

    def forward(self, a_tensor,s_tensor, d_tensor, a_idx, s_idx, d_idx, mask, finetune_type=0,mask_d=False):
        """
        Args:
            a_tensor: (total_A, 1)
            s_tensor: (total_S, 13)
            d_tensor: (total_D, 1)
            a_idx: (total_A, 2) [batch_idx, token_idx]
            s_idx: (total_S, 2)
            d_idx: (total_D, 2)
            mask: (B, T) bool
            finetune_use_combined: bool

        Returns:
            output: (B, D) if not finetune_use_combined else (B, T+1, D)
        """
        B, T = mask.shape
        device = a_tensor.device

        x = torch.zeros(B, T, self.embed_dim, device=device)

        if a_idx.numel() > 0:
            A_embed = self.action_embed(a_tensor)
            _, T_a, _ = A_embed.shape
            batch_idx = torch.arange(B, device=A_embed.device).view(B, 1).expand(B, T_a)  # [32, 30]

            # 放入 x
            x[batch_idx, a_idx] = A_embed

        if s_idx.numel() > 0:
            S_dis = s_tensor[:, :, :10]
            S_v = s_tensor[:, :, 10:]
            S_embed = self.dis_embed(S_dis) + self.v_embed(S_v)
            _, T_s, _ = S_embed.shape
            batch_idx = torch.arange(B, device=S_embed.device).view(B, 1).expand(B, T_s)  # [32, 30]
            # 放入 x
            x[batch_idx, s_idx] = S_embed


        if d_idx.numel() > 0:
            D_embed = self.decision_embed(d_tensor)
            _, T_d, _ = D_embed.shape
            batch_idx = torch.arange(B, device=D_embed.device).view(B, 1).expand(B, T_d)  # [32, 30]
            # 放入 x
            x[batch_idx, d_idx] = D_embed

        # 拼接 cls token
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat((cls_token, x), dim=1)

        # 拼接 mask
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)

        if mask_d:
            # 如果需要 mask D token
            mask = mask.clone()
            mask[:, d_idx[:, 1]] = False  # 将 D token 的位置设为 False
        full_mask = torch.cat((cls_mask, ~mask), dim=1)



        # 编码器
        x = x.transpose(0, 1)  # (T+1, B, D)
        x = self.transformer(x, src_key_padding_mask=full_mask)
        x = x.transpose(0, 1)  # (B, T+1, D)

        # === Output ===
        cls_output = x[:, 0, :]  # [B, D]


        if self.mode == "pretrain":
            return cls_output

        else:  # "finetune"
            if  finetune_type == 0:
                """策略1：用全局cls"""
                decision_logits = self.decision_head(cls_output)
                action_pred = self.action_head(cls_output)

                return decision_logits,action_pred
                
            elif finetune_type == 1:
                """策略2：用全局cls和最后的几个 A/S"""
                last_sec = 1
                last_seq = last_sec*10*3 -1
                last_x = x[:, -last_seq:, :]  # (B, 29, D)
                mask_a = torch.zeros(B, T+1, dtype=torch.bool, device=device)
                mask_s = torch.zeros(B, T+1, dtype=torch.bool, device=device)
                mask_d = torch.zeros(B, T+1, dtype=torch.bool, device=device)

                # 假设 s_a_idx: (total_A, 2)
                if a_idx.numel() > 0:
                    mask_a[a_idx[:,0], a_idx[:,1]] = True
                if s_idx.numel() > 0:
                    mask_s[s_idx[:,0], s_idx[:,1]] = True
                if d_idx.numel() > 0:
                    mask_d[d_idx[:,0], d_idx[:,1]] = True

                # 截取最后 29
                mask_a = mask_a[:, -last_seq:]
                mask_s = mask_s[:, -last_seq:]
                mask_d = mask_d[:, -last_seq:]

                last_a = (last_x * mask_a.unsqueeze(-1).float()).sum(1) / (mask_a.sum(1, keepdim=True).clamp(min=1e-6))
                last_s = (last_x * mask_s.unsqueeze(-1).float()).sum(1) / (mask_s.sum(1, keepdim=True).clamp(min=1e-6))
                last_d = (last_x * mask_d.unsqueeze(-1).float()).sum(1) / (mask_d.sum(1, keepdim=True).clamp(min=1e-6))

                combined = torch.cat([cls_output, last_a, last_s, last_d], dim=-1)  # (B, 4D)
                decision_logits = self.decision_head_combined(combined)
                action_pred = self.action_head_combined(combined)

                return decision_logits,action_pred
            
            elif finetune_type == 2:
                """策略3：所有的 A/S/D 加入复杂transformer"""
                agg_out = self.finetune_transformer(x[:, 1:, :], src_key_padding_mask = full_mask[:, 1:])  # 去掉 cls token
                agg_out = self.finetune_norm(agg_out)
                valid_mask = ~full_mask  # (B, T+1)，True 表示有效
                valid_mask = valid_mask.unsqueeze(-1).float()  # (B, T+1, 1)
                pooled = (agg_out * valid_mask).sum(1) / valid_mask.sum(1).clamp(min=1e-6)  # (B, D)
                decision_logits = self.decision_head(pooled)
                action_pred = self.action_head(pooled)

                return decision_logits,action_pred
            
            elif finetune_type == 3:
                """策略4：用全局cls和所有的 A/S/D 加入复杂transform"""
                agg_out = self.finetune_transformer(x, src_key_padding_mask = full_mask)  # 去掉 cls token
                agg_out = self.finetune_norm(agg_out)
                valid_mask = ~full_mask  # (B, T+1)，True 表示有效
                valid_mask = valid_mask.unsqueeze(-1).float()  # (B, T+1, 1)
                pooled = (agg_out * valid_mask).sum(1) / valid_mask.sum(1).clamp(min=1e-6)  # (B, D)
                decision_logits = self.decision_head(pooled)
                action_pred = self.action_head(pooled)

                return decision_logits,action_pred
                
