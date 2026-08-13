
import torch
from torch import nn
import torch.nn.functional as F


class DenseChebConv(nn.Module):
    def __init__(self, in_channels, out_channels, K=3, bias=False):
        super().__init__()
        assert K >= 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.weight = nn.Parameter(torch.Tensor(K, in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)
        # nn.init.xavier_uniform_(self.weight.view(self.K, self.in_channels, self.out_channels))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @staticmethod  # 计算归一化拉普拉斯矩阵
    def normalized_laplacian(A, eps=1e-12):
        """
        A: [B, N, N]
        L = I - D^{-1/2} A D^{-1/2}
        """
        B, N, _ = A.shape
        device = A.device
        I = torch.eye(N, device=device).unsqueeze(0).expand(B, N, N)
        deg = A.sum(dim=-1)                                  # [B,N]
        deg_inv_sqrt = (deg + eps).rsqrt()                   # [B,N]
        Dn = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
        L = I - Dn
        return L

    def forward(self, x, A):
        """
        x: [B, N, Fin], A: [B, N, N]
        """
        B, N, Fin = x.shape
        assert Fin == self.in_channels
        L = self.normalized_laplacian(A)  # [B,N,N]

        T0 = x
        T_list = [T0]
        if self.K > 1:
            T1 = torch.matmul(L, x)
            T_list.append(T1)
            for _ in range(2, self.K):
                T2 = 2 * torch.matmul(L, T_list[-1]) - T_list[-2]
                T_list.append(T2)

        out = 0.0
        for k in range(self.K):
            out = out + T_list[k].matmul(self.weight[k])  # (B,N,Fin)@(Fin,Fout)
        if self.bias is not None:
            out = out + self.bias.view(1, 1, -1)
        return out  # [B,N,Fout]  # DenseChebConv 用 A_g @ X 输出 [B, N, H]


# -----------------------------
# 工具：从索引列表构边；从边+权重得稠密邻接
# -----------------------------
def build_edge_index_from_list(idx_list, num_nodes, mode: str = "clique") -> torch.Tensor:
    """
    根据一个索引列表构造 edge_index: [2, E]
    - "chain": 相邻连边（仅存一向，后续可对称化）
    - "clique": 完全图（上三角）
    """
    idx = list(dict.fromkeys(idx_list))  # 去重保序
    idx = [i for i in idx if 0 <= i < num_nodes]
    if len(idx) < 2:
        return torch.zeros(2, 0, dtype=torch.long)

    edges = []
    if mode == "chain":
        for u, v in zip(idx[:-1], idx[1:]):
            edges.append((u, v))
    elif mode == "clique":
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                edges.append((idx[i], idx[j]))
    else:
        raise ValueError(f"Unknown connect_mode: {mode}")

    ei = torch.tensor(edges, dtype=torch.long).t().contiguous()  # [2,E]
    return ei


def dense_adj_from_edges(B, N, edge_index, w, device, symmetrize=True, add_self_loops=False):
    """
    edge_index: [2, E]（单向）
    w: [B, E]
    返回: A: [B, N, N]
    """
    E = edge_index.shape[1]
    A = torch.zeros((B, N, N), device=device, dtype=w.dtype)
    if E == 0:
        if add_self_loops:
            I = torch.eye(N, device=device, dtype=w.dtype).unsqueeze(0).expand(B, N, N)
            A = A + I
        return A

    src = edge_index[0]  # [E]
    dst = edge_index[1]  # [E]
    b_idx = torch.arange(B, device=device).unsqueeze(-1).expand(B, E)  # [B,E]

    A[b_idx, src, dst] = w
    if symmetrize: # 让图的边变成无向的 
        A[b_idx, dst, src] = w
    if add_self_loops:
        I = torch.eye(N, device=device, dtype=w.dtype).unsqueeze(0).expand(B, N, N)
        A = A + I
    return A


def edge_dropout(w, p, training):
    """
    w: [B, E]
    """
    if p <= 0 or not training:
        return w
    keep = (F.dropout(torch.ones_like(w).unsqueeze(-1), p=p, training=True) > 0).squeeze(-1)
    return w * keep.float()




# -----------------------------
# PAE: 预测边权（端点特征拼接）
# 输入：一条边两端节点的特征
# 输出：这两个节点之间的相似度（权重）
# -----------------------------
class cos_IN2(nn.Module):
    def __init__(self, input_dim, dropout=0.2):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.input_dim = input_dim
    def forward(self, x1, x2):
        B,m,h =x1.shape
        x1 =x1.reshape(B * m, h)
        x2 =x2.reshape(B * m, h)
        p = (self.cos(x1, x2) + 1) * 0.5       
        p = p.reshape(B , m)
        return p



# class ROI_Learner(nn.Module):
#     def __init__(self, configs):
#         super().__init__()
#         self.seq_len = configs.seq_len
    
#         self.channels = configs.enc_in
#         self.hidden_size = 64
#         self.dropout=0.2

#         self.extract_common_pattern = nn.Sequential(
#             nn.Linear(self.channels, self.hidden_size),
#             nn.GELU(),
#             nn.Dropout(self.dropout),
#             nn.Linear(self.hidden_size, 1)
#         )

#         self.model_specific_pattern = nn.Sequential(
#             nn.Linear(self.channels, self.hidden_size),
#             nn.GELU(),
#             nn.Dropout(self.dropout),
#             nn.Linear(self.hidden_size, self.hidden_size),
#             nn.GELU(),
#             nn.Dropout(self.dropout),
#             nn.Linear(self.hidden_size, self.channels)
#         )

#     def forward(self, x):
#         B,_,_=x.shape
    
#         x = x.transpose(1, 2) # B T N
              
#         # 共享给所有通道的时间曲线
#         common_pattern = self.extract_common_pattern(x)  # [B, T, N] -> [B, T, 1]
           
#         specific_pattern = x - common_pattern.expand(-1, -1,self.channels)      # [B, T,n]

#         specific_pattern = self.model_specific_pattern(specific_pattern)   

#         x = specific_pattern.transpose(1, 2)

#         return x


class ROI_Learner(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
    
        self.channels = configs.enc_in
        self.hidden_size = 64
        self.dropout=0.2

        self.extract_common_pattern = nn.Sequential(
            nn.Linear(self.channels, self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1)
        )

        self.extract_pattern = nn.Sequential(
            nn.Linear(self.channels, self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.channels)
        )

        self.model_specific_pattern = nn.Sequential(
            nn.Linear(self.channels, self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.channels)
        )

    def forward(self, x):
        B,_,_=x.shape
    
        x = x.transpose(1, 2) # B T N
              
        # 共享给所有通道的时间曲线
        common_pattern = self.extract_common_pattern(x)  # [B, T, N] -> [B, T, 1
        
        specific_pattern = self.extract_pattern(x) - common_pattern.expand(-1, -1,self.channels)      # [B, T,n]

        specific_pattern = self.model_specific_pattern(specific_pattern)   

        x = specific_pattern.transpose(1, 2)

        return x




class BatchPearsonMatrix(nn.Module):
    def __init__(self, norm=False, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.norm = norm

    def forward(self, x):
        """
        x: (B, m, h)
        返回: (B, m, m)，每个 batch 一个 m×m 的 Pearson 相关矩阵
        """
        B, m, h = x.shape

        # 1. 按最后一维做去均值：对每个向量减掉自己的均值
        x_centered = x - x.mean(dim=-1, keepdim=True)      # (B, m, h)
        # 2. 按最后一维算范数：相当于 std（差一个常数因子，做相关时会相互抵消）
        x_norm = torch.norm(x_centered, dim=-1, keepdim=True) + self.eps  # (B, m, 1)
        # 3. 归一化：每个向量变成“零均值 + 单位范数”
        x_normalized = x_centered / x_norm                 # (B, m, h)
        # 4. 两两点积：得到 m×m 的相关矩阵    # 对每个 batch: corr[b] = x_normalized[b] @ x_normalized[b]^T
        corr = torch.matmul(x_normalized, x_normalized.transpose(1, 2))  # (B, m, m)
        # 理论上 corr ∈ [-1, 1]，可以视作 Pearson 相关矩阵

       
        if self.norm  == True:
             # x: (B, m, m)
            x_min = corr.view(B, -1).min(dim=1)[0].view(B, 1, 1)
            x_max = corr.view(B, -1).max(dim=1)[0].view(B, 1, 1)
            corr = (corr - x_min) / (x_max - x_min + 1e-8)  # 防止除0
        else:
            pass

        # ---- Step 2: z-score of correlation matrix ----
        # mean = corr.mean(dim=(1, 2), keepdim=True)
        # std = corr.std(dim=(1, 2), keepdim=True) + self.eps
        # z_corr = (corr - mean) / std
        # z_corr = (corr - corr.mean(dim=-1, keepdim=True)) / (corr.std(dim=-1, keepdim=True) + self.eps)

        return corr


# class PAE(nn.Module):
#     def __init__(self, input_dim, dropout=0.2):
#         super().__init__()
#         self.input_dim = input_dim

#     def forward(self, x1, x2):
#         B, m, h = x1.shape

#         # reshape to (B*m, h)
#         x1 = x1.reshape(B * m, h)
#         x2 = x2.reshape(B * m, h)

#         # mean-centering
#         x1_centered = x1 - x1.mean(dim=1, keepdim=True)
#         x2_centered = x2 - x2.mean(dim=1, keepdim=True)

#         # numerator: dot product of centered vectors
#         numerator = (x1_centered * x2_centered).sum(dim=1)

#         # denominator: product of norms
#         denominator = (
#             torch.norm(x1_centered, dim=1) * torch.norm(x2_centered, dim=1)
#         ) + 1e-8  # avoid /0

#         pearson = numerator / denominator  # shape: (B*m)

#         # ☆ 如果下游把这个当“概率”用，强烈建议做这一步：
#         p = (pearson + 1.0) * 0.5                  # 映射到 [0, 1]
#         p = p.clamp(1e-6, 1.0 - 1e-6)              # 防止 log(0) 之类炸掉
        
#         p = pearson.reshape(B, m)
#         return p




from einops import rearrange
from torch import einsum
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, q_in, kv_in, dim, heads=4, dropout=0.):
        super().__init__()

        dim_head = dim // heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(q_in, dim, bias=False)
        self.to_k = nn.Linear(kv_in, dim, bias=False)
        self.to_v = nn.Linear(kv_in, dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, q_in),
            nn.Dropout(dropout)
        )

    def forward(self, x, source):
        b, n, _, h = *x.shape, self.heads
        q, k, v = self.to_q(x), self.to_k(source), self.to_v(source)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        scores = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = F.softmax(scores, dim=-1, dtype=scores.dtype) 

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)