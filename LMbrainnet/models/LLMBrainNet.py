import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np

from models.tool import DenseChebConv,cos_IN2,BatchPearsonMatrix
from models.tool import build_edge_index_from_list,dense_adj_from_edges,edge_dropout

class Model(nn.Module):
 
    def __init__(self, args):
        super(Model, self).__init__()
        
        if args.llm_method in ("GPT2",'BERT','T5'):
            llm_dim=768
        if args.llm_method in ("CLIP"):
            llm_dim=512
    
        self.dropout = args.dropout

       
        self.enc_in =args.enc_in
        self.seq_len =args.seq_len
    
        self.d_ff = args.d_ff
        self.hgc =args.hgc
        self.lg =args.lg
        self.k =args.k

        self.trans_layer =args.trans_layer
        self.trans_indim =args.trans_indim
        self.trans_head =args.trans_head

        
        self.roi_learner= ROI_Learner(args)

        
        roi_net_pkl = args.roi_net_pkl
        with open(roi_net_pkl, "rb") as f:
            groups = pickle.load(f)

        
        self.ts_proj = nn.Sequential(
            nn.Linear(self.seq_len, self.d_ff),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_ff, self.d_ff)
        )
        self.llm_proj = nn.Sequential(
            nn.Linear(llm_dim, self.d_ff),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_ff, self.d_ff)
        )
        
        self.model_fused = EV_GCN_GroupAttn(
            num_nodes=self.enc_in, input_dim=self.d_ff, groups=groups,
            hgc=self.hgc, 
            lg=self.lg, # Chebyshev 卷积层数 叠层深度，信息传播更远   K=3, lg=3 → 每个节点最终整合 9 跳邻居
            K=self.k, # Chebyshev 阶数  每层能看多远的邻居. 
                # K=1  只看节点自身特征（无图结构传播）
                # K=2  看自身 + 一阶邻居（相当于经典 GCN）
                # k=3  看自身 + 一阶 + 二阶邻居（传播更远
            dropout=0.2, edge_dropout_p=0.1,
            connect_mode="clique", symmetrize=True, add_self_loops=False,
        )

        # out = sum([self.hgc for _ in range(self.lg)]) 
        
        self.proj = LinearAttentionPool(groups,num_layers=self.trans_layer,out_dim=self.trans_indim,nhead=self.trans_head)

        self.predictor  = nn.Linear(len(groups)*self.trans_indim, 1)
          
  
    def forward(self, ts, llm_embd):

        B, N, L = ts.shape

        ts_embd = self.roi_learner(ts)
        ts_embd = self.ts_proj(ts_embd)
        llm_embd = self.llm_proj(llm_embd)
        x= self.model_fused(ts_embd,llm_embd)
        x = self.proj(x)
        x = self.predictor(x.reshape(B,-1))
        return x,0



class LinearAttentionPool(nn.Module):

    def __init__(self,groups,num_layers=2,out_dim=64,nhead=4):

        super().__init__()

        self.dropout =0.2

        self.corr = BatchPearsonMatrix()
        self.layers = nn.ModuleDict()
        for id,k in enumerate(groups):
            num =len(k)
            # self.layers[str(id)] = nn.Linear(num* num, out_dim)
            self.layers[str(id)] = nn.Sequential(
                nn.Linear(num* num, out_dim),
                nn.GELU(),
                # nn.Dropout(self.dropout),
                # nn.Linear(out_dim, out_dim)
            )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_dim,
            nhead=nhead,
            batch_first=True,   # 让输入是 (B, S, E)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


    def forward(self, x_list):
        """
        mats_list: 长度 11 的 list
            每个元素 shape: (B, N_g, N_g)
        """
        outputs = []

        for id,x in enumerate(x_list):
            B, Ng, _ = x.shape
            x = self.corr(x)
            x = x.reshape(B, -1)  # flatten
            x = self.layers[str(id)](x)
            outputs.append(x)

        out =  torch.stack(outputs, dim=1)
        out = self.transformer(out)
        return out


class EV_GCN_GroupAttn(nn.Module):
    """
    内部:
      - 用 11 个索引列表分别构“小图”
      - 对 group g，假设该组节点索引为 idx_g，长度 = k_g
        * 小图节点数 = k_g
        * 邻接矩阵 A_g: [B, k_g, k_g]
        * 卷积在小图上做，得到 [B, k_g, lg*hgc]
      - 最后再 scatter 回全图 N 个节点上，输出仍为 [B, N, lg*hgc]
    """
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        groups,  # list[list[int]]，每个元素是“全图节点索引”
        hgc: int = 32,
        lg: int = 3,
        K: int = 3,
        dropout: float = 0.2,
        edge_dropout_p: float = 0.0,
        connect_mode: str = "chain",
        symmetrize: bool = True,
        add_self_loops: bool = False,
    ):
        super().__init__()
        self.N = num_nodes
        self.Fin = input_dim
        self.dropout = dropout
        self.edge_dropout_p = edge_dropout_p
        self.lg = lg
        self.K = K
        self.symmetrize = symmetrize
        self.add_self_loops = add_self_loops
        self.hgc = hgc  # 记录一下，方便后面用

        # 1) 每个 group 自己的“节点子集” & 对应小图的 edge_index（局部索引）
        #
        # group_node_indices[g]: 该组在“全图”中的节点索引 (global indices)，形状 [k_g]
        # edge_indices[g]: 在小图上的边 (local indices 0..k_g-1)，形状 [2, E_g]
        self.group_node_indices = nn.ParameterList()
        self.edge_indices = nn.ParameterList()

        for idx_list in groups:
            
            global_idx = torch.tensor(idx_list, dtype=torch.long)
            k_g = len(idx_list)

            # 对“小图节点”使用局部索引 0..k_g-1 来构造 edge_index
            local_idx_list = list(range(k_g))
            local_ei = build_edge_index_from_list(
                local_idx_list, num_nodes=k_g, mode=connect_mode
            )  # [2, E_g]，节点范围是 0..k_g-1

            self.group_node_indices.append(
                nn.Parameter(global_idx, requires_grad=False)
            )
            self.edge_indices.append(
                nn.Parameter(local_ei, requires_grad=False)
            )

        # 2) PAE（边权预测）（不变）
        self.edge_net = cos_IN2(input_dim=input_dim, dropout=dropout)

        # 3) 每个 group 一套卷积堆叠（不变）
        hidden = [hgc for _ in range(lg)]
        self.group_convs = nn.ModuleList()
        for _ in range(len(groups)):           # 对每个 group 建一套 conv
            convs_g = nn.ModuleList()
            for i in range(lg):
                in_ch = input_dim if i == 0 else hidden[i - 1]
                convs_g.append(DenseChebConv(in_ch, hidden[i], K=K, bias=False))
            self.group_convs.append(convs_g)

    def forward(self, ts_embd, llm_embd):
    
        device = ts_embd.device
        B, N, Fin = ts_embd.shape
        assert N == self.N and Fin == self.Fin

        x_in = F.dropout(ts_embd, p=self.dropout, training=self.training)

        group_embs = []
        for g_idx, (node_idx_param, ei_param) in enumerate(
            zip(self.group_node_indices, self.edge_indices)
        ):
            # 该组在“全图”中的节点索引（global indices）
            node_idx = node_idx_param.to(device)  # [k_g]
            k_g = node_idx.shape[0]  # 两两组合 k_g*(k_g-1)/2

            # 取出该组的小图节点特征（局部顺序与 idx_clean / local_idx_list 一致）
            x_in_g = x_in[:, node_idx, :]      # [B, k_g, Fin]
            llm_embd_g = llm_embd[:, node_idx, :]  # [B, k_g, Fin]

            # 该组的小图 edge_index（局部索引 0..k_g-1）
            edge_index_g = ei_param.to(device)     # [2, E_g]
            E_g = edge_index_g.shape[1]

            # 1) 构建该小图的邻接矩阵 A_g: [B, k_g, k_g]
            if E_g == 0:
                # 无边：退化为零邻接（可选自环）
                A_g = dense_adj_from_edges(
                    B=B,
                    N=k_g,
                    edge_index=edge_index_g,
                    w=torch.zeros(B, 0, device=device),
                    device=device,
                    symmetrize=self.symmetrize,
                    add_self_loops=self.add_self_loops,
                )
            else:
                src = edge_index_g[0]  # 局部索引 [E_g]
                dst = edge_index_g[1]

                # 利用小图节点上的 llm_embd_g 来预测边权
                x_src = llm_embd_g[:, src, :]       # [B, E_g, Fin]
                x_dst = llm_embd_g[:, dst, :]       # [B, E_g, Fin]
                w_g = self.edge_net(x_src, x_dst)   # [B, E_g]
                w_g = edge_dropout(w_g, self.edge_dropout_p, self.training)

                A_g = dense_adj_from_edges(
                    B=B,
                    N=k_g,
                    edge_index=edge_index_g,
                    w=w_g,
                    device=device,
                    symmetrize=self.symmetrize,
                    add_self_loops=self.add_self_loops,
                )

            # 2) 使用本组自己的卷积堆叠 + JK（在小图上做）
            convs_g = self.group_convs[g_idx]  # 取出该组的卷积列表
            h = F.gelu(convs_g[0](x_in_g, A_g))  # [B, k_g, hgc]
            jk = h
            for i in range(1, self.lg):
                h = F.dropout(h, p=self.dropout, training=self.training)
                h = F.gelu(convs_g[i](h, A_g))   # [B, k_g, hgc]
                jk = torch.cat([jk, h], dim=-1)  # [B, k_g, lg*hgc]

            group_embs.append(jk)  # [B, N, D]

        return group_embs




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
