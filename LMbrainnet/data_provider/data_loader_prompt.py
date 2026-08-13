import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import scipy.io as scio
import pickle

class Dataset_fMRI(Dataset):
    def __init__(self,args,
                root_path,shuffle_ix,
                train_index,test_index,
                flag='train'):
        assert flag in ["train", "val", "test"]

        self.flag = flag
        self.root_path = root_path
        self.shuffle_ix = shuffle_ix
        self.train_index =train_index
        self.test_index= test_index

        self.network_id = args.network_id
        self.roi_net_pkl = args.roi_net_pkl
        self.prompt_method = args.prompt_method
        self.dataname = args.data

        self.__read_data__()

        self._build_ROI_prompts(self.X, self.prompt_method)

    
    def __getitem__(self, index):
        return (self.X[index], self.prompts[index]), self.Y[index]

    def __len__(self):
        return len(self.Y)
    
    def __read_data__(self):

        # load data
        final_data = scio.loadmat(self.root_path)     # TS :  (S, N, T)     

        valid_indices = np.squeeze(final_data['age']) != -1   
        
        try:
            TS = np.array(final_data['timeSeries'])[valid_indices]
        except Exception as e:
            TS = np.array(final_data['ts'])[valid_indices]
        Age = final_data['age'][valid_indices]
        
        
        if self.network_id is not None:
            try:
                with open(self.roi_net_pkl, "rb") as f:
                    each_net_roiid = pickle.load(f)
                net_roi_index = np.array(each_net_roiid[self.network_id])
                TS = TS[:, net_roi_index, :]
            except Exception:
                pass

        TS = TS[self.shuffle_ix]        # (S, N, T)
        Age = Age[self.shuffle_ix]      # (S,)

        train_X, train_Y = TS[self.train_index], Age[self.train_index]
        test_X,  test_Y  = TS[self.test_index],  Age[self.test_index]

        # 标准化, 对每个时间点的
        for i in range(train_X.shape[2]):
            scaler = StandardScaler()
            scaler.fit(train_X[:, :, i])
            train_X[:, :, i] = torch.tensor(scaler.transform(train_X[:, :, i]))
            test_X[:, :, i] = torch.tensor(scaler.transform(test_X[:, :, i]))

        
        pro = int(train_Y.shape[0] * 0.9)
        if self.flag == "train":
            self.X, self.Y = train_X[:pro], train_Y[:pro]
        elif self.flag == "val":
            self.X, self.Y = train_X[pro:], train_Y[pro:]
        elif self.flag == "test":
            self.X, self.Y = test_X, test_Y


    def _build_ROI_prompts(self, X_nt, method):
        S, N, T = X_nt.shape
        # 向量化统计
        mins = X_nt.min(axis=2)                 # (S, N)
        maxs = X_nt.max(axis=2)
        meds = np.median(X_nt, axis=2)
        xm   = X_nt.mean(axis=2, keepdims=True)  # (S, N, 1)

        t = np.arange(T, dtype=X_nt.dtype)
        t_center = t - t.mean()
        denom = float((t_center ** 2).sum())
        slopes = ((X_nt - xm) * t_center).sum(axis=2) / denom  # (S, N)
        trend = np.full_like(slopes, "", dtype=object) 
        trend[slopes > 0] = "upward"
        trend[slopes < 0] = "downward"
        trend[slopes == 0] = "flat"

        def fmt(v): return f"{float(v):.3f}"

        self.prompts = []
        for s in range(S):
            ps = []
            for c in range(N):
                if method == "ROI-static":
                    p= (f"Signal in this region: min {fmt(mins[s, c])}, max {fmt(maxs[s, c])}, median {fmt(meds[s, c])}, {trend[s, c]} trend.")         
                else:
                    raise ValueError(f"Unknown prompt_method: {method}")
                ps.append(p)
            self.prompts.append(ps)
        # 访问方式：prompts[s][c] 是第 s 个样本、第 c 个通道的一条字符串 prompt



