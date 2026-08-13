import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import scipy.io as scio
import pickle
import glob, os, re, h5py, numpy as np


class Dataset_fMRI(Dataset):
    def __init__(self,args,
                root_path,shuffle_ix,
                fold_i,train_index,test_index,
                flag='train'):
        assert flag in ["train", "val", "test"]

        self.flag = flag
        self.root_path = root_path
        self.shuffle_ix = shuffle_ix
        self.fold_i = fold_i
        self.train_index =train_index
        self.test_index= test_index

        self.network_id = args.network_id
        self.roi_net_pkl = args.roi_net_pkl

        self.data=args.data
        self.embd_root = args.embd_root
        self.prompt_method =args.prompt_method
        self.llm_method=args.llm_method

        self.__read_data__()
        
    def __getitem__(self, index):
        return (self.X[index],self.E[index]), self.Y[index]

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

        # 先打乱数据集，并保持数据和标签的打乱顺序一致，再划分十折--
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
        
        embed_path = f'{self.embd_root}/{self.prompt_method}_{self.llm_method}/'
        pattern = os.path.join(embed_path, f"{self.data}_fold{self.fold_i}_{self.flag}_*.h5")
        self.E = self.__read_embd__(pattern)

        self.X = torch.from_numpy(self.X.astype(np.float32))
        self.Y = torch.from_numpy(self.Y.astype(np.float32))
        self.E = torch.from_numpy(self.E.astype(np.float32))


     
    def __read_embd__(self,pattern):
        paths = glob.glob(pattern)
        file_paths = sorted(paths)
        emb_list = []
        for fp in file_paths:
            with h5py.File(fp, "r") as f:
                data =f['embeddings'][:]
                emb_list.append(data)
        emb_all = np.concatenate(emb_list, axis=0)
        return emb_all


    
