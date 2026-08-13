
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import regre_metric,class_metric #metric
# from sklearn.metrics import corruracy_score,roc_mae_score, confusion_matrix
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import h5py
import os
import time
import json
import pickle
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

import logging
logger = logging.getLogger('__main__')



from models import LLMBrainNet
from models.Ablation import static_mlp,OnlyLLM,OnlyTS,SingleNet,fusion_MOE,fusion_SUM,fusion_CAT ,communities8net,communities12net



class Exp_Main(Exp_Basic):
    def __init__(self, args,fold_i,shuffle_ix, train_index, test_index):
        super(Exp_Main, self).__init__(args)
        self.data = args.data
        self.fold_i = fold_i
        self.shuffle_ix= shuffle_ix
        self.train_index = train_index
        self.test_index = test_index

        self.embd_root=args.embd_root
        self.prompt_method = args.prompt_method
        self.llm_method = args.llm_method
        

    def _build_model(self):
        model_dict = {
            "LLMBrainNet":LLMBrainNet,
            "SingleNet":SingleNet,
            "fusion_MOE":fusion_MOE,
            "fusion_SUM":fusion_SUM,
            "fusion_CAT":fusion_CAT,
            "OnlyLLM":OnlyLLM,
            "OnlyTS":OnlyTS,
            "communities8net":communities8net,
            "communities12net":communities12net,
            "static_mlp":static_mlp
        }

        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args,self.shuffle_ix,self.fold_i,self.train_index,self.test_index, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    # def _select_optimizer(self):
    #     # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
    #     model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=min(self.args.train_epochs, 50), eta_min=1e-6)
    #     return model_optim,scheduler

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def _select_metric(self,pred, true):
        pred = np.array(pred).squeeze()
        true = np.array(true).squeeze()
        metric1,metric2 = regre_metric(pred, true)
        return metric1,metric2


    def vali(self, data, data_loader,criterion, flag):
        
        total_loss = 0.0
        true_list = []
        pred_list = []
       
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(data_loader):
        
                pred, true,moeloss = self._process_one_batch(batch_x, batch_y)
                
                loss = criterion(pred, true)+moeloss
                total_loss += loss.item()
                true_list.extend(true.detach().cpu().numpy())
                pred_list.extend(pred.detach().cpu().numpy())

        total_loss = total_loss/len(data_loader)

        metric1,metric2 = self._select_metric(pred_list,true_list)

        # regression
        logger.info("          {}   loss:{:.4f},    corr|||{:.4f}|||corr,   mae|||{:.4f}|||mae".format(flag,total_loss,metric1,metric2))

        return total_loss

    def train(self, path):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            
            time_now = time.time()
            train_loss = 0.0
            true_list = []
            pred_list = []
        
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                
                model_optim.zero_grad()
                pred, true, moeloss = self._process_one_batch(batch_x, batch_y)# torch.Size([16, 1])

                loss = criterion(pred, true)+moeloss
                loss.backward()
                model_optim.step()

                train_loss += loss.item()

                true_list.extend(true.detach().cpu().numpy())
                pred_list.extend(pred.detach().cpu().numpy())

            train_loss = train_loss/len(train_loader)

            logger.info("Epoch: {} cost time: {:.4f}".format(epoch + 1, time.time() - epoch_time))

            metric1,metric2 = self._select_metric(pred_list,true_list)
            logger.info("          train loss:{:.4f},    corr|||{:.4f}|||corr,   mae|||{:.4f}|||mae".format(train_loss,metric1,metric2))

            vali_loss = self.vali(vali_data, vali_loader, criterion,"val")
            test_loss = self.vali(test_data, test_loader, criterion,"test")

            
            # scheduler.step() 
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

            # adjust_learning_rate(model_optim, epoch + 1, self.args)

        # 早停时已保存最优模型，而当前不一定是最优模型，所以要先加载，再保存
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        state_dict = self.model.module.state_dict() if isinstance(self.model, DataParallel) else self.model.state_dict()
        torch.save(state_dict, path + '/' + 'checkpoint.pth')

        return self.model

    def test(self, path, save_pred=False, inverse=False):
        
        test_data, test_loader = self._get_data(flag='test')
        
        # 加载最优模型
        logger.info('loading model')
        self.model.load_state_dict(torch.load(path + '/' + 'checkpoint.pth'))

        true_list = []
        pred_list = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                pred, true,_ = self._process_one_batch(batch_x, batch_y)
                
                true_list.extend(true.detach().cpu().numpy()) 
                pred_list.extend(pred.detach().cpu().numpy())

        metric1,metric2 = self._select_metric(pred_list,true_list)
        logger.info("                                corr|||{:.4f}|||corr,   mae|||{:.4f}|||mae".format(metric1,metric2))
        return metric1,metric2, pred_list, true_list
    


    def _process_one_batch(self, batch_x, batch_y):

        x,embd = batch_x
        x =       x.to(self.device)
        embd = embd.to(self.device)
        y = batch_y.to(self.device)

        outputs,moeloss = self.model(x,embd)
        return outputs, y,moeloss