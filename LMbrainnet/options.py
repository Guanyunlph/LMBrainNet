import argparse
import torch


class Options(object):

    def __init__(self):
        
        self.parser = argparse.ArgumentParser(description='')

        self.parser.add_argument('--seed', type=int, default=0)
        self.parser.add_argument('--fold', type=int, default=10)
        # basic config       
        self.parser.add_argument('--model', type=str, default='LLMBrainNet') #BSGN  Ablation6  BSGN

        self.parser.add_argument('--prompt_method', type=str, default='ROI-static') 
        self.parser.add_argument('--llm_method', type=str, default='BERT') 
        self.parser.add_argument('--embd_root', type=str, default='/data/gyun/project/LLMBrainNet/gen_prompt_embd/Embeddings')  # 16


        self.parser.add_argument('--data', type=str, default='camcan-movie', help='')
        self.parser.add_argument('--d_subj_num', type=int, default=563, help='')
        self.parser.add_argument('--seq_len', type=int, default=188, help='') 
        

        # self.parser.add_argument('--data', type=str, default='camcan-rest', help='')
        # self.parser.add_argument('--d_subj_num', type=int, default=595, help='')
        # self.parser.add_argument('--seq_len', type=int, default=256, help='')  

        # self.parser.add_argument('--data', type=str, default='nki', help='')
        # self.parser.add_argument('--d_subj_num', type=int, default=1137, help='')
        # self.parser.add_argument('--seq_len', type=int, default=115, help='')  

        self.parser.add_argument('--enc_in', type=int, default=264, help='')  
        self.parser.add_argument('--network_id', type=int, default=None)
        self.parser.add_argument('--roi_net_pkl', default='/data/gyun/project/each_net_roiid.pkl')

        # model define
        self.parser.add_argument('--d_output_root', default='./debug_result', help='')
       
        
        self.parser.add_argument("--d_llm", type=int, default=768, help="hidden dimensions")
        self.parser.add_argument('--d_ff', type=int, default=16)  # 16
        self.parser.add_argument('--k', type=int, default=5, help='')
        self.parser.add_argument('--hgc', type=int, default=16, help='')
        self.parser.add_argument('--lg', type=int, default=2, help='')
        self.parser.add_argument('--trans_layer', type=int, default=2, help='')
        self.parser.add_argument('--trans_indim', type=int, default=64, help='')
        self.parser.add_argument('--trans_head', type=int, default=4, help='')


        # self.parser.add_argument('--embed', type=str, default='fixed', help='time features encoding, options:[timeF, fixed, learned]')
        # self.parser.add_argument('--freq', type=str, default='s',
        #                 help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
        #                      'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
        # self.parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers  (N)') # 3
        # self.parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
        # self.parser.add_argument('--factor', type=int, default=1, help='attn factor')
        # self.parser.add_argument('--activation', type=str, default='gelu', help='activation')
        # self.parser.add_argument('--n_heads', type=int, default=4, help='num of heads, no use for WITRAN') 


    
        self.parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
        
        
        # optimization
        self.parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
        self.parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
        self.parser.add_argument('--train_epochs', type=int, default=200, help='train epochs')  # 200
        self.parser.add_argument('--patience', type=int, default=10, help='early stopping patience') # 3  20
        self.parser.add_argument('--learning_rate', type=float, default=1e-3, help='optimizer initial learning rate')# 5e-5
        self.parser.add_argument("--weight_decay", type=float, default=1e-3, help="weight decay rate")
   
        self.parser.add_argument('--lradj', type=str, default='type2', help='adjust learning rate')

        self.parser.add_argument('--save_pred', action='store_true', help='whether to save the predicted future MTS', default=True)

        self.parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
        self.parser.add_argument('--gpu', type=int, default=0, help='gpu')
        self.parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        self.parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')


        # others
        # self.parser.add_argument('--d_model', type=int, default=128, help='dimension of model hidden states (d_model)')
        # self.parser.add_argument('--n_heads', type=int, default=4, help='num of heads, no use for WITRAN') 
        # self.parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers  (N)') # 3
        # self.parser.add_argument('--d_layers', type=int, default=3, help='num of decoder layers, no use for WITRAN') #  
        # self.parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn, no use for WITRAN') # 4* d_model 

       
      


    def parse(self):
        args = self.parser.parse_args()
        return args
