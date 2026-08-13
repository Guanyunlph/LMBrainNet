import os
import numpy as np
import torch
from torch import randperm
import random
from sklearn.model_selection import KFold
import shutil
from exp.exp_main import Exp_Main
from options import Options
import json
import datetime
import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
import time
import datetime


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 对 cuBLAS 算子也强制可复现（两种值任选其一）
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'  # 或 ':4096:8'

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch 1.8+：遇到不可复现算子直接报错，帮你定位问题
    torch.use_deterministic_algorithms(True)

    # 避免 TF32 带来的数值不一致（不是随机，但不同硬件会略有差异）
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def main(args):

    start_time = time.time()

    set_seed(args.seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]


    Exp = Exp_Main

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y%m%d%H%M%S")
    # setting = '{}_{}_{}'.format(
    #     args.data,  
    #     args.model,  
    #     formatted_time
    #     )

    llm_config=f'{args.prompt_method}-{args.llm_method}'

    if args.network_id == None:
        setting = '{}_{}_{}_dff{}'.format(
            args.data,
            args.model, 
            llm_config,
            # args.e_layer,
            args.d_ff,
            # args.head
            )
    else:
        setting = '{}_{}_net{}_{}'.format(
            args.data, 
            args.model,
            args.network_id,
            llm_config,
    )

    d_output = args.d_output_root+"/"+setting
    if not os.path.exists(d_output):
        os.makedirs(d_output)
    args.checkpoints = args.d_output_root+"/"+setting+'/checkpoints'

    file_handler = logging.FileHandler(d_output +'/log.log', mode='w')
    logger.addHandler(file_handler)

    logger.info('Args in experiment:')
    logger.info(args)
    # 保存参数
    with open(os.path.join(d_output, "args.json"), 'w') as f:
        json.dump(vars(args), f, indent=True)

    metric1_list = []
    metric2_list = []
    Preds = []
    Trues = []

    subj_id = [i for i in range(args.d_subj_num)] 
    shuffle_ix = np.random.permutation(args.d_subj_num).tolist()
     
    kf = KFold(n_splits=args.fold)
    fold_i = 0  # 第几折
    for train_index, test_index in kf.split(subj_id):
        fold_i += 1
        logger.info('%s fold %s %s', '>' * 40, str(fold_i), '<' * 40)    

        # fold path
        fold_path = os.path.join(args.checkpoints,'fold'+str(fold_i))
        if not os.path.exists(fold_path):
            os.makedirs(fold_path)
        else:
            shutil.rmtree(fold_path)
            os.makedirs(fold_path)

        # init
        exp = Exp(args,fold_i,shuffle_ix, train_index, test_index)  # set experiments

        # train
        logger.info(f"{'~' * 20} start training {'~' * 20}")
        exp.train(fold_path)
        
        # test
        logger.info(f"{'~' * 20} start testing {'~' * 20}")
        metric1,metric2, pred, true = exp.test(fold_path, args.save_pred)

        # save fold result
        result = [(t.item(), p.item()) for t, p in zip(true, pred)]
        with open(f'{fold_path}/result.txt', 'w') as f:
            for t, p in result:
                f.write(f"{t}  {p}\n")                  

        metric1_list.append(metric1)
        metric2_list.append(metric2)
        
        Preds.append(np.array(pred).squeeze(1))
        Trues.append(np.array(true).squeeze(1))
    
    metric_1_mean = np.mean(metric1_list)
    metric_1_std  = np.std(metric1_list)
    metric_2_mean = np.mean(metric2_list)
    metric_2_std  = np.std(metric2_list)

    logger.info("\n%s %s %s", '>' * 40, 'fold_total', '<' * 40)
    logger.info("                    corr|||{:.3f}±{:.3f}|||corr,   mae|||{:.3f}±{:.3f}|||mae".format(metric_1_mean, metric_1_std, metric_2_mean,metric_2_std))
    logger.info("\n%s %s %s", '>' * 40, 'fold_total', '<' * 40)


    # preds = np.concatenate(Preds, axis=0)
    # Trues = np.concatenate(Trues, axis=0)
    # logger.info(len(preds))

    np.save(d_output + '/pred.npy', np.array(Preds, dtype=object))
    np.save(d_output + '/true.npy', np.array(Trues, dtype=object))

    with open(d_output + '/result.txt', 'w') as f:
        f.write(f"{metric_1_mean:.3f} {metric_1_std:.3f} {metric_2_mean:.3f} {metric_2_std:.3f}\n")

    end_time = time.time()
    elapsed = end_time - start_time
    logger.info("运行时间为" + str(datetime.timedelta(seconds=elapsed)))

    # 终止当前 logger 的记录
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    return metric_1_mean,metric_1_std,metric_2_mean,metric_2_std,setting

if __name__ == "__main__":

    options = Options()
    args = options.parse()

    # some specific
    # args.d_ff = 4 * args.d_model
    
    results = main(args)
    
    print(','.join(map(str, results))) # 这句话不能注释掉，要不然wandb接收结果时会报错


