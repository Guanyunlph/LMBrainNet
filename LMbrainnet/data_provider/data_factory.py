from data_provider.data_loader import Dataset_fMRI
from torch.utils.data import DataLoader
import logging
logger = logging.getLogger('__main__')


def data_provider(args,shuffle_ix,fold_i,train_index,test_index,flag):
    
    
    Data = Dataset_fMRI

    if args.data =='camcan-rest':
        data_root = '/data/CamCan-rest-timeSeries-FC-label.mat'
    elif args.data =='camcan-movie':
        data_root = '/data/CamCan-movie-timeSeries-FC-label.mat'
    elif args.data =='nki':
        data_root = '/data/NKI-rest-timeSeries-FC-label.mat'

    if flag == 'test':
        shuffle_flag = False
        drop_last = False  #  保留 最后的批次
        batch_size = args.batch_size
    elif flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size

    data_set = Data(
        args,        
        data_root, shuffle_ix,
        fold_i,train_index, test_index,
        flag=flag
    )

    logger.info("%s %d",flag, len(data_set))
    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    
    return data_set, data_loader

