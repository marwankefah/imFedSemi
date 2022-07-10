import argparse

def args_parser():
     parser = argparse.ArgumentParser()
     parser.add_argument('--root_path', type=str, default='/data/RSNA-ICH/', help='dataset root dir')
     parser.add_argument('--csv_file_train', type=str, default='data/brain_split/train.csv', help='training set csv file')
     parser.add_argument('--csv_file_val', type=str, default='data/brain_split/validation.csv', help='validation set csv file')
     parser.add_argument('--csv_file_test', type=str, default='data/brain_split/test.csv', help='testing set csv file')
     parser.add_argument('--dataset', type=str,  default='brain', help='dataset option')
     parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
     parser.add_argument('--drop_rate', type=int, default=0.2, help='dropout rate')
     parser.add_argument('--base_lr', type=float,  default=2e-4, help='maximum epoch number to train')
     parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
     parser.add_argument('--seed', type=int,  default=1338, help='random seed')
     parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
     parser.add_argument('--local_ep', type=int,  default=1, help='local epoch')
     parser.add_argument('--class_num', type=int,  default=5, help='number of classes')
     parser.add_argument('--sub_bank_num', type=int,  default=5, help='number of classes')
     parser.add_argument('--warmup', type=int,  default=30, help='epochs of train sup clients only')
     parser.add_argument('--rounds', type=int,  default=200, help='communication rounds')
     parser.add_argument('--model_path', type = str, default='models/imfed_semi/brain.pth', help='Path for loading the checkpoint')
     parser.add_argument('--hi_lp', type=float,  default=0.9, help='confidence for label proportion estimation degree')
     parser.add_argument('--lo_lp', type=float,  default=0.5, help='confidence for label proportion estimation degree')
     parser.add_argument('--label_uncertainty', type=str,  default='U-Ones', help='label type')
     parser.add_argument('--lambda-u', default=1, type=float,
                         help='coefficient of unlabeled loss')
     parser.add_argument('--threshold', default=0.95, type=float,
                         help='pseudo label threshold')
     parser.add_argument('--T', default=1, type=float,
                         help='pseudo label temperature')

     parser.add_argument('--label_uncertainty', type=str, default='U-Ones', help='label type')
     parser.add_argument('--ema_decay', type=float, default=0.999, help='ema_decay')
     parser.add_argument('--consistency', type=float, default=1, help='consistency')
     parser.add_argument('--consistency_rampup', type=float, default=30, help='consistency_rampup')

     args = parser.parse_args()
     return args
