import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import copy
import logging
import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture
from collections import Counter
import glob
import yaml

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans


from utils.options import args_parser
from utils.utils import set_seed, set_output_files
from utils.client_selection import *
from utils.FedAvg import *
from train_all.train_ours import ours
from train_all.train_fednoro import fednoro
from train_all.train_bl import balancefl
from train_all.train_fedavg import fedavg
from train_all.train_fedcorr import fedcorr
from train_all.train_fedloge import fedloge
from train_all.train_fedned import fedned

import multiprocessing as mp

np.set_printoptions(threshold=np.inf)



"""
Train framework
"""

if __name__ == '__main__':
    
    args = args_parser()
    args.num_users = args.n_clients
    args.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
            
    # ------------------------------ deterministic or not ------------------------------
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        set_seed(args.seed)

    # ------------------------------ output files ------------------------------
    writer, models_dir = set_output_files(args)
        
    # ------------------------------ begin training ------------------------------
    set_seed(args.seed)
    logging.info("\n ---------------------begin training---------------------")
    best_performance = 0.

    # ------------------------ Stage 1: warm up ------------------------ 
    if "ours" in args.exp:
        ours(args, writer, models_dir)
    elif "fednoro" in args.exp:
        fednoro(args, writer, models_dir)
    elif "balancefl" in args.exp:
        balancefl(args)
    elif "fedcorr" in args.exp:
        fedcorr(args)   
    elif "fedloge" in args.exp:
        fedloge(args)
    elif "fedned" in args.exp:
        fedned(args)    
    elif "fedavg" in args.exp:
        fedavg(args, writer, models_dir)
