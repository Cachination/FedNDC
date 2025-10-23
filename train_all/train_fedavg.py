import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import copy
import logging
import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
import torch
from collections import Counter
import pandas as pd
import openpyxl
import time

from utils.local_training import LocalUpdate, globaltest
from utils.utils import DynamicWeightScheduler, add_noise
from utils.client_selection import *
from utils.FedAvg import *
from dataset.dataset import get_dataset, get_speech
from model.build_model import set_model

def fedavg(args, writer, models_dir):
    # ------------------------------ dataset ------------------------------
    logging.info(str(args))
    if args.dataset == "speech":
        dataset_train, dataset_test, dict_users = get_speech(args)
    else:
        dataset_train, dataset_test, dict_users = get_dataset(args)
    logging.info(
        f"train: {Counter(dataset_train.targets)}, total: {len(dataset_train.targets)}")
    logging.info(
        f"test: {Counter(dataset_test.targets)}, total: {len(dataset_test.targets)}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    
    # --------------------- Add Noise ---------------------------
    y_train = np.array(dataset_train.targets)
    y_train_noisy, gamma_s, real_noise_level = add_noise(
        args, y_train, dict_users)
    dataset_train.targets = y_train_noisy
    
    # --------------------- Build Models ---------------------------
    args.use_etf = False
    netglob, args = set_model(args)
    user_id = list(range(args.n_clients))
    trainer_locals = []
    for id in user_id:
        trainer_locals.append(LocalUpdate(
            args, id, copy.deepcopy(dataset_train), dict_users[id]))
    best_performance = 0.0    
    # ------------------------ Stage 1: warm up ------------------------             
    if args.warm:
        start = time.perf_counter()
        for rnd in range(args.s1):
            w_locals, loss_locals = [], []
            for idx in user_id:  # training over the subset
                local = trainer_locals[idx]
                w_local, loss_local = local.train_normal(args,
                    net=copy.deepcopy(netglob).to(args.device), writer=writer)

                # store every updated model
                w_locals.append(copy.deepcopy(w_local))
                loss_locals.append(copy.deepcopy(loss_local))

            w_locals_last = copy.deepcopy(w_locals)
            dict_len = [len(dict_users[idx]) for idx in user_id]
            w_glob_fl = FedAvg(w_locals, dict_len)
            netglob.load_state_dict(copy.deepcopy(w_glob_fl))
            end = time.perf_counter()
            run_time = end - start

            pred = globaltest(copy.deepcopy(netglob).to(
                args.device), dataset_test, args)
            acc = accuracy_score(dataset_test.targets, pred)
            bacc = balanced_accuracy_score(dataset_test.targets, pred)
            # cm = confusion_matrix(dataset_test.targets, pred)
            # logging.info(
            #     "******** round: %d ********" % (rnd))
            # logging.info(cm)
            # writer.add_scalar(f'test/acc', acc, rnd)
            # writer.add_scalar(f'test/bacc', bacc, rnd)
            logging.info(
                "******** round: %d, acc: %.4f, bacc: %.4f ********" % (rnd, acc, bacc))

            # if bacc > best_performance:
            #     best_performance = bacc
            # logging.info(f'best bacc: {best_performance}, now bacc: {bacc}')
            # logging.info('\n')
        # save model
        # torch.save(netglob.state_dict(),  models_dir +
        #            f'/stage1_model_{rnd}.pth')
    #'''

    # ------------------------ Stage 2: ------------------------ 
    # model_path = f"outputs_{args.dataset}_{args.level_n_system}_{args.level_n_lowerb}_{args.level_n_upperb}/{args.exp}_{args.level_n_system}_{args.level_n_lowerb}_{args.level_n_upperb}_{args.local_ep}/models/stage1_model_{args.s1-1}.pth"
    # logging.info(
    #     f"********************** load model from: {model_path} **********************")
    # netglob.load_state_dict(torch.load(model_path))
    BACC = []
    for rnd in range(args.s1, args.rounds):
        w_locals, loss_locals = [], []
        for idx in user_id:  # training over the subset
            local = trainer_locals[idx]
            w_local, loss_local = local.train_normal(args,
                net=copy.deepcopy(netglob).to(args.device), writer=writer)
            ### store every updated model
            w_locals.append(copy.deepcopy(w_local))
            loss_locals.append(copy.deepcopy(loss_local))
            assert len(w_locals) == len(loss_locals) == idx+1
            
        dict_len = [len(dict_users[idx]) for idx in user_id]
        w_glob_fl = FedAvg(w_locals, dict_len)
        netglob.load_state_dict(copy.deepcopy(w_glob_fl))

        if rnd > 84:
            pred = globaltest(copy.deepcopy(netglob).to(
                args.device), dataset_test, args)
            acc = accuracy_score(dataset_test.targets, pred)
            bacc = balanced_accuracy_score(dataset_test.targets, pred)
            cm = confusion_matrix(dataset_test.targets, pred)
            logging.info(
                "******** round: %d, acc: %.4f, bacc: %.4f ********" % (rnd, acc, bacc))
            # logging.info(cm)
            writer.add_scalar(f'test/acc', acc, rnd)
            writer.add_scalar(f'test/bacc', bacc, rnd)
            BACC.append(bacc)

            # save model
            if bacc > best_performance:
                best_performance = bacc
            logging.info(f'best bacc: {best_performance}, now bacc: {bacc}')
            logging.info('\n')
        else:
            logging.info(
                "******** round: %d ********" % (rnd))
    # torch.save(netglob.state_dict(),  models_dir+f'/stage2_model_{rnd}.pth')
    
    # df = pd.DataFrame({
    #     'Round': range(11, len(BACC) + 11),  # 轮次从 1 开始
    #     'Balanced Accuracy': [f"{x:.4f}%" for x in BACC]
    # })
    
    # df.to_csv('nonoi_bacc_history.csv', index=False)  # 保存为 CSV

    # # 保存为 Excel 文件（需要安装 openpyxl: pip install openpyxl）
    # df.to_excel('nonoi_bacc_history.xlsx', index=False)  # 保存为 Excel
    BACC = np.array(BACC)
    logging.info("last:")
    logging.info(BACC[-10:].mean())
    logging.info("best:")
    logging.info(BACC.max())

    torch.cuda.empty_cache()