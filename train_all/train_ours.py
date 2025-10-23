import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import copy
import logging
import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
import torch
from collections import Counter
import csv
import pandas as pd
import time

from model.model_units import sparse_ETF_Classifier
from utils.local_training import LocalUpdate, globaltest, globaltest_radar
from utils.utils import DynamicWeightScheduler, add_noise, get_current_consistency_weight, aggregate_avg_probs, get_test_weight
from utils.client_selection import *
from utils.FedAvg import *
from dataset.dataset import get_dataset, get_speech
from model.build_model import set_model


def ours(args, writer, models_dir):
    user_id = list(range(args.n_clients))
    
    # ------------------------------ dataset ------------------------------
    logging.info(str(args))

    if args.dataset == "speech":
        dataset_train, dataset_test, dataset_auxiliary, dict_users = get_speech(args)
    else:
        dataset_train, dataset_test, dataset_auxiliary, dict_users = get_dataset(args)
    logging.info(
        f"train: {Counter(dataset_train.targets)}, total: {len(dataset_train.targets)}")
    logging.info(
        f"test: {Counter(dataset_test.targets)}, total: {len(dataset_test.targets)}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    
    # --------------------- Add Noise ---------------------------
    y_train = np.array(dataset_train.targets)
    y_train_tensor = torch.tensor(y_train).to(args.device)
    y_train_noisy, gamma_s, real_noise_level = add_noise(
        args, y_train, dict_users)
    dataset_train.targets = y_train_noisy
    noisy_num = np.sum(y_train != y_train_noisy)
    
    # --------------------- Build Models ---------------------------
    args.use_etf = True
    netglob, args = set_model(args)
    user_id = list(range(args.n_clients))
    img_num_per_class_per_client = {}
    trainer_locals = []
    for id in user_id:
        trainer_locals.append(LocalUpdate(
            args, id, copy.deepcopy(dataset_train), dict_users[id], y_train_tensor))
    
    # ------------------------ Stage 1: warm up ------------------------ 
    local_models_dir = "local_models_5e3"
    os.makedirs(local_models_dir, exist_ok=True)
    best_performance = 0.0
    # '''
    if args.warm: 
        for rnd in range(args.s1):
            w_locals, loss_locals = [], []
            for idx in user_id:  # training over the subset
                local = trainer_locals[idx]
                w_local, loss_local = local.train_ours_first(args,
                    net=copy.deepcopy(netglob).to(args.device), writer=writer)

                # store every updated model
                w_locals.append(copy.deepcopy(w_local))
                loss_locals.append(copy.deepcopy(loss_local))

            dict_len = [len(dict_users[idx]) for idx in user_id]
            w_glob_fl = FedAvg(w_locals, dict_len)
            netglob.load_state_dict(copy.deepcopy(w_glob_fl))

            # pred = globaltest(copy.deepcopy(netglob).to(
            #     args.device), dataset_test, args)
            # acc = accuracy_score(dataset_test.targets, pred)
            # bacc = balanced_accuracy_score(dataset_test.targets, pred)
            logging.info(
                "******** round: %d ********" % (rnd))
            # logging.info(
            #     "******** round: %d, acc: %.4f, bacc: %.4f ********" % (rnd, acc, bacc))
            # logging.info(cm)
            # writer.add_scalar(f'test/acc', acc, rnd)
            # writer.add_scalar(f'test/bacc', bacc, rnd)

            # if bacc > best_performance:
            #     best_performance = bacc
            # logging.info(f'best bacc: {best_performance}, now bacc: {bacc}')
            # logging.info('\n')

        torch.save(netglob.state_dict(),  models_dir +
                   f'/stage1_model_{rnd}.pth')

    # #  ------------------------ client selection ------------------------
    # model_path = f"outputs_{args.dataset}_{args.level_n_system}_{args.level_n_lowerb}_{args.level_n_upperb}/{args.exp}_{args.level_n_system}_{args.level_n_lowerb}_{args.level_n_upperb}_{args.local_ep}/models/stage1_model_{args.s1-1}.pth"
    # logging.info(
    #     f"********************** load model from: {model_path} **********************")
    # netglob.load_state_dict(torch.load(model_path))
    # clean_clients, noisy_clients = client_selection_fednoro(args, 
    #     netglob, dict_users, dataset_train, gamma_s)
    # clean_clients, noisy_clients = client_selection_ours(args,
    #     dataset_auxiliary, w_locals, 
    #     net = copy.deepcopy(netglob).to(args.device))
    # '''
    noisy_clients = (1,6,8,9,10,13,16,17)
    clean_clients = set(user_id) - set(noisy_clients)
    logging.info(
        f"selected noisy clients: {noisy_clients}, real noisy clients: {np.where(gamma_s>0.)[0]}")
    # clean_clients = list(set(user_id) - set(noisy_clients))
    logging.info(f"selected clean clients: {clean_clients}")
    
    # ------------------------ Stage 2: ------------------------ 

    model_path = f"outputs_{args.dataset}_{args.level_n_system}_{args.level_n_lowerb}_{args.level_n_upperb}/{args.exp}_{args.level_n_system}_{args.level_n_lowerb}_{args.level_n_upperb}_5/models/stage1_model_{args.s1-1}.pth"
    logging.info(
        f"********************** load model from: {model_path} **********************")
    # args.warm = 0
    netglob.load_state_dict(torch.load(model_path))
    BACC = []
    correct_acc = []
    correct_per = []
    


    for rnd in range(args.s1, args.rounds):
        start = time.perf_counter()
        
        correct_correction = 0
        total_correction = 0

        w_locals, loss_locals, class_avg_probs = {}, {}, {}
        w_kl, w_ce, w_neg = get_test_weight(rnd, args.rounds)
        for idx in clean_clients:  
            local = trainer_locals[idx]
            # pass
            w_local, loss_local, class_avg_prob = local.train_ours_second(args, rnd,
                net=copy.deepcopy(netglob).to(args.device), writer=writer)
            ### store every updated model
            w_locals[idx] = copy.deepcopy(w_local)
            loss_locals[idx] = copy.deepcopy(loss_local)
            class_avg_probs[idx] = copy.deepcopy(class_avg_prob)
            
        # ## calculate global confidence
        global_avg_prob = aggregate_avg_probs(args, class_avg_probs)
        thresholds = [global_avg_prob[c] for c in range(args.n_classes)]
            
        for idx in noisy_clients:  
            local = trainer_locals[idx]
            w_local, loss_local, correct, correct_total = local.train_ours_noise(args, w_kl, w_ce, w_neg, thresholds,
                local_net=copy.deepcopy(netglob).to(args.device), global_net=copy.deepcopy(netglob).to(args.device), 
                writer=writer) 
            # w_local, loss_local = local.train_FedNoRo_ours(args,
            #     student_net=copy.deepcopy(netglob).to(args.device), teacher_net=copy.deepcopy(netglob).to(args.device), 
            #     writer=writer, weight_kd=weight_kd)
            w_locals[idx] = copy.deepcopy(w_local)
            loss_locals[idx] = copy.deepcopy(loss_local)
            correct_correction = correct_correction + correct
            total_correction = total_correction + correct_total
                
        assert len(w_locals) == len(loss_locals) == args.n_clients
        dict_len = [len(dict_users[idx]) for idx in user_id]
        # w_glob_fl = KLAgg(args, 
        #     w_locals, dict_len, clean_clients, noisy_clients, dataset_auxiliary,
        #     net = copy.deepcopy(netglob).to(args.device))
        w_glob_fl = DaAgg(
            w_locals, dict_len, clean_clients, noisy_clients)
        end = time.perf_counter()
        run_time = end - start
        netglob.load_state_dict(copy.deepcopy(w_glob_fl))
        correction1 = correct_correction / total_correction
        correction = correct_correction / noisy_num
        correct_acc.append(correction)
        correct_per.append(correction1)
        # if rnd > 84: 
        # pred, class_accuracies = globaltest_radar(copy.deepcopy(netglob).to(
        #     args.device), dataset_test, args)
        pred = globaltest(copy.deepcopy(netglob).to(
            args.device), dataset_test, args)            
        acc = accuracy_score(dataset_test.targets, pred)
        bacc = balanced_accuracy_score(dataset_test.targets, pred)
        logging.info(
            "******** round: %d, acc: %.4f, bacc: %.4f ********" % (rnd, acc, bacc))
        if bacc > best_performance:
            best_performance = bacc
            logging.info(f'best bacc: {best_performance}, now bacc: {bacc}')
            logging.info('\n')
                # with open(f'fedloge_class_acc_round_{rnd}.csv', 'w', newline='') as f:
                #     writer = csv.writer(f)
                #     writer.writerow(['Class', 'Accuracy'])  # 表头
                #     for cls, acc in sorted(class_accuracies.items()):
                #         writer.writerow([cls, acc])
            # writer.add_scalar(f'test/acc', acc, rnd)
            # writer.add_scalar(f'test/bacc', bacc, rnd)
            BACC.append(bacc)
        # else:
        #     logging.info("******** round: %d ********" % (rnd))
# '''
    df = pd.DataFrame({
        'Round': range(11, len(correct_acc) + 11),  # 轮次从 1 开始
        'Percent': [f"{x:.4f}%" for x in correct_per],
        'Accuracy': [f"{x:.4f}%" for x in correct_acc]
    })
    
    df.to_csv('ours_relabel_final2.csv', index=False)  # 保存为 CSV
    # 保存为 Excel 文件（需要安装 openpyxl: pip install openpyxl）
    df.to_excel('ours_relabel_final2.xlsx', index=False)  # 保存为 Excel
    
    # save model
    # torch.save(netglob.state_dict(),  models_dir+f'/stage2_model_{rnd}.pth')

    BACC = np.array(BACC)
    logging.info("last:")
    logging.info(BACC[-10:].mean())
    logging.info("best:")
    logging.info(BACC.max())

    torch.cuda.empty_cache()
    # '''