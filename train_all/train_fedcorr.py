# python version 3.7.1
# -*- coding: utf-8 -*-

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import copy
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from sklearn.mixture import GaussianMixture
import torch.nn as nn
import logging
import time

from fedcorr.util.options import args_parser
from fedcorr.util.local_training import LocalUpdate, globaltest, globaltest_ours
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from fedcorr.util.fedavg import FedAvg
from fedcorr.util.util import add_noise, lid_term, lid_term_ISIC, get_output
from dataset.dataset import get_dataset, get_speech
from fedcorr.model.build_model import build_model

np.set_printoptions(threshold=np.inf)

"""
Major framework of noise FL
"""

def fedcorr(args):
    # parse args
    args = args_parser()
    print(args)

    logging.info(str(args))
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    rootpath = "./record/"
    if args.dataset == "speech":
        dataset_train, dataset_test, dict_users = get_speech(args)
    else:
        dataset_train, dataset_test, dict_users = get_dataset(args)

    # ---------------------Add Noise ---------------------------
    y_train = np.array(dataset_train.targets)
    y_train_noisy, gamma_s, real_noise_level = add_noise(args, y_train, dict_users)
    dataset_train.targets = y_train_noisy
    noisy_num = np.sum(y_train != y_train_noisy)

    if not os.path.exists(rootpath + 'txtsave/'):
        os.makedirs(rootpath + 'txtsave/')
    txtpath = rootpath + 'txtsave/%s_%s_NL_%.1f_LB_%.1f_Iter_%d_Rnd_%d_%d_ep_%d_Frac_%.3f_%.2f_LR_%.3f_ReR_%.1f_ConT_%.1f_ClT_%.1f_Beta_%.1f_Seed_%d' % (
        args.dataset, args.model, args.level_n_system, args.level_n_lowerb, args.iteration1, args.rounds1,
        args.rounds2, args.local_ep, args.frac1, args.frac2, args.lr, args.relabel_ratio,
        args.confidence_thres, args.clean_set_thres, args.beta, args.seed)

    if args.iid:
        txtpath += "_IID"
    else:
        txtpath += "_nonIID_p_%.1f_dirich_%.1f"%(args.non_iid_prob_class,args.alpha_dirichlet)
    if args.fine_tuning:
        txtpath += "_FT"
    if args.correction:
        txtpath += "_CORR"
    if args.mixup:
        txtpath += "_Mix_%.1f" % (args.alpha)

    f_acc = open(txtpath + '_acc.txt', 'a')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # build model
    netglob = build_model(args)
    net_local = build_model(args)

    client_p_index = np.where(gamma_s == 0)[0]
    client_n_index = np.where(gamma_s > 0)[0]
    criterion = nn.CrossEntropyLoss(reduction='none')
    LID_accumulative_client = np.zeros(args.num_users)
    # '''
    for iteration in range(args.iteration1):
        LID_whole = np.zeros(len(y_train))
        loss_whole = np.zeros(len(y_train))
        LID_client = np.zeros(args.num_users)
        loss_accumulative_whole = np.zeros(len(y_train))

        # ---------Broadcast global model----------------------
        if iteration == 0:
            mu_list = np.zeros(args.num_users)
        else:
            mu_list = estimated_noisy_level

        prob = [1 / args.num_users] * args.num_users

        for _ in range(int(1/args.frac1)):
            idxs_users = np.random.choice(range(args.num_users), int(args.num_users*args.frac1), p=prob)
            w_locals = []
            for idx in idxs_users:
                
                prob[idx] = 0
                if sum(prob) > 0:
                    prob = [prob[i] / sum(prob) for i in range(len(prob))]

                net_local.load_state_dict(netglob.state_dict())
                sample_idx = np.array(list(dict_users[idx]))
                dataset_client = Subset(dataset_train, sample_idx)
                loader = torch.utils.data.DataLoader(dataset=dataset_client, batch_size=100, shuffle=False)

                # proximal term operation
                mu_i = mu_list[idx]
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=sample_idx)
                w, loss = local.update_weights(net=copy.deepcopy(net_local).to(args.device), seed=args.seed,
                                                w_g=netglob.to(args.device), epoch=args.local_ep, mu=mu_i)

                net_local.load_state_dict(copy.deepcopy(w))
                w_locals.append(copy.deepcopy(w))
                
                # pred = globaltest_ours(copy.deepcopy(netglob).to(
                # args.device), dataset_test, args)
                # acc = accuracy_score(dataset_test.targets, pred)
                # bacc = balanced_accuracy_score(dataset_test.targets, pred)
                
                # # acc_t = globaltest(copy.deepcopy(net_local).to(args.device), dataset_test, args)
                # f_acc.write("iteration %d, client %d, acc: %.4f , bacc: %.4f \n" % (iteration, idx, acc, bacc))
                # logging.info(
                #     "******** iteration %d, client %d, acc: %.4f , bacc: %.4f" % (iteration, idx, acc, bacc))
                logging.info(
                    "******** iteration %d, client %d" % (iteration, idx))
                # f_acc.flush()

                local_output, loss = get_output(loader, net_local.to(args.device), args, False, criterion)
                if args.dataset == "ISIC":
                    LID_local = list(lid_term_ISIC(local_output, local_output))
                else:
                    LID_local = list(lid_term(local_output, local_output))
                LID_whole[sample_idx] = LID_local
                loss_whole[sample_idx] = loss
                LID_client[idx] = np.mean(LID_local)
            dict_len = [len(dict_users[idx]) for idx in idxs_users]
            w_glob = FedAvg(w_locals, dict_len)

            netglob.load_state_dict(copy.deepcopy(w_glob))

        LID_accumulative_client = LID_accumulative_client + np.array(LID_client)
        loss_accumulative_whole = loss_accumulative_whole + np.array(loss_whole)

        # Apply Gaussian Mixture Model to LID
        # gmm_LID_accumulative = GaussianMixture(n_components=2, random_state=args.seed).fit(
        #     np.array(LID_accumulative_client).reshape(-1, 1))
        # labels_LID_accumulative = gmm_LID_accumulative.predict(np.array(LID_accumulative_client).reshape(-1, 1))
        # clean_label = np.argsort(gmm_LID_accumulative.means_[:, 0])[0]

        # noisy_set = np.where(labels_LID_accumulative != clean_label)[0]
        # clean_set = np.where(labels_LID_accumulative == clean_label)[0]
        user_id = list(range(20))
        noisy_set = (1,6,8,9,10,13,16,17)
        selected_clean_idx = np.array(list(set(user_id) - set(noisy_set)))
        logging.info(
            f"selected noisy clients: {noisy_set}, real noisy clients: {np.where(gamma_s>0.)[0]}")
        estimated_noisy_level = np.zeros(args.num_users)

        for client_id in noisy_set:
            sample_idx = np.array(list(dict_users[client_id]))

            loss = np.array(loss_accumulative_whole[sample_idx])
            gmm_loss = GaussianMixture(n_components=2, random_state=args.seed).fit(np.array(loss).reshape(-1, 1))
            labels_loss = gmm_loss.predict(np.array(loss).reshape(-1, 1))
            gmm_clean_label_loss = np.argsort(gmm_loss.means_[:, 0])[0]

            pred_n = np.where(labels_loss.flatten() != gmm_clean_label_loss)[0]
            estimated_noisy_level[client_id] = len(pred_n) / len(sample_idx)
            y_train_noisy_new = np.array(dataset_train.targets)

        if args.correction:
            for idx in noisy_set:
                sample_idx = np.array(list(dict_users[idx]))
                dataset_client = Subset(dataset_train, sample_idx)
                loader = torch.utils.data.DataLoader(dataset=dataset_client, batch_size=100, shuffle=False)
                loss = np.array(loss_accumulative_whole[sample_idx])
                local_output, _ = get_output(loader, netglob.to(args.device), args, False, criterion)
                relabel_idx = (-loss).argsort()[:int(len(sample_idx) * estimated_noisy_level[idx] * args.relabel_ratio)]
                relabel_idx = list(set(np.where(np.max(local_output, axis=1) > args.confidence_thres)[0]) & set(relabel_idx))


                y_train_noisy_new = np.array(dataset_train.targets)
                y_train_noisy_new[sample_idx[relabel_idx]] = np.argmax(local_output, axis=1)[relabel_idx]
                dataset_train.targets = y_train_noisy_new

    # reset the beta,
    args.beta = 0

    # ---------------------------- second stage training -------------------------------
    # model_path = "./pretrain/cifar10_1.pth"
    # logging.info(
    #     f"********************** load model from: {model_path} **********************")
    
    # # 加载旧模型的 state_dict，包含 weight_vec 和 weight_norm
    # old_state_dict = torch.load(model_path, map_location='cpu')

    # # 替换为你实际的模型定义
    # new_state_dict = netglob.state_dict()

    # # 替换所有原始参数
    # for key in new_state_dict:
    #     if key == 'encoder.fc.weight':
    #         # 特别处理 encoder.fc.weight
    #         weight_vec = old_state_dict['encoder.fc.weight_vec']
    #         weight_norm = old_state_dict['encoder.fc.weight_norm']
    #         new_state_dict[key] = weight_vec * weight_norm
    #     else:
    #         # 直接从旧模型复制其他参数（如果存在）
    #         if key in old_state_dict:
    #             new_state_dict[key] = old_state_dict[key]

    # # 将更新后的权重加载到新模型中
    # netglob.load_state_dict(new_state_dict)
    
    if args.correction:
        relabel_idx_whole = []
        correct_correction = 0  # 成功纠正的数量
        total_correction = 0    # 尝试纠正的总数
        for idx in noisy_set:
            sample_idx = np.array(list(dict_users[idx]))
            dataset_client = Subset(dataset_train, sample_idx)
            loader = torch.utils.data.DataLoader(dataset=dataset_client, batch_size=100, shuffle=False)
            loss = np.array(loss_accumulative_whole[sample_idx])
            local_output, _ = get_output(loader, netglob.to(args.device), args, False, criterion)
            y_predicted = np.argmax(local_output, axis=1)
            relabel_idx = (-loss).argsort()[:int(len(sample_idx) * estimated_noisy_level[idx] * args.relabel_ratio)]
            relabel_idx = list(set(np.where(np.max(local_output, axis=1) > args.confidence_thres)[0]) & set(relabel_idx))

            relabel_idx_whole.append(sample_idx[relabel_idx])
            old_labels = y_train_noisy[sample_idx[relabel_idx]]  # 原来错误的标签
            new_labels = y_predicted[relabel_idx]                # 预测出的新标签
            true_labels = y_train[sample_idx[relabel_idx]]       # 真实标签
            # 找出原来就是错的（即 old_labels != true_labels）
            wrong_mask = (old_labels != true_labels) 
            # 在这些错的里面，新标签和真实标签相同的就是成功纠正
            corrected_mask = (new_labels == true_labels) & wrong_mask
            correct_correction += corrected_mask.sum()
            total_correction += wrong_mask.sum() 
            
            y_train_noisy_new = np.array(dataset_train.targets)
            y_train_noisy_new[sample_idx[relabel_idx]] = np.argmax(local_output, axis=1)[relabel_idx]
            dataset_train.targets = y_train_noisy_new
            
        # 最后计算纠正正确率
        if total_correction > 0:
            correction_per = correct_correction / noisy_num 
            correction_acc = correct_correction / total_correction
        else:
            correction_acc = 0.0
        logging.info("******** iteration stage correction per %.4f, correction acc %.4f" % (correction_per, correction_acc))       
             
    if args.fine_tuning:
        start1 = time.perf_counter()
        selected_clean_idx = np.where(estimated_noisy_level <= args.clean_set_thres)[0]
    
        prob = np.zeros(args.num_users) # np.zeros(100)
        prob[selected_clean_idx] = 1 / len(selected_clean_idx)
        m = max(int(args.frac2 * args.num_users), 1)  # num_select_clients
        m = min(m, len(selected_clean_idx))
        netglob = copy.deepcopy(netglob)
        # add fl training
        for rnd in range(args.rounds1):
            w_locals, loss_locals = [], []
            idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)
            for idx in idxs_users:  # training over the subset
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w_local, loss_local = local.update_weights(net=copy.deepcopy(netglob).to(args.device), seed=args.seed,
                                                           w_g=netglob.to(args.device), epoch=args.local_ep,  mu=0)
                w_locals.append(copy.deepcopy(w_local))  # store every updated model
                loss_locals.append(copy.deepcopy(loss_local))
            dict_len = [len(dict_users[idx]) for idx in idxs_users]
            w_glob_fl = FedAvg(w_locals, dict_len)
            netglob.load_state_dict(copy.deepcopy(w_glob_fl))
            end1 = time.perf_counter()
            run_time1 = end1 - start1
    
            # pred = globaltest_ours(copy.deepcopy(netglob).to(
            # args.device), dataset_test, args)
            # acc = accuracy_score(dataset_test.targets, pred)
            # bacc = balanced_accuracy_score(dataset_test.targets, pred)
            
            # # acc_t = globaltest(copy.deepcopy(net_local).to(args.device), dataset_test, args)
            # f_acc.write("fine tuning stage round %d, test acc  %.4f, bacc: %.4f \n" % (rnd, acc, bacc))
            # logging.info(
            #     "******** fine tuning stage round %d, test acc  %.4f, bacc: %.4f" % (rnd, acc, bacc))
            logging.info(
                "******** fine tuning stage round %d" % rnd)
            # f_acc.flush()

        if args.correction:
            relabel_idx_whole = []
            correct_correction = 0  # 成功纠正的数量
            total_correction = 0    # 尝试纠正的总数
            for idx in noisy_set:
                sample_idx = np.array(list(dict_users[idx]))
                dataset_client = Subset(dataset_train, sample_idx)
                loader = torch.utils.data.DataLoader(dataset=dataset_client, batch_size=100, shuffle=False)
                glob_output, _ = get_output(loader, netglob.to(args.device), args, False, criterion)
                y_predicted = np.argmax(glob_output, axis=1)
                relabel_idx = np.where(np.max(glob_output, axis=1) > args.confidence_thres)[0]
                
                relabel_idx_whole.append(sample_idx[relabel_idx])
                old_labels = y_train_noisy[sample_idx[relabel_idx]]  # 原来错误的标签
                new_labels = y_predicted[relabel_idx]                # 预测出的新标签
                true_labels = y_train[sample_idx[relabel_idx]]       # 真实标签
                # 找出原来就是错的（即 old_labels != true_labels）
                wrong_mask = (old_labels != true_labels) 
                # 在这些错的里面，新标签和真实标签相同的就是成功纠正
                corrected_mask = (new_labels == true_labels) & wrong_mask
                correct_correction += corrected_mask.sum()
                total_correction += wrong_mask.sum() 
                
                y_train_noisy_new = np.array(dataset_train.targets)
                y_train_noisy_new[sample_idx[relabel_idx]] = y_predicted[relabel_idx]
                dataset_train.targets = y_train_noisy_new
                
            # 最后计算纠正正确率
            if total_correction > 0:
                correction_per = correct_correction / noisy_num 
                correction_acc = correct_correction / total_correction
            else:
                correction_acc = 0.0
            logging.info("******** fine tuning stage correction per %.4f, correction acc %.4f" % (correction_per, correction_acc))

    # ---------------------------- third stage training -------------------------------
    # third stage hyper-parameter initialization
    m = max(int(args.frac2 * args.num_users), 1)  # num_select_clients
    prob = [1/args.num_users for i in range(args.num_users)]

    for rnd in range(args.rounds2):
        start2 = time.perf_counter()
        w_locals, loss_locals = [], []
        idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)
        for idx in idxs_users:  # training over the subset
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w_local, loss_local = local.update_weights(net=copy.deepcopy(netglob).to(args.device), seed=args.seed,
                                                        w_g=netglob.to(args.device), epoch=args.local_ep, mu=0)
            w_locals.append(copy.deepcopy(w_local))  # store every updated model
            loss_locals.append(copy.deepcopy(loss_local))

        # w_glob_fl = FedAvg(w_locals)  # global averaging
        # if args.iid:
        dict_len = [len(dict_users[idx]) for idx in idxs_users]
        w_glob_fl = FedAvg(w_locals, dict_len)
        netglob.load_state_dict(copy.deepcopy(w_glob_fl))
        end2 = time.perf_counter()
        run_time2 = end2 - start2
        if rnd > 29:
            pred = globaltest_ours(copy.deepcopy(netglob).to(
            args.device), dataset_test, args)
            acc = accuracy_score(dataset_test.targets, pred)
            bacc = balanced_accuracy_score(dataset_test.targets, pred)
            
            # acc_t = globaltest(copy.deepcopy(net_local).to(args.device), dataset_test, args)
            f_acc.write("third stage round %d, test acc  %.4f, bacc: %.4f \n" % (rnd, acc, bacc))
            logging.info(
                    "******** third stage round %d, test acc  %.4f, bacc: %.4f" % (rnd, acc, bacc))
            f_acc.flush()
        else:
            logging.info(
                    "******** third stage round %d" % rnd)

    torch.cuda.empty_cache()
