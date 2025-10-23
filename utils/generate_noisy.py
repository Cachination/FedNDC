import copy
import random
import os
import sys
import shutil
import numpy as np
import pandas as pd
import logging
from tensorboardX import SummaryWriter

import torch
import torch.nn.functional as F

from sklearn.metrics import balanced_accuracy_score

def create_asym_noisy_label(args, y_train_noisy, dict_users, per_noise_ratio, y_train):
    real_noise_level = np.zeros(args.num_users)
    for i in np.where(per_noise_ratio > 0)[0]:
        sample_idx = np.array(list(dict_users[i]))
        prob = np.random.rand(len(sample_idx))
        noisy_idx = np.where(prob <= per_noise_ratio[i])[0]

        if args.dataset == "cifar10":
            ### generate according to predefined class flip relationships
            mapping = {9: 1, 1: 9, 2: 0, 0: 2, 3: 5, 5: 3, 4: 7, 7: 4, 6: 8, 8: 6}
            y_train_noisy[sample_idx[noisy_idx]] = [mapping[label] for label in y_train_noisy[sample_idx[noisy_idx]]]
        
        else :
            ### make each category adjacent
            y_train_noisy[sample_idx[noisy_idx]] = (y_train_noisy[sample_idx[noisy_idx]] + 1) % args.n_classes

        noise_ratio = np.mean(y_train[sample_idx] != y_train_noisy[sample_idx])
        logging.info("Client %d, noise level: %.4f (%.4f), real noise ratio: %.4f" % (
            i, per_noise_ratio[i], per_noise_ratio[i] * 0.9, noise_ratio))
        
        real_noise_level[i] = noise_ratio

    return y_train_noisy, real_noise_level


def create_sym_noisy_label(args, y_train_noisy, dict_users, per_noise_ratio, y_train):
    real_noise_level = np.zeros(args.num_users)
    for i in np.where(per_noise_ratio > 0)[0]:
        sample_idx = np.array(list(dict_users[i]))
        prob = np.random.rand(len(sample_idx))
        noisy_idx = np.where(prob <= per_noise_ratio[i])[0]
        y_train_noisy[sample_idx[noisy_idx]] = np.random.randint(
            0, args.n_classes, len(noisy_idx))
        noise_ratio = np.mean(
            y_train[sample_idx] != y_train_noisy[sample_idx])
        logging.info("Client %d, noise level: %.4f (%.4f), real noise ratio: %.4f" % (
            i, per_noise_ratio[i], per_noise_ratio[i] * 0.9, noise_ratio))
        real_noise_level[i] = noise_ratio
    return (y_train_noisy, real_noise_level)



def create_asym_noisy_label_balancefl(config, y_train_noisy, dict_users, per_noise_ratio, y_train):
    real_noise_level = np.zeros(config["fl_opt"]["num_clients"])
    for i in np.where(per_noise_ratio > 0)[0]:
        sample_idx = np.array(list(dict_users[i]))
        prob = np.random.rand(len(sample_idx))
        noisy_idx = np.where(prob <= per_noise_ratio[i])[0]

        if config["dataset"]["name"] == "cifar10":
            ### generate according to predefined class flip relationships
            mapping = {9: 1, 1: 9, 2: 0, 0: 2, 3: 5, 5: 3, 4: 7, 7: 4, 6: 8, 8: 6}
            y_train_noisy[sample_idx[noisy_idx]] = [mapping[label] for label in y_train_noisy[sample_idx[noisy_idx]]]
        
        else :
            ### make each category adjacent
            y_train_noisy[sample_idx[noisy_idx]] = (y_train_noisy[sample_idx[noisy_idx]] + 1) % config["dataset"]["num_classes"]

        noise_ratio = np.mean(y_train[sample_idx] != y_train_noisy[sample_idx])
        logging.info("Client %d, noise level: %.4f (%.4f), real noise ratio: %.4f" % (
            i, per_noise_ratio[i], per_noise_ratio[i] * 0.9, noise_ratio))
        
        real_noise_level[i] = noise_ratio

    return y_train_noisy, real_noise_level


def create_sym_noisy_label_balancefl(config, y_train_noisy, dict_users, per_noise_ratio, y_train):
    real_noise_level = np.zeros(config["fl_opt"]["num_clients"])
    for i in np.where(per_noise_ratio > 0)[0]:
        sample_idx = np.array(list(dict_users[i]))
        prob = np.random.rand(len(sample_idx))
        noisy_idx = np.where(prob <= per_noise_ratio[i])[0]
        y_train_noisy[sample_idx[noisy_idx]] = np.random.randint(
            0, config["dataset"]["num_classes"], len(noisy_idx))
        noise_ratio = np.mean(
            y_train[sample_idx] != y_train_noisy[sample_idx])
        logging.info("Client %d, noise level: %.4f (%.4f), real noise ratio: %.4f" % (
            i, per_noise_ratio[i], per_noise_ratio[i] * 0.9, noise_ratio))
        real_noise_level[i] = noise_ratio
    return (y_train_noisy, real_noise_level)