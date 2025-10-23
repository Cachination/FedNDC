import copy
import random
import os
import sys
import shutil
import numpy as np
import pandas as pd
import logging
import math
from tensorboardX import SummaryWriter
from collections import defaultdict
from math import cos, pi, exp


from torch import nn
import torch
import torch.nn.functional as F
import traceback
from utils.generate_noisy import *
from sklearn.cluster import KMeans

from sklearn.metrics import balanced_accuracy_score

class DynamicWeightScheduler:
    def __init__(self, total_epochs):
        self.total_epochs = torch.tensor(total_epochs, dtype=torch.float32)
        self.init_kl=torch.tensor(0.7)   
        self.kl_decay = torch.tensor(0.98)    
        self.min_ce = torch.tensor(0.3)      
        self.max_ce = torch.tensor(0.7)
        self.init_neg = torch.tensor(0.2)  
        self.final_neg = torch.tensor(0.05)
        self.decay_epoch = torch.tensor(total_epochs * 0.6)

    def get_weights(self, epoch):
        epoch_tensor = torch.tensor(epoch, dtype=torch.float32)
        # 计算各项权重（未归一化）
        w_kl = self.init_kl * (self.kl_decay ** epoch_tensor)
        w_ce = self.min_ce + 0.5 * (self.max_ce - self.min_ce) * \
                (1 + math.cos(math.pi * (1 - epoch/self.total_epochs)))
        w_neg = self.init_neg if epoch < self.decay_epoch else self.final_neg
        weight_sum = w_kl + w_ce + w_neg
        w_kl = w_kl / weight_sum
        w_ce = w_ce / weight_sum
        w_neg = w_neg / weight_sum
        return w_kl.item(), w_ce.item(), w_neg.item()
    
    def get_dynamic_weights(self, epoch):
        lambda_ce = 1.0          # CE 损失权重，基准设为1
        lambda_kl_max = 0.5      # 最大的 KL 损失权重
        lambda_neg_max = 0.1     # 最大的负学习损失权重
        lambda_kl = lambda_kl_max * (epoch / self.total_epochs)
        lambda_neg = lambda_neg_max * (epoch / self.total_epochs)
        return lambda_ce, lambda_kl, lambda_neg
        
    
    def get_fixed_weights(self, epoch):
        # 计算各项权重（未归一化）
        w_ce = 1
        if epoch>70:
            w_kl = 0.5
            w_neg = 0.1
        else:
            w_kl = 0.2
            w_neg = 0.05
        weight_sum = w_kl + w_ce + w_neg
        w_kl = w_kl / weight_sum
        w_ce = w_ce / weight_sum
        w_neg = w_neg / weight_sum
        return w_kl, w_ce, w_neg
    
def get_test_weight(epoch, total_epochs, strategy='cosine'):
    t = (epoch-10) / total_epochs  # normalize

    # === 可调参数区 ===
    alpha_max = 1.0
    beta_min, beta_max = 0.1, 1.0
    gamma_min, gamma_max = 0.5, 0.5  # 固定 or 可调

    if strategy == 'cosine':
        alpha = alpha_max * (cos(pi * t) + 1) / 2
        beta = beta_min + (1 - cos(pi * t)) / 2 * (beta_max - beta_min)
        gamma = gamma_min  # or same as beta
    elif strategy == 'sigmoid':
        k = 10
        sigmoid_t = 1 / (1 + exp(-k * (t - 0.5)))
        alpha = alpha_max * (1 - sigmoid_t)
        beta = beta_min + sigmoid_t * (beta_max - beta_min)
        gamma = gamma_min + sigmoid_t * (gamma_max - gamma_min)
    else:
        raise ValueError("Unknown strategy")

    return alpha, beta, gamma
    
def add_noise(args, y_train, dict_users):
    assert args.dataset in ("ISIC", "cifar10", "cifar100", "speech" ), "Dataset set error"
    np.random.seed(args.seed)
    gamma_s = np.array([0.] * args.num_users)
    gamma_s[:int(args.level_n_system*args.num_users)] = 1.
    np.random.shuffle(gamma_s)
    gamma_c_initial = np.random.rand(args.num_users)
    gamma_c_initial = (args.level_n_upperb - args.level_n_lowerb) * \
        gamma_c_initial + args.level_n_lowerb
    gamma_c = gamma_s * gamma_c_initial
    y_train_noisy = copy.deepcopy(y_train)

    if args.n_type == "symmetric":
        y_train_noisy, real_noise_level = create_sym_noisy_label(args, y_train_noisy, dict_users, gamma_c, y_train)
    elif args.n_type == "asymmetric":
        y_train_noisy, real_noise_level = create_asym_noisy_label(args, y_train_noisy, dict_users, gamma_c, y_train)
    else:
        raise NotImplementedError

    return (y_train_noisy, gamma_s, real_noise_level)

def add_noise_balancefl(config, y_train, dict_users):
    assert config["dataset"]["name"] in ("ISIC", "cifar10", "cifar100", "speech" ), "Dataset set error"
    np.random.seed(config["fl_opt"]["seed"])
    gamma_s = np.array([0.] * config["fl_opt"]["num_clients"])
    gamma_s[:int(config["dataset"]["level_n_system"]*config["fl_opt"]["num_clients"])] = 1.
    np.random.shuffle(gamma_s)
    gamma_c_initial = np.random.rand(config["fl_opt"]["num_clients"])
    gamma_c_initial = (config["dataset"]["level_n_upperb"] - config["dataset"]["level_n_lowerb"]) * \
        gamma_c_initial + config["dataset"]["level_n_lowerb"]
    gamma_c = gamma_s * gamma_c_initial
    y_train_noisy = copy.deepcopy(y_train)

    if config["dataset"]["n_type"] == "symmetric":
        y_train_noisy, real_noise_level = create_sym_noisy_label_balancefl(config, y_train_noisy, dict_users, gamma_c, y_train)
    elif config["dataset"]["n_type"] == "asymmetric":
        y_train_noisy, real_noise_level = create_asym_noisy_label_balancefl(config, y_train_noisy, dict_users, gamma_c, y_train)
    else:
        raise NotImplementedError

    return (y_train_noisy, gamma_s, real_noise_level)


def sigmoid_rampup(current, begin, end):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    current = np.clip(current, begin, end)
    phase = 1.0 - (current-begin) / (end-begin)
    return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(rnd, begin, end):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return sigmoid_rampup(rnd, begin, end)


def get_output(loader, net, args, softmax=False, criterion=None):
    net.eval()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(args.device)
            labels = labels.to(args.device)
            labels = labels.long()
            if softmax == True:
                outputs = net(images)
                outputs = F.softmax(outputs, dim=1)
            else:
                outputs = net(images)
            if criterion is not None:
                loss = criterion(outputs, labels)
            if i == 0:
                output_whole = np.array(outputs.cpu())
                if criterion is not None:
                    loss_whole = np.array(loss.cpu())
            else:
                output_whole = np.concatenate(
                    (output_whole, outputs.cpu()), axis=0)
                if criterion is not None:
                    loss_whole = np.concatenate(
                        (loss_whole, loss.cpu()), axis=0)
    if criterion is not None:
        return output_whole, loss_whole
    else:
        return output_whole


def get_output_and_loss(loader, net, args, softmax=False, criterion=None, out_feature=False):
    net.eval()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(args.device)
            labels = labels.to(args.device)
            labels = labels.long()
            if softmax == True:
                outputs = net(images, out_feature)
                outputs = F.softmax(outputs, dim=1)
            else:
                outputs = net(images)
            if criterion is not None:
                loss = criterion(outputs, labels)
            if i == 0:
                output_whole = np.array(outputs.cpu())
                if criterion is not None:
                    loss_whole = np.array(loss.cpu())
            else:
                output_whole = np.concatenate(
                    (output_whole, outputs.cpu()), axis=0)
                if criterion is not None:
                    loss_whole = np.concatenate(
                        (loss_whole, loss.cpu()), axis=0)
    if criterion is not None:
        return output_whole, loss_whole
    else:
        return output_whole


def get_output_and_label(loader, net, args):
    net.eval()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(args.device)
            labels = labels.to(args.device)
            labels = labels.long()

            outputs = net(images)
            outputs = F.softmax(outputs, dim=1)

            if i == 0:
                output_whole = np.array(outputs.cpu())
                label_whole = np.array(labels.cpu())
            else:
                output_whole = np.concatenate(
                    (output_whole, outputs.cpu()), axis=0)
                label_whole = np.concatenate(
                    (label_whole, labels.cpu()), axis=0)

    return output_whole, label_whole


def cal_training_acc(prediction, noisy_labels, true_labels):
    prediction = np.array(prediction)
    noisy_labels = np.array(noisy_labels)
    true_labels = np.array(true_labels)

    acc_noisy = balanced_accuracy_score(noisy_labels, prediction)
    acc_true = balanced_accuracy_score(true_labels, prediction)

    return acc_noisy, acc_true


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_output_files(args):
    outputs_dir = 'outputs_' + str(args.dataset) + '_' + str(
        args.level_n_system) + '_' + str(args.level_n_lowerb) + '_' + str(args.level_n_upperb)
    if not os.path.exists(outputs_dir):
        os.mkdir(outputs_dir)
    exp_dir = os.path.join(outputs_dir, args.exp + '_' +
                           str(args.level_n_system) + '_' + str(args.level_n_lowerb) + '_' +
                           str(args.level_n_upperb) + '_' + str(args.local_ep))
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    models_dir = os.path.join(exp_dir, 'models')
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    logs_dir = os.path.join(exp_dir, 'logs')
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
    tensorboard_dir = os.path.join(exp_dir, 'tensorboard')
    if not os.path.exists(tensorboard_dir):
        os.mkdir(tensorboard_dir)
    code_dir = os.path.join(exp_dir, 'code')
    if os.path.exists(code_dir):
        shutil.rmtree(code_dir)
    # shutil.copytree('.', code_dir, ignore=shutil.ignore_patterns('.git'))

    logging.basicConfig(filename=logs_dir+'/logs.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    writer = SummaryWriter(tensorboard_dir)
    return writer, models_dir
    
def model_kl_div(args, state_dict_noisy, state_dict_clean, net, images_unlabeled):
    net.to(args.device)
    
    # 计算噪声模型的预测分布
    net.load_state_dict(state_dict_noisy)
    net.eval()
    with torch.no_grad():
        logits_noisy = net(images_unlabeled) / args.temperature
        pred_noisy = torch.softmax(logits_noisy, dim=1)
    
    # 计算干净模型的预测分布
    net.load_state_dict(state_dict_clean)
    net.eval()
    with torch.no_grad():
        logits_clean = net(images_unlabeled) / args.temperature
        pred_clean = torch.softmax(logits_clean, dim=1)
    
    # 计算 KL 散度：KL( pred_clean || pred_noisy )
    # 注意：F.kl_div 要求第一个输入为对数概率，第二个为目标概率分布
    kl = F.kl_div(pred_clean.log(), pred_noisy, reduction='batchmean')
    
    return kl.item()

def aggregate_avg_probs(args, class_avg_probs):
    classwise_values = defaultdict(list)

    for client_avg in class_avg_probs.values():
        for cls, prob in client_avg.items():
            classwise_values[cls].append(prob)

    aggregated_avg = {cls: sum(probs) / len(probs) for cls, probs in classwise_values.items()}
    aggregated_avg = {
        cls: min(max(prob, 2 / args.n_classes), 0.5) for cls, prob in aggregated_avg.items()
    }
    return aggregated_avg

def get_pseudo_class_num_list(loader, net, class_num_list, thresholds, y_train, args):
    # thresholds: list or tensor, len = n_classes
    net.eval()
    pseudo_class_num_list = class_num_list.copy()

    correct_correction = 0
    total_correction = 0
    
    for (indices, images, labels) in loader:
        images, labels = images.to(args.device), labels.to(args.device)
        indices = indices.to(args.device)
        with torch.no_grad():
            logits = net(images)
            prob = torch.softmax(logits.detach(), dim=-1)
            
            
            thresholds_tensor = torch.tensor(thresholds, device=images.device).unsqueeze(0)  # [1, n_classes]
            percent_exceed = (prob - thresholds_tensor) / thresholds_tensor # [batch_size, n_classes]
            mask = percent_exceed > 0  # bool tensor: [batch_size, n_classes]

            # 将小于阈值的类置为极小值，以避免被选为最大
            diff_masked = torch.where(mask, percent_exceed, torch.full_like(percent_exceed, -float('inf')))

            # 找到每个样本“超过最多的那个类”
            pseudo_labels = torch.argmax(diff_masked, dim=-1)  # [batch_size]

            # 找出至少有一个类超过阈值的样本
            valid_mask = mask.any(dim=-1)  # [batch_size]
            
            update_orig = labels[valid_mask]
            update_pseudo = pseudo_labels[valid_mask]         

            delta_orig = -torch.bincount(update_orig, minlength=args.n_classes)
            delta_pseudo = torch.bincount(update_pseudo, minlength=args.n_classes)

            delta = delta_orig + delta_pseudo
            pseudo_class_num_list = [x + y for x, y in zip(pseudo_class_num_list, delta.tolist())]

            # 计算伪标签与真实标签一致的数量（仅对 valid 样本）
            real_labels = torch.tensor(y_train)[indices[valid_mask]].to(labels.device)  # 正确标签
            correct_correction += (update_pseudo == real_labels).sum().item()
            total_correction += valid_mask.sum().item()
            
    return pseudo_class_num_list, correct_correction, total_correction
            