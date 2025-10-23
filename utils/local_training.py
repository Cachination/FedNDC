import logging
import numpy as np

import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from collections import defaultdict
from math import cos, pi

from model.model_units import ETFClassifier
from utils.losses import set_loss, LogitAdjust, LA_KD, LogitAdjust_our, LogitAdjust_our_add
from utils.utils import DynamicWeightScheduler, get_pseudo_class_num_list
from model.all_models import modify_last_layer

def globaltest(net, test_dataset, args):
    net.eval()
    outFeature = False
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    pred = np.array([])
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs = net(images, outFeature)
            _, predicted = torch.max(outputs.data, 1)
            pred = np.concatenate([pred, predicted.detach().cpu().numpy()], axis=0)
    return pred

def globaltest_radar(net, test_dataset, args):
    net.eval()
    outFeature = False
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    pred = np.array([])
    true_labels = np.array([])
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs = net(images, outFeature)
            _, predicted = torch.max(outputs.data, 1)
            pred = np.concatenate([pred, predicted.detach().cpu().numpy()], axis=0)
            true_labels = np.concatenate([true_labels, labels.detach().cpu().numpy()], axis=0)
            
    class_accuracies = {}
    unique_classes = np.unique(true_labels)
    for cls in unique_classes:
        # 获取当前类别的样本索引
        cls_mask = (true_labels == cls)
        # 计算当前类别的精度
        cls_acc = np.mean(pred[cls_mask] == true_labels[cls_mask])
        class_accuracies[cls] = cls_acc  
    return pred, class_accuracies


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return self.idxs[item], image, label

    def get_num_of_each_class(self, args):
        class_sum = np.array([0] * args.n_classes)
        for idx in self.idxs:
            label = self.dataset.targets[idx]
            class_sum[label] += 1
        return class_sum.tolist()


class LocalUpdate(object):
    def __init__(self, args, id, dataset, idxs, y_train=0):
        self.args = args
        self.id = id
        self.idxs = idxs
        self.local_dataset = DatasetSplit(dataset, idxs)
        self.class_num_list = self.local_dataset.get_num_of_each_class(self.args)
        logging.info(
            f'client{id} each class num: {self.class_num_list}, total: {len(self.local_dataset)}')
        self.epoch = 0
        self.iter_num = 0
        self.lr = self.args.first_lr
        self.first_lr = self.args.first_lr
        self.second_lr = self.args.second_lr
        self.temperature = args.temperature
        self.y_train = y_train
        
    def train_normal(self, args, net, writer):
        net.train()
        # set the optimizer
        self.optimizer = torch.optim.Adam(
            net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        
        batch_size = self.args.batch_size
        while len(self.local_dataset) % batch_size == 1:
            batch_size += 1
        self.ldr_train = DataLoader(
            self.local_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        # train and update
        epoch_loss = []
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for (_, images, labels) in self.ldr_train:
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                logits = net(images)
                loss = F.cross_entropy(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
                writer.add_scalar(
                    f'client{self.id}/loss_train', loss.item(), self.iter_num)
                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())

        return net.state_dict(), np.array(epoch_loss).mean()
    
        
    def train_LA(self, args, net, writer):
        net.train()
        # set the optimizer
        self.optimizer = torch.optim.Adam(
            net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        batch_size = self.args.batch_size
        while len(self.local_dataset) % batch_size == 1:
            batch_size += 1
        self.ldr_train = DataLoader(
            self.local_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        # train and update
        epoch_loss = []
        ce_criterion = LogitAdjust(args, cls_num_list=self.class_num_list)
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for (_, images, labels) in self.ldr_train:
                images, labels = images.to(self.args.device), labels.cuda().to(self.args.device)

                logits = net(images)
                loss = ce_criterion(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
                # writer.add_scalar(
                #     f'client{self.id}/loss_train', loss.item(), self.iter_num)
                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())

        return net.state_dict(), np.array(epoch_loss).mean()
    

    def train_FedNoRo(self, args, student_net, teacher_net, writer, weight_kd):
        student_net.train()
        teacher_net.eval()
        # set the optimizer
        outFeature = True
        active_layers = student_net.get_trained_params_list(args)
        # self.optimizer = torch.optim.SGD([{'params': active_layers, 'lr': self.lr}],
        #     lr=self.lr, momentum=0.9)
        # self.optimizer = torch.optim.SGD(student_net.parameters(),
        #     lr=self.lr, momentum=0.9)
        self.optimizer = torch.optim.Adam(
            student_net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        
        batch_size = self.args.batch_size
        while len(self.local_dataset) % batch_size == 1:
            batch_size += 1
        self.ldr_train = DataLoader(
            self.local_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        # train and update
        epoch_loss = []
        criterion = LA_KD(args, cls_num_list=self.class_num_list)
        
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for (img_idx, images, labels) in self.ldr_train:
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                logits = student_net(images)
                with torch.no_grad():
                    teacher_output = teacher_net(images)
                    soft_label = torch.softmax(teacher_output/0.8, dim=1)

                loss = criterion(logits, labels, soft_label, weight_kd)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
                # writer.add_scalar(
                #     f'client{self.id}/loss_train', loss.item(), self.iter_num)
                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())

        return student_net.state_dict(), np.array(epoch_loss).mean()

    def train_ours_first(self, args, net, writer):
        net.train()
        outFeature = True
        # set the optimizer
        active_layers = net.get_trained_params_list(args)
        # self.optimizer = torch.optim.SGD([{'params': active_layers, 'lr': self.lr}],
        #     lr=self.lr, momentum=0.9, weight_decay=args.weight_decay)
        self.optimizer = torch.optim.Adam(
            [{'params': active_layers, 'lr': self.first_lr}], lr=self.first_lr, betas=(0.9, 0.999), weight_decay=5e-4)
        # self.optimizer = torch.optim.SGD(net.parameters(),
        #     lr=self.lr, momentum=0.9, weight_decay=args.weight_decay)
        # self.optimizer = torch.optim.Adam(
        #     net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        batch_size = self.args.batch_size
        while len(self.local_dataset) % batch_size == 1:
            batch_size += 1
        self.ldr_train = DataLoader(
            self.local_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        # train and update
        epoch_loss = []
        process_logits = LogitAdjust_our_add(args, cls_num_list=self.class_num_list)
        ce_criterion = set_loss(args, cls_num_list=self.class_num_list)
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for (_, images, labels) in self.ldr_train:
                images, labels = images.to(self.args.device), labels.cuda().to(self.args.device)

                features = net(images, outFeature)
                logits = net.forward_last_layer(features)
                logits = process_logits(logits)
                # logits = net(images)
                loss = ce_criterion(logits, labels, net, features, epoch)
                # loss = loss.mean()
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
                writer.add_scalar(
                    f'client{self.id}/loss_train', loss.item(), self.iter_num)
                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())

        return net.state_dict(), np.array(epoch_loss).mean()    
    
    def train_ours_second(self, args, rnd, net, writer):
    # def train_ours(self, args, net, writer):

        net.train()
        outFeature = True
        active_layers = net.get_trained_params_list(args)
        # set the optimizer
        self.optimizer = torch.optim.Adam(
            [{'params': active_layers, 'lr': self.second_lr}], lr=self.second_lr, betas=(0.9, 0.999), weight_decay=5e-4)
  
        # self.optimizer = torch.optim.SGD([{'params': active_layers, 'lr': self.lr}],
        #     lr=self.lr, momentum=0.9)
        
        batch_size = self.args.batch_size
        while len(self.local_dataset) % batch_size == 1:
            batch_size += 1
        self.ldr_train = DataLoader(
            self.local_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        class_probs = defaultdict(list)
        ### calculate DPT
        for (_, images, labels) in self.ldr_train:
            images, labels = images.to(self.args.device), labels.cuda().to(self.args.device)
            logits = net(images)
            probs = F.softmax(logits, dim=1)
            for i in range(labels.size(0)):
                true_label = labels[i].item()
                prob = probs[i, true_label].item()
                class_probs[true_label].append(prob)
                
        phase = (rnd - 10) / (args.rounds - 10)
        quantiles = 0.8 * cos(pi * phase)
        class_avg_prob = {}    
        for c in sorted(class_probs.keys()):
            probs = class_probs[c]
            probs_np = np.array(probs)
            max_val = np.max(probs_np)
            min_val = np.min(probs_np)
            if len(probs) == 0:
                continue  # 防止空列表出错
            
            q_val = min_val + quantiles * (max_val - min_val)
            class_avg_prob[c] = q_val
            
        # train and update
        epoch_loss = []
        process_logits = LogitAdjust_our(args, cls_num_list=self.class_num_list)
        ce_criterion = set_loss(args, cls_num_list=self.class_num_list)
        # ce_criterion = LogitAdjust(args, cls_num_list=self.class_num_list)
        for epoch in range(self.args.local_ep):
            batch_loss = []
                
            for (_, images, labels) in self.ldr_train:
                images, labels = images.to(self.args.device), labels.cuda().to(self.args.device)
                
                features = net(images, outFeature)
                logits = net.forward_last_layer(features)
                
                logits = process_logits(logits)
                # logits = net(images)
                loss = ce_criterion(logits, labels, net, features, epoch)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
                # writer.add_scalar(
                #     f'client{self.id}/loss_train', loss.item(), self.iter_num)
                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())
                
        return net.state_dict(), np.array(epoch_loss).mean(), class_avg_prob 
    
    ### used for noisy clients
    def train_ours_noise(self, args, w_kl, w_ce, w_neg, thresholds, local_net, global_net, writer):
        local_net.train()
        global_net.eval()
        
        outFeature = True
        active_layers = global_net.get_trained_params_list(args)
        
        ### get pseudo labels number per class
        criterion_nll = nn.NLLLoss()
        self.optimizer = torch.optim.Adam(
            [{'params': active_layers, 'lr': self.second_lr}], lr=self.second_lr, betas=(0.9, 0.999), weight_decay=5e-4)
        # self.optimizer = torch.optim.SGD([{'params': active_layers, 'lr': self.lr}],
        #     lr=self.lr, momentum=0.9)   
        
        batch_size = self.args.batch_size
        while len(self.local_dataset) % batch_size == 1:
            batch_size += 1
        self.ldr_train = DataLoader(
            self.local_dataset, batch_size=batch_size, shuffle=True, num_workers=4)        
        self.pseudo_class_num_list, correct_correction, total_correction = get_pseudo_class_num_list(self.ldr_train, global_net, 
                                                               self.class_num_list, thresholds, self.y_train, args)
        
        process_logits = LogitAdjust_our(args, cls_num_list=self.pseudo_class_num_list)
        # ce_criterion = set_loss(args, self.pseudo_class_num_list)
        
        epoch_loss = []    
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for (_, images, labels) in self.ldr_train:
                images, labels = images.to(self.args.device), labels.cuda().to(self.args.device)
                ### pseudo labels from global net
                with torch.no_grad():
                    global_logits = global_net(images)
                    global_prob = torch.softmax(global_logits.detach()/0.8, dim=-1)
                    global_probs = torch.softmax(global_logits.detach(), dim=-1)
                    glob_max_probs, _ = torch.max(global_probs, dim=-1)
                    _, global_labels = torch.max(global_prob, dim=-1)
                
                ### train local net
                local_features = local_net(images, outFeature)
                local_logits_ori = local_net.forward_last_layer(local_features)
                local_logits = process_logits(local_logits_ori)
                local_log_softmax = torch.log_softmax(local_logits, dim=1)
                log_pred = torch.where(torch.isinf(local_log_softmax), torch.full_like(local_log_softmax, 0), local_log_softmax)
                
                local_softmax = torch.softmax(local_logits, dim=1)
                local_max_probs, _ = torch.max(local_softmax, dim=-1)
                
                ### pseudo label ce
                thresholds_tensor = torch.tensor(thresholds, device=images.device)
                percent_exceed = (glob_max_probs.unsqueeze(1) - thresholds_tensor) / thresholds_tensor
                mask = percent_exceed > 0 
                diff_masked = torch.where(mask, percent_exceed, torch.full_like(percent_exceed, -float('inf')))
                pseudo_labels = torch.argmax(diff_masked, dim=-1)
                has_valid_pseudo = mask.any(dim=-1)
                ce_labels = torch.zeros_like(labels)
                ce_labels = torch.where(has_valid_pseudo, 
                       pseudo_labels, 
                       labels)

                local_ce_loss = F.nll_loss(log_pred, ce_labels)
                
                ### local and global kl
                kl_loss = F.kl_div(log_pred, global_prob, reduction="batchmean")
                
                ### noisy label negative ce
                neg_selected_labels = []
                neg_selected_logits = []
                
                neg_thresholds = torch.tensor(thresholds).to(local_max_probs.device)
                # 获取每个样本对应的类别阈值
                sample_thresholds = neg_thresholds[labels]  # 形状变为 [batch_size]

                for i in range(images.size(0)):
                    # 比较当前样本的max_prob与其对应类别的阈值
                    if glob_max_probs[i] > sample_thresholds[i]:
                        neg_selected_labels.append(global_labels[i].item())
                        neg_selected_logits.append(local_logits[i])
                        
                if not neg_selected_labels or not neg_selected_logits:
                    neg_loss = torch.tensor(0.0, device=args.device)
                else:
                    neg_selected_labels = torch.tensor(neg_selected_labels, dtype=torch.long) if neg_selected_labels else torch.empty(0, dtype=torch.long)
                    neg_selected_logits = torch.stack(neg_selected_logits)
                    
                    pseudo_class_num_tensor = torch.tensor(self.pseudo_class_num_list, dtype=torch.long, device=neg_selected_labels.device)
                    nonzero_classes = torch.nonzero(pseudo_class_num_tensor).squeeze(-1) 
                    random_tensor = torch.zeros_like(neg_selected_labels)  # 初始化随机偏移量
                    for i in range(len(neg_selected_labels)):
                        original_class = neg_selected_labels[i].item()
                        
                        # 获取当前类以外的其他存在类别
                        other_classes = nonzero_classes[nonzero_classes != original_class]
                        
                        # 从其他存在的类中随机选一个
                        random_class = other_classes[torch.randint(0, len(other_classes), (1,))]
                        random_tensor[i] = random_class - original_class 
                    random_tensor = torch.randint(1, args.n_classes, (neg_selected_labels.shape[0],), dtype=torch.long)
                    labels_neg = (neg_selected_labels + random_tensor) % args.n_classes
                    assert labels_neg.max() <= args.n_classes-1 
                    assert labels_neg.min() >= 0
                    assert (labels_neg != neg_selected_labels).sum() == len(neg_selected_labels)
                    neg_selected_logits = torch.log(torch.clamp(1.-F.softmax(neg_selected_logits, -1), 
                                                                min=1e-5, max=1.)).to(args.device)
                    labels_neg = labels_neg.to(args.device)
                    neg_loss = F.nll_loss(neg_selected_logits, labels_neg)
                
                local_loss = w_ce * local_ce_loss + w_kl * kl_loss + w_neg * neg_loss
                self.optimizer.zero_grad()
                local_loss.backward()
                self.optimizer.step()
                
                batch_loss.append(local_loss.item())
                # writer.add_scalar(
                #     f'client{self.id}/loss_train', local_loss.item(), self.iter_num)
                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())
            
        return local_net.state_dict(), np.array(epoch_loss).mean(), correct_correction, total_correction
    
    def train_FedNoRo_ours(self, args, student_net, teacher_net, writer, weight_kd):
        student_net.train()
        teacher_net.eval()
        # set the optimizer
        outFeature = True
        active_layers = student_net.get_trained_params_list(args)
        # self.optimizer = torch.optim.SGD([{'params': active_layers, 'lr': self.lr}],
        #     lr=self.lr, momentum=0.9)
        # self.optimizer = torch.optim.SGD(student_net.parameters(),
        #     lr=self.lr, momentum=0.9)
        # self.optimizer = torch.optim.Adam(
        #     student_net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        self.optimizer = torch.optim.Adam(
            [{'params': active_layers, 'lr': self.second_lr}], lr=self.second_lr, betas=(0.9, 0.999), weight_decay=5e-4)
        
        batch_size = self.args.batch_size
        while len(self.local_dataset) % batch_size == 1:
            batch_size += 1
        self.ldr_train = DataLoader(
            self.local_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        # train and update
        epoch_loss = []
        criterion = LA_KD(args, cls_num_list=self.class_num_list)
        
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for (img_idx, images, labels) in self.ldr_train:
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                logits = student_net(images)
                with torch.no_grad():
                    teacher_output = teacher_net(images)
                    soft_label = torch.softmax(teacher_output/0.8, dim=1)

                loss = criterion(logits, labels, soft_label, weight_kd)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
                writer.add_scalar(
                    f'client{self.id}/loss_train', loss.item(), self.iter_num)
                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())

        return student_net.state_dict(), np.array(epoch_loss).mean()
    
    
