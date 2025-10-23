import copy
import logging
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import stack, no_grad
from torch.nn.functional import softmax, log_softmax

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from utils.utils import get_output_and_loss

def client_selection_fednoro(args, netglob, dict_users, dataset_train, gamma_s):
    criterion = nn.CrossEntropyLoss(reduction='none')

    user_id = list(range(args.n_clients))
    loader = DataLoader(dataset=dataset_train, batch_size=32,
                    shuffle=False, num_workers=4)
    out_feature = False
    local_output, loss = get_output_and_loss(
        loader, netglob.to(args.device), args, False, criterion, out_feature)
    metrics = np.zeros((args.n_clients, args.n_classes)).astype("float")
    num = np.zeros((args.n_clients, args.n_classes)).astype("float")
    for id in range(args.n_clients):
        idxs = dict_users[id]
        for idx in idxs:
            c = dataset_train.targets[idx]
            num[id, c] += 1
            metrics[id, c] += loss[idx]
    metrics = metrics / num
    # metrics = np.where(num != 0, metrics / num, 0)
    # metrics = np.divide(metrics, num, out=np.zeros_like(metrics), where=num!=0)
    for i in range(metrics.shape[0]):
        for j in range(metrics.shape[1]):
            if np.isnan(metrics[i, j]):
                metrics[i, j] = np.nanmin(metrics[:, j])
    for j in range(metrics.shape[1]):
        metrics[:, j] = (metrics[:, j]-metrics[:, j].min()) / \
            (metrics[:, j].max()-metrics[:, j].min())
    # logging.info("metrics:")
    # logging.info(metrics)

    vote = []
    for i in range(9):
        gmm = GaussianMixture(n_components=2, random_state=i).fit(metrics)
        gmm_pred = gmm.predict(metrics)
        noisy_clients = np.where(gmm_pred == np.argmax(gmm.means_.sum(1)))[0]
        noisy_clients = set(list(noisy_clients))
        vote.append(noisy_clients)
    cnt = []
    for i in vote:
        cnt.append(vote.count(i))
    noisy_clients = list(vote[cnt.index(max(cnt))])
    clean_clients = list(set(user_id) - set(noisy_clients))
        
    return clean_clients, noisy_clients
    
    
    
def client_selection_ours(args, dataset_auxiliary, w_locals_last, net):
    random_state = np.random.RandomState(args.seed)
    user_id = list(range(args.n_clients))
    total_indices_unlabeled = [i for i in range(len(dataset_auxiliary))]
    batch_indices_unlabeled = random_state.choice(total_indices_unlabeled,
                                                        args.batch_size_auxiliary,
                                                        replace=False)
    images_unlabeled = []
    for idx in batch_indices_unlabeled:
        ### no need labels
        image, _ = dataset_auxiliary[idx]  
        images_unlabeled.append(image)
    images_unlabeled = stack(images_unlabeled, dim=0)  
    images_unlabeled = images_unlabeled.to(args.device)

    list_var = {}
    clean_clients = []
    noisy_clients = []
    list_softmax = []
    ii = 0
    
    for dict_local in w_locals_last:
        net.load_state_dict(dict_local)

        uncen = torch.tensor([]).cuda()
        pred_prob = torch.tensor([]).cuda()
        for step in range(15):
            with no_grad():
                local_drop = net(images_unlabeled, outDropout=True)
                drop_softmax = softmax(local_drop, dim=1).unsqueeze(-1)
                pred_prob = torch.cat([pred_prob, drop_softmax], dim=-1)
        list_softmax.append(pred_prob.reshape(-1).cpu().detach().numpy())
    
        uncen = pred_prob.mean(-1)
        per_uncen = -torch.sum(torch.log(uncen) * uncen, dim=-1)
        list_var[user_id[ii]] = torch.sum(per_uncen)
        ii = ii + 1
    
    list_var = sorted(list_var.items(), key=lambda x:x[1], reverse=False)
    
    with open("value.txt", "w") as f:
        f.write("客户端ID\t不确定性值（原始）\t归一化值\n")  # 表头
        total = sum(v for _, v in list_var)
        for client_id, value in list_var:
            normalized = value / total  # 计算归一化值
            # 写入: ID, 原始值, 归一化值（保留4位小数）
            f.write(f"{client_id}\t{value.item():.4f}\t{normalized.item():.4f}\n")

    X = np.array([[value.item()] for _, value in list_var])  # shape: [n_clients, 1]
    
    # 使用K-means聚类为2类
    kmeans = KMeans(n_clusters=2, random_state=args.seed).fit(X)
    
    # 确定哪一类是"低不确定性"类（比较类中心）
    cluster_centers = kmeans.cluster_centers_
    if cluster_centers[0][0] < cluster_centers[1][0]:
        clean_label = 0  # 第一个类中心更小
    else:
        clean_label = 1  # 第二个类中心更小
    
    # 根据聚类结果分类客户端
    clean_clients = []
    y_clients = []
    for (client_id, _), label in zip(list_var, kmeans.labels_):
        if label == clean_label:
            clean_clients.append(client_id)
        else:
            y_clients.append(client_id)
    noisy_clients = list(set(user_id) - set(clean_clients))              
    # sum_value = 0
    # for key, value in list_var:
    #     sum_value = sum_value + value

    # for key, value in list_var:
    #     value = value / sum_value
    #     if value <= 0.05:
    #         clean_clients.append(key)
    #     else:
    #         noisy_clients.append(key)
    # client_keys = np.array(list(list_var.keys()))
    # uncertainty_values = torch.stack(list(list_var.values())).cpu().numpy().reshape(-1, 1)

    # ### use k-means classify clients
    # kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    # cluster_labels = kmeans.fit_predict(uncertainty_values)
    # clean_clients = client_keys[cluster_labels == cluster_labels.min()].tolist()
    # noisy_clients = client_keys[cluster_labels == cluster_labels.max()].tolist()
    
    return clean_clients, noisy_clients


