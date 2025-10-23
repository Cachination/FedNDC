import copy
import torch
import numpy as np
from torch.nn.functional import softmax, log_softmax
from torch import stack, no_grad
from .utils import model_kl_div

def FedAvg(w, dict_len):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():        
        w_avg[k] = w_avg[k] * dict_len[0] 
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * dict_len[i]
        w_avg[k] = w_avg[k] / sum(dict_len)
    return w_avg


def DaAgg(w, dict_len, clean_clients, noisy_clients):
    client_weight = np.array(dict_len)
    client_weight = client_weight / client_weight.sum()
    distance = np.zeros(len(dict_len))
    for n_idx in noisy_clients:
        dis = []
        for c_idx in clean_clients:
            dis.append(model_dist(w[n_idx], w[c_idx]))
        distance[n_idx] = min(dis)
    distance = distance / distance.max()
    client_weight = client_weight * np.exp(-distance)
    client_weight = client_weight / client_weight.sum()
    # print(client_weight)

    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * client_weight[0] 
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * client_weight[i]
    return w_avg

def KLAgg(args, w, dict_len, clean_clients, noisy_clients, dataset_auxiliary, net):
    client_weight = np.array(dict_len)
    client_weight = client_weight / client_weight.sum()
    distance = np.zeros(len(dict_len))
    
    random_state = np.random.RandomState(args.seed)
    user_id = list(range(args.n_clients))
    total_indices_unlabeled = [i for i in range(len(dataset_auxiliary))]
    batch_indices_unlabeled = random_state.choice(total_indices_unlabeled,
                                                        args.batch_size_auxiliary,
                                                        replace=False)
    images_unlabeled = []
    for idx in batch_indices_unlabeled:
        image, _ = dataset_auxiliary[idx]  
        images_unlabeled.append(image)
    images_unlabeled = stack(images_unlabeled, dim=0)  
    images_unlabeled = images_unlabeled.to(args.device)

    for n_idx in noisy_clients:
        kl_list = []
        for c_idx in clean_clients:
            # 这里假设 model_kl_div 是您定义的函数，计算两个模型（噪声客户端 n_idx 与干净客户端 c_idx）
            # 在某个固定数据集上的 KL 散度（越小表示预测分布越接近）
            kl_value = model_kl_div(args, w[n_idx], w[c_idx], net, images_unlabeled)
            kl_list.append(kl_value)
        distance[n_idx] = min(kl_list)
        
    if distance.max() > 0:
        distance = distance / distance.max()
        
    # 使用指数函数调制权重：KL 值越大（噪声越严重），权重越低
    client_weight = client_weight * np.exp(-distance)
    client_weight = client_weight / client_weight.sum()
    
    # 聚合所有客户端模型参数
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * client_weight[0]
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * client_weight[i]
    return w_avg


def model_dist(w_1, w_2):
    assert w_1.keys() == w_2.keys(), "Error: cannot compute distance between dict with different keys"
    dist_total = torch.zeros(1).float()
    for key in w_1.keys():
        if "int" in str(w_1[key].dtype):
            continue
        dist = torch.norm(w_1[key] - w_2[key])
        dist_total += dist.cpu()

    return dist_total.cpu().item()