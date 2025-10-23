import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from datetime import datetime
from fedned.server import Server
from fedned.dataset.get_cifar10 import get_dataset
from fedned.dataset.utils.dataset import Indices2Dataset
from fedned.models.model_feature import ResNet_cifar_feature
from fedned.dataset.utils.noisify import noisify_label
from fedned.utils.tools import get_set_gpus
from fedned.options import args_parser
from fedned.models.build_model import build_model
from utils.utils import add_noise
import numpy as np
import logging
import copy
import torch
import random
import pandas as pd

def get_train_label(data_local_training, index_list):
    trian_label_list = []
    for index in index_list:
        label = data_local_training[index][1]
        trian_label_list.append(label)
    return trian_label_list


def label_rate(test_label_list, train_label_list):
    true_num = 0
    for true_label, nos_label in zip(test_label_list, train_label_list):
        if true_label == nos_label:
            true_num += 1
    rate = true_num / len(test_label_list)
    print(rate)

def set_seed(seed):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn

def main(args):
    prev_time = datetime.now()
    
    args.use_etf = False
    args.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    data_local_training, data_global_test, list_client2indices, global_distill_dataset, dict_users= get_dataset(args)


    # model = ResNet_cifar_feature(resnet_size=8, scaling=4,
    #                              save_activations=False, group_norm_num_groups=None,
    #                              freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)
    model = build_model(args)
    
    ###add noise
    train_data_list = []
    label_list = []
    y_train = np.array(data_local_training.targets)
    y_train_noisy, gamma_s, real_noise_level = add_noise(
        args, y_train, dict_users)
    noisy_num = np.sum(y_train != y_train_noisy)
    
    for i in range(args.num_clients):
        current_client_index_list = list_client2indices[i]
        client_noisy_labels = y_train_noisy[current_client_index_list].tolist()    
        # 构建客户端数据集（假设Indices2Dataset支持标签替换）
        indices2data = Indices2Dataset(data_local_training)
        data_client = indices2data
        data_client.load(current_client_index_list, client_noisy_labels)
        # 保存数据集和标签
        train_data_list.append(data_client)
        label_list.append(client_noisy_labels)
        
    # for i in range(args.num_clients):
    #     current_client_index_list = list_client2indices[i]
    #     train_label_list = get_train_label(data_local_training, current_client_index_list)
    #     num_classes = train_label_list
    #     test_label_list = copy.deepcopy(train_label_list)
    #     noise_index = int(len(list_client2indices[i]) * args.noise_rate_list[i])
    #     for idx, true_label in enumerate(train_label_list[:noise_index]):
    #         noisy_label = noisify_label(true_label, num_classes, noise_type=args.noise_type)
    #         train_label_list[idx] = noisy_label

    #     label_rate(test_label_list, train_label_list)
    #     indices2data = Indices2Dataset(data_local_training)
    #     data_client = indices2data
    #     data_client.load(list_client2indices[i], train_label_list)
    #     train_data_list.append(data_client)
    #     label_list.append(train_label_list)


    for client, indices in enumerate(label_list):
        nums_data = [0 for _ in range(args.num_classes)]
        for idx in indices:
            # label = data_local_training[idx][1]
            nums_data[idx] += 1
        print(f'{client}: {nums_data}')

    server = Server(args=args,
                    train_data_list=train_data_list,
                    global_test_dataset=data_global_test,
                    global_distill_dataset=global_distill_dataset,
                    global_student=model,
                    temperature=args.temperature,
                    mini_batch_size_distillation=args.mini_batch_size_distillation,
                    lamda=args.lamda,
                    y_train = y_train,
                    noisy_num = noisy_num
                    )

    server.train()
    
    correct_acc = server.correct_acc
    correct_per = server.correct_per
    acc = server.test_acc
    acc.sort()
    acc = acc[90:]
    # print('train finished---> final_acc={}'.format(sum(acc) / len(acc)))
    
    df = pd.DataFrame({
        'Round': range(11, len(correct_acc) + 11),  # 轮次从 1 开始
        'Percent': [f"{x:.4f}%" for x in correct_per],
        'Accuracy': [f"{x:.4f}%" for x in correct_acc]
    })
    
    df.to_csv('fedned_relabel_all.csv', index=False)  # 保存为 CSV
    # 保存为 Excel 文件（需要安装 openpyxl: pip install openpyxl）
    df.to_excel('fedned_relabel_all.xlsx', index=False)  # 保存为 Excel
    
    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = "train time %02d:%02d:%02d" % (h, m, s)
    print(time_str)


def fedned(args):
    set_seed(0)
    args = args_parser()
    logging.info(str(args))

    main(args)

