import os
import numpy as np
import argparse
import yaml
import shutil
from pathlib import Path
import copy
import torch
import logging
import time
import csv

# torch.backends.cudnn.benchmark = True
from torch.utils.tensorboard import SummaryWriter

from balancefl.utils.misc import update_config, deterministic
from balancefl.utils.logger import Logger, print_write, write_summary
from balancefl.data import dataloader
from balancefl.models.utils import *
from balancefl.fed import Fed_server_ISIC, Fed_server, train_lt, train_lt_speech
from utils.utils import add_noise

"""
Script for original federated training (train only a global model). 
Support different datasets/losses/samplings. 
"""


def balancefl(args):

    data_root_dict = {
        "cifar10": "./data",
    }

    """
    Parameters
    """
    # important parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="./balancefl/config/fedlt.yaml", type=str)
    parser.add_argument("--exp_name", default="balancefl", type=str, help="exp name")
    parser.add_argument(
        "--non_iidness", default=1, type=int, help="non-iid degree of distributed data"
    )
    parser.add_argument("--tao_ratio", type=float, default=2, choices=[1, 2, 4])
    parser.add_argument("--lep", type=int, default=5, choices=[1, 2, 5])
    parser.add_argument("--frac", type=float, default=1, choices=[0.2, 0.6, 1])
    # optional params
    parser.add_argument("--seed", default=0, type=int, help="using fixed random seed")
    parser.add_argument("--work_dir", default="./runs_exp", type=str, help="output dir")
    # unused params
    parser.add_argument("--test", default=False, action="store_true")
    args = parser.parse_args()

    # random seed
    if args.seed is not None:
        deterministic(args.seed)

    # config
    log_dir = f"{args.work_dir}/{args.exp_name}"    #_{args.tao_ratio}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    with open(args.cfg) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = update_config(config, args, log_dir)  # let args overwite YAML config

    logging.info(str(config))

    # logger
    logger = Logger(log_dir)
    logger.log_cfg(config)  # save cfg
    log_file = f"{log_dir}/log.txt"
    if os.path.exists(log_file):
        os.remove(log_file)

    # tensorboard
    tensorboard = SummaryWriter(log_dir=f"{log_dir}/tensorboard")

    # prepare dataset
    dataset = config["dataset"]["name"]
    data_root = data_root_dict[dataset]
    if dataset in ["cifar100", "cifar10"]:
        if config["dataset"]["shot_few"] > 0:
            (
                per_client_data,
                per_client_label,
                test_data,
                test_label,
                cls_per_client,
            ) = dataloader.CIFAR_FL_mixed(data_root, config)
        else:
            (
                per_client_data,
                per_client_label,
                test_data,
                test_label,
                cls_per_client,
                num_per_cls_per_client,
                train_num_per_cls,
            ) = dataloader.CIFAR_FL(data_root, config)
            
        print(type(num_per_cls_per_client))        
        training_num_per_cls = np.array([len(i) for i in per_client_label])
        print_write([cls_per_client], log_file)
        print_write([train_num_per_cls], log_file)
        print_write([num_per_cls_per_client], log_file)
        print(len(per_client_label[0]))
    elif dataset in ["ISIC"]:
        (
            client_datasets,
            train_dataset,
            test_dataset,
            cls_per_client,
            num_per_cls_per_client,
            train_num_per_cls,
            idx_per_client,
        ) = dataloader.ISIC_FL(data_root, config)
    else:
        (
            client_datasets,
            train_dataset,
            test_dataset,
            cls_per_client,
            num_per_cls_per_client,
            train_num_per_cls,
            idx_per_client,
        ) = dataloader.speech_FL(data_root, config)
    

        print(type(num_per_cls_per_client))        
        training_num_per_cls = np.sum(num_per_cls_per_client, axis=1)
        print_write([cls_per_client], log_file)
        print_write([train_num_per_cls], log_file)
        print_write([num_per_cls_per_client], log_file)


    
    # y_train = np.array(dataset_train.targets)
    # y_train_noisy, gamma_s, real_noise_level = add_noise(
    #     args, y_train, dict_users)
    # dataset_train.targets = y_train_noisy    

    """"
    FL config setup 
    """
    # assign devices
    num_client = config["fl_opt"]["num_clients"]
    num_cls = config["dataset"]["num_classes"]

    # gpu_list = [0,1,2,3][: torch.cuda.device_count()]
    # gpu_idx = [gpu_list[i % len(gpu_list)] for i in range(num_client)]
    # config.update(
    #     {"device": torch.device("cpu" if torch.cuda.is_available() else "cpu")}
    # )  # init_network
    # config.update(
    #     {
    #         "device_client": [
    #             torch.device(f"cuda:{i}" if torch.cuda.is_available() else "cpu")
    #             for i in gpu_idx
    #         ]
    #     }
    # )  # client device\
    # print(f"gpu of clients: {gpu_idx}")

    # init model and criterion on cpu
    network = init_models(config)  # print(network["feat_model"])
    criterion = init_criterion(config)

    # multi-process setup
    import multiprocessing as mp


    if dataset in ["cifar100", "cifar10"]:
    # FL class for servrer
        fed = Fed_server(
            network,
            criterion,
            config,
            per_client_data,
            per_client_label,
            training_num_per_cls,
            test_data,
            test_label,
        )
    
    else:
        fed = Fed_server_ISIC(
            network,
            criterion,
            config,
            train_dataset,
            training_num_per_cls,
            test_dataset,
        )
        
    aggre_mode = config["fl_opt"]["aggregation"]
    frac = config["fl_opt"]["frac"]
    client_per_round = max(int(frac * fed.num_clients), 1)
    best_performance = 0.
    """"
    FL starts 
    """
    # training mode
    if not args.test:
        best_acc = 0
        start = time.perf_counter()
        # FL rounds
        for round_i in range(config["fl_opt"]["rounds"]):
            # select users
            selected_idx = np.random.choice(
                range(fed.num_clients), client_per_round, replace=False
            )
            
            # select classes and num_of_samples per class
            selected_cls = []
            for i in selected_idx:
                selected_cls += list(cls_per_client[i])
            print_write(
                [f"\n Round: {round_i}, selected clients: {selected_idx}"], log_file
            )
            # print_write([f'selected cls: {set(selected_cls)}'], log_file)
            
            client_models = []
            for i in selected_idx.tolist():
                # start = time.perf_counter()
            # for i in range(0,3):
                if dataset in ["cifar100", "cifar10"]:
                    permode = train_lt(copy.deepcopy(fed.server_network), i, per_client_data, per_client_label, config,
                            num_per_cls_per_client)
                else:
                    permode = train_lt_speech(copy.deepcopy(fed.server_network), i, client_datasets, config,
                            num_per_cls_per_client, train_dataset, idx_per_client)
                client_models.append(permode)
                # end = time.perf_counter()
                # runtime = end - start
            fed.aggregation(selected_idx, aggre_mode, client_models)
            end = time.perf_counter()
            run_time = end - start
            # evaluate
            if round_i > -1:
                (
                    acc,
                    bacc,
                    class_accuracies,
                ) = fed.evaluate_global()
                # train_loss = train_loss_per_cls.mean()
                # train_acc = train_acc_per_cls.mean()
                
                logging.info(
                    "******** round: %d, acc: %.4f, bacc: %.4f ********" % (round_i, acc, bacc))
                if bacc > best_performance:
                    best_performance = bacc
                    # with open(f'balancefl_class_acc_round_{round_i}.csv', 'w', newline='') as f:
                    #     writer = csv.writer(f)
                    #     writer.writerow(['Class', 'Accuracy'])  # 表头
                    #     for cls, acc in sorted(class_accuracies.items()):
                    #         writer.writerow([cls, acc])
                # testset (e.g., Cifar-100) is equally distributed among classes
                # test_loss_mean = test_loss_per_cls.mean()
                # test_acc_mean = test_acc_per_cls.mean()

                # logging
                kd_loss = np.array(fed.losses_kd)[selected_idx].mean()
                cls_loss = np.array(fed.losses_cls)[selected_idx].mean()
                np.set_printoptions(precision=3)
                print_write(
                    [
                        "cls_loss, kd_loss, train_loss, test_loss, train_acc, test_acc: ",
                        [
                            cls_loss,
                            kd_loss,
                            acc,
                            bacc,
                        ],
                    ],
                    log_file,
                )
                # print_write(["per_cls_acc (train): ", train_acc_per_cls], log_file)
                # print_write(["per_cls_acc: ", test_acc_per_cls], log_file)

                # write_summary(
                #     tensorboard,
                #     split="train",
                #     step=round_i,
                #     kd_loss=kd_loss,
                #     cls_loss=cls_loss,
                #     loss=train_loss,
                #     acc=train_acc,
                #     cls0_acc=train_acc_per_cls[0],
                #     cls2_acc=train_acc_per_cls[1],
                #     cls3_acc=train_acc_per_cls[2],
                #     cls4_acc=train_acc_per_cls[3],
                #     cls5_acc=train_acc_per_cls[4],
                #     cls6_acc=train_acc_per_cls[5],
                # )
                write_summary(
                    tensorboard,
                    split="val",
                    step=round_i,
                    # loss=test_loss_mean,
                    # acc=test_acc_mean,
                    # cls0_acc=test_acc_per_cls[0],
                    # cls2_acc=test_acc_per_cls[1],
                    # cls3_acc=test_acc_per_cls[2],
                    # cls4_acc=test_acc_per_cls[3],
                    # cls5_acc=test_acc_per_cls[4],
                    # cls6_acc=test_acc_per_cls[5],
                    acc=acc,
                    bacc=bacc,
                )
                # torch.save(
                #     {
                #         "round_i": round_i,
                #         "server_network": fed.server_network,
                #         "client_network": fed.networks,
                #         "train_acc_per_cls": train_acc_per_cls,
                #         "test_acc_per_cls": test_acc_per_cls,
                #     },
                #     f"{log_dir}/{round_i}.pth",
                # )

                # save ckpts
                if bacc > best_acc:
                    ckpt = {"round_i": round_i, "model": fed.server_network}
                    ckpt_name = f"{log_dir}/best.pth"
                    torch.save(ckpt, ckpt_name)
                    best_acc = bacc
                    print_write(
                        [f"best round: {round_i}, accuracy: {bacc*100}"], log_file
                    )
                # del ckpt
            else:
                logging.info(
                    "******** round: %d ********" % (round_i))
    # Currently do not support test mode. Use `evaluate.py` instead.
    else:
        pass

    print("Training_end")
