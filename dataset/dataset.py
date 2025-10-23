import numpy as np
import torch
import os
import torchvision.transforms as transforms
from torch.utils.data import Subset, Dataset
from torch.utils.data.sampler import WeightedRandomSampler

from .all_datasets import cifar10, cifar100, ISIC
from .all_datasets import imbalanced_cifar10, imbalanced_cifar100, imbalanced_speech

from utils.sampling import iid_sampling, non_iid_dirichlet_sampling


def get_speech(args):
    root = "./data"
    args.n_classes = 35
    data_file_path = f"./data/training_data.pt"
    if os.path.exists(data_file_path) is False:
        train_dataset = imbalanced_speech("training", imbalance_ratio=args.imb_ratio, root=root, reverse=None) # imb_ratio only for training set
        test_dataset = imbalanced_speech("testing", imbalance_ratio=None, root=root, reverse=None)
    else:
        # train_dataset = ImbalanceSpeech_clean("training", imbalance_ratio=imb_ratio, root=root, reverse=None)   # imb_ratio only for training set
        # val_dataset = ImbalanceSpeech_clean("validation", imbalance_ratio=None, root=root, reverse=None)
        # test_dataset = ImbalanceSpeech_clean("testing", imbalance_ratio=None, root=root, reverse=None)

        # training
        train_data_all, train_label_all, train_num_per_cls = \
            ImbalanceSpeech_clean("training", args.imb_ratio, root)

        # testing
        test_data, test_label, test_num_per_cls = \
            ImbalanceSpeech_clean("testing", None, root)
    
    n_train = len(train_data_all)
    y_train = np.array(train_label_all)
    assert n_train == len(y_train)

    if args.iid:
        dict_users = iid_sampling(n_train, args.n_clients, args.seed)
    else:
        dict_users = non_iid_dirichlet_sampling(y_train, args.n_classes, args.non_iid_prob_class, args.n_clients, seed=100, alpha_dirichlet=args.alpha_dirichlet)

    # check
    assert len(dict_users.keys()) == args.n_clients
    items = []
    for key in dict_users.keys():
        items += list(dict_users[key])
    assert len(items) == len(set(items)) == len(y_train)

    print("### Datasets are ready ###")
    
    train_dataset = local_client_dataset(train_data_all, train_label_all, args)
    test_dataset = gen_test_dataset(test_data, test_label, args)
    
    ### get auxiliary dataset (subsequent labels are not required)
    targets = np.array(test_label)
    classes = np.unique(targets)
    auxiliary_indices = []
    np.random.seed(args.seed)  
    for cls in classes:
        cls_indices = np.where(targets == cls)[0]
        print(cls_indices.shape)
        selected = np.random.choice(cls_indices, 80, replace=False)
        auxiliary_indices.extend(selected.tolist())
    np.random.shuffle(auxiliary_indices)
    auxiliary_dataset = Subset(test_dataset, auxiliary_indices)
        
    if "ours" in args.exp :
        return train_dataset, test_dataset, auxiliary_dataset, dict_users
    else:
        return train_dataset, test_dataset, dict_users
    
    
def get_dataset(args):
    if args.dataset == "ISIC":
        root = "./data"
        args.n_classes = 8

        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = ISIC(root, "train", train_transform)
        test_dataset = ISIC(root, "test", val_transform) 
            
    elif args.dataset == "cifar10" :
        root = "./data"
        args.n_classes = 10

        normalize = transforms.Normalize([0.4914, 0.4822, 0.4465],
                                            [0.2023, 0.1994, 0.2010])
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ])
        if args.imb_ratio == 1:
            train_dataset = cifar10(root, "train", train_transform)
        else:
            train_dataset = imbalanced_cifar10(root, "train", train_transform, args.imb_ratio)
        test_dataset = cifar10(root, "test", val_transform)


    elif args.dataset == "cifar100":
        root = "./data"
        args.n_classes = 100

        normalize = transforms.Normalize([0.4914, 0.4822, 0.4465],
                                            [0.2023, 0.1994, 0.2010])
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ])
        if args.imb_ratio == 1:
            train_dataset = cifar100(root, "train", train_transform)
        else:
            train_dataset = imbalanced_cifar100(root, "train", train_transform, args.imb_ratio)
        test_dataset = cifar100(root, "test", val_transform)
    else:
        exit("Error: unrecognized dataset")

    ### get auxiliary dataset (subsequent labels are not required)
    targets = np.array(test_dataset.targets)
    classes = np.unique(targets)
    auxiliary_indices = []
    np.random.seed(args.seed)  
    for cls in classes:
        cls_indices = np.where(targets == cls)[0]
        selected = np.random.choice(cls_indices, 5*args.batch_size, replace=False)
        auxiliary_indices.extend(selected.tolist())
    np.random.shuffle(auxiliary_indices)
    auxiliary_dataset = Subset(test_dataset, auxiliary_indices)
        
        
    n_train = len(train_dataset)
    y_train = np.array(train_dataset.targets)
    assert n_train == len(y_train)

    if args.iid:
        dict_users = iid_sampling(n_train, args.n_clients, args.seed)
    else:
        dict_users = non_iid_dirichlet_sampling(y_train, args.n_classes, args.non_iid_prob_class, args.n_clients, seed=100, alpha_dirichlet=args.alpha_dirichlet)

    # check
    assert len(dict_users.keys()) == args.n_clients
    items = []
    for key in dict_users.keys():
        items += list(dict_users[key])
    assert len(items) == len(set(items)) == len(y_train)

    print("### Datasets are ready ###")
    if "ours" in args.exp :
        return train_dataset, test_dataset, auxiliary_dataset, dict_users
    else:
        return train_dataset, test_dataset, dict_users


def ImbalanceSpeech_clean(mode, imbalance_ratio, root, imb_type='exp', test_imb_ratio=None, reverse=False):
    """
    mode: training, testing, validation  
    imbalance_ratio: only for training
    root: data folder  

    Important paramters
    ---
    Return: (all are lists)
    data: [a, b, ...], a and b are tensor of 1*32*32
    labels: [a, b, ...], a and b are scalar
    """
    
    cls_num = 35
    data_file_path = f"./data/{mode}_data.pt"
    target_file_path = f"./data/{mode}_target.pt"

    if os.path.exists(data_file_path) is False:
        raise RuntimeError

    data = torch.load(data_file_path)
    targets = torch.load(target_file_path)

    # obtain the distribution
    num_per_cls = [0 for i in range(cls_num)]
    for label in targets:
        num_per_cls[label] += 1
    print(mode, "original num_per_cls:", num_per_cls)

    # change the distribution to long-tail
    if mode == "training":
        selected_num_per_cls = get_num_per_cls(num_per_cls, imb_type, imbalance_ratio, reverse=reverse)
        data, targets = gen_imbalanced_data(data, targets, selected_num_per_cls)
        print("selected num_per_cls:", mode, selected_num_per_cls)
    else:
        selected_num_per_cls = num_per_cls

    # adjust the data according to max/min/mean ~ (-68, 51, -18)
    min, max, mean = torch.min(data), torch.max(data), torch.mean(data)
    print("min/max/mean:", min, max, mean)
    data = (data + 9)/61
    print("data/target shape:", data.shape, targets.shape, "\n")

    # change data and targets into list
    data_list, targets_list = [], []
    for i, j in zip(data, targets):
        data_list.append(i)
        targets_list.append(int(j))

    return data_list, targets_list, selected_num_per_cls
    

def gen_imbalanced_data(data, targets, selected_num_per_cls):
    """
    self.data: numpy.array
    self.targets: list
    """
    new_data = []
    new_targets = []
    targets_np = np.array(targets, dtype=np.int64)
    classes = np.unique(targets_np)

    num_per_cls_dict = dict()
    for the_class, the_img_num in zip(classes, selected_num_per_cls):
        num_per_cls_dict[the_class] = the_img_num
        idx = np.where(targets_np == the_class)[0]
        # print(the_class, the_img_num, idx)
        np.random.shuffle(idx)
        selec_idx = idx[:the_img_num]
        new_data.append(data[selec_idx, ...])
        new_targets.extend([the_class, ] * the_img_num)
    data = torch.cat(new_data, 0)        # n*32*32*3
    targets = torch.tensor(new_targets)   # list of length n
    return data, targets


def get_num_per_cls(num_per_cls, imb_type, imb_factor, reverse=False):
        cls_num = len(num_per_cls)
        img_max = 3000      # manually defined
        img_min = min(num_per_cls)

        selected_num_per_cls = []
        if imb_factor == 1:
            selected_num_per_cls = [img_min for i in range(cls_num)]

        elif imb_factor <= 0.1:
            if imb_type == 'exp':
                for cls_idx in range(cls_num):
                    if reverse:
                        num = img_max * (imb_factor**((cls_num - 1 - cls_idx) / (cls_num - 1.0)))
                    else:
                        num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                    selected_num_per_cls.append(int(num))
            elif imb_type == 'step':
                for cls_idx in range(cls_num // 2):
                    selected_num_per_cls.append(int(img_max))
                for cls_idx in range(cls_num // 2):
                    selected_num_per_cls.append(int(img_max * imb_factor))
            else:
                selected_num_per_cls.extend([int(img_max)] * cls_num)
        return selected_num_per_cls

class local_client_dataset(Dataset):
    def __init__(self, per_client_data, per_client_label, args, aug=False):
        self.data = per_client_data
        self.targets = per_client_label
        self.dataset_name = args.dataset
        self.aug = aug

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):        
        data = self.data[index]
        # plt.imshow(data); plt.savefig("./a.jpg")

        if self.dataset_name == "CUB":
            return self.val_transform(data), self.targets[index], index
        else:
            if self.aug:
                return self.train_transform(data), self.targets[index], index
            else:
                return data, self.targets[index]

    def get_balanced_sampler(self):
        labels = np.array(self.label)       # e.g., size (n, )
        unique_labels = list(np.unique(labels))   # e.g., size (6, )
        transformed_labels = torch.tensor([unique_labels.index(i) for i in labels])             # e.g., size (n, )
        class_sample_count = np.array([len(np.where(labels==t)[0]) for t in unique_labels])     # e.g., size (6, )
        weight = 1. / class_sample_count    # make every class to have balanced chance to be chosen
        samples_weight = torch.tensor([weight[t] for t in transformed_labels])
        self.class_sample_count = class_sample_count
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
        return sampler


class gen_test_dataset(Dataset):
    def __init__(self, per_client_data, per_client_label, args):
        self.data = per_client_data
        self.targets = per_client_label
        self.dataset = args.dataset

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        
        data = self.data[index]
        # plt.imshow(data)
        # plt.savefig("./a.jpg")

        if self.dataset == "CUB":
            return self.val_transform(data), self.targets[index], index
        else:
            return data, self.targets[index]