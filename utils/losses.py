import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.class_balanced_loss import CB_loss, DotRegressionLoss, NC_loss
from utils.regularizers import set_features_regularization, set_weight_decay

def set_loss(args, cls_num_list):
    loss_function = get_loss_function(cls_num_list, args)
    loss_function = set_weight_decay(loss_function, cls_num_list, args)
    # loss_function = set_features_regularization(loss_function, args)
    return loss_function
    
    
class LogitAdjust(nn.Module):
    def __init__(self, args, cls_num_list, tau=1, weight=None):
        super(LogitAdjust, self).__init__()
        # cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_num_list = torch.tensor(cls_num_list, dtype=torch.float).to(args.device)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight

    def forward(self, x, target):
        x_m = x + self.m_list
        return F.cross_entropy(x_m, target, weight=self.weight)


class LA_KD(nn.Module):
    def __init__(self, args, cls_num_list, tau=1):
        super(LA_KD, self).__init__()
        # cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_num_list = torch.tensor(cls_num_list, dtype=torch.float).to(args.device)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)

    def forward(self, x, target, soft_target, w_kd):
        x_m = x + self.m_list
        log_pred = torch.log_softmax(x_m, dim=-1)
        log_pred = torch.where(torch.isinf(log_pred), torch.full_like(log_pred, 0), log_pred)

        kl = F.kl_div(log_pred, soft_target, reduction='batchmean')

        return w_kd * kl + (1 - w_kd) * F.nll_loss(log_pred, target)


class LogitAdjust_our(nn.Module):
    def __init__(self, args, cls_num_list, tau=1, weight=None, epsilon=1e-8):
        super(LogitAdjust_our, self).__init__()
        # cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_num_list = torch.tensor(cls_num_list, dtype=torch.float).to(args.device)
        ### use epsilon to avoid zero class num
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        # m_list = torch.where(torch.isinf(m_list), torch.full_like(m_list, 0), m_list)
        # m_list = torch.where(torch.isnan(m_list), torch.full_like(m_list, 0), m_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight
        self.use_additive = args.use_additive
        
    def forward(self, x):
        if self.use_additive:
            return x + self.m_list
        else:
            return x * torch.exp(-1 * self.m_list)
        
        
class LogitAdjust_our_add(nn.Module):
    def __init__(self, args, cls_num_list, tau=1, weight=None, epsilon=1e-8):
        super(LogitAdjust_our_add, self).__init__()
        # cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_num_list = torch.tensor(cls_num_list, dtype=torch.float).to(args.device)
        ### use epsilon to avoid zero class num
        cls_p_list = cls_num_list / cls_num_list.sum() 
        m_list = tau * torch.log(cls_p_list)
        # m_list = torch.where(torch.isinf(m_list), torch.full_like(m_list, 0), m_list)
        # m_list = torch.where(torch.isnan(m_list), torch.full_like(m_list, 0), m_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight
        self.use_additive = args.use_additive
        
    def forward(self, x):
        if self.use_additive:
            return x + self.m_list
        else:
            return x * torch.exp(-1 * self.m_list)
        
def get_loss_function(img_num_per_cls: list, args, reduction = 'none'):
    loss_type = args.loss_type
    use_features = False
    if loss_type == "ce":
        def CE_lossFunc(logits, labels, model = None, features = None, epoch = 0):
            return F.cross_entropy(logits, labels)
        loss = CE_lossFunc
    elif loss_type == "cb":
        if "cb_type" in vars(args.optim):
            use_features = args.cb_type == "dr"
            def CB_lossFunc(logits, labels, model = None, features = None, epoch = 0): #defince CB loss function
                drw = check_drw(args, epoch, logits, labels, reduction)
                if drw is not None:
                    return drw
                return CB_loss(labels, logits, img_num_per_cls, args.data.n_classes,
                args.optim.cb_type, args.optim.cb_beta, args.optim.cb_gamma, args.device,
                model, features, reduction=reduction)
        else:
            def CB_lossFunc(logits, labels, model = None, features = None, epoch= 0): #defince CB loss function
                drw = check_drw(args, epoch, logits, labels, reduction)
                if drw is not None:
                    return drw
                return CB_loss(labels, logits, img_num_per_cls, args.data.n_classes,
                "softmax", 0.9999, 2.0, args.device, model, features, reduction=reduction)
        loss = CB_lossFunc
    elif loss_type == "dr_loss":
        use_features = True
        def DR_Loss(logits, labels, model, features, epoch = 0):
            drw = check_drw(args, epoch, logits, labels, reduction)
            if drw is not None:
                return drw
            return DotRegressionLoss(logits, labels, model, features, reduction)
        loss = DR_Loss
    elif loss_type == "focal_loss":
        def CB_lossFunc(logits, labels, model = None, features = None, epoch = 0): #defince CB loss function
            drw = check_drw(args, epoch, logits, labels, reduction)
            if drw is not None:
                return drw
            return CB_loss(labels, logits, img_num_per_cls, args.data.n_classes,
                        "focal", 1.0, args.optim.cb_gamma, args.device,
                        model, features, reduction=reduction)
        loss = CB_lossFunc
    elif loss_type == "nc_loss":
        use_features = True
        def NC_lossFunc(logits, labels, model, features, epoch = 0):
            drw = check_drw(args, epoch, logits, labels, reduction)
            if drw is not None:
                return drw
            return NC_loss(logits, labels, features, args.optim.nc_lambda1, args.optim.nc_lambda2, reduction)
        loss = NC_lossFunc
    else:
        assert False, "Invalid loss function type."
    return loss


def check_drw(args, epoch: int, logits: torch.Tensor, labels: torch.Tensor, reduction: str = "mean"):
    if args.optim.drw <= epoch:
        return None
    else:
        return  nn.CrossEntropyLoss(reduction=reduction)(logits, labels)