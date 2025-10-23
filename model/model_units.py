from datetime import datetime as dt
from types import SimpleNamespace
import os, sys, glob, json, math
import torch
from torch import nn
import numpy as np
from scipy.stats import ortho_group
from abc import ABCMeta, abstractmethod, abstractclassmethod
from torch.nn import functional as F
import torch.optim as optim
import copy

class LogitAdjuster(nn.Module):
    def __init__(self, tau: float= 0, img_num_per_cls: list = None, for_training: bool = False, use_additive: bool = True):
        super().__init__()
        self.tau = tau
        if tau > 0:
            assert img_num_per_cls is not None, "img_num_per_cls should not be None when tau > 0"
            img_num_per_cls = np.array(img_num_per_cls)
            pY = img_num_per_cls / img_num_per_cls.sum()
            pY = torch.log(torch.Tensor(pY)).unsqueeze(0)
            self.register_buffer("pY", pY, persistent=False)
            self.for_training = for_training
            self.use_additive = use_additive
            
    
    def forward(self, x):
        if self.tau > 0:
            if self.use_additive:
                if not self.for_training and not self.training:
                    return x - self.tau * self.pY
                elif self.for_training and self.training:
                    return x + self.tau * self.pY
            else:
                return x * torch.exp(-1 * self.tau * self.pY)
        return x


def get_standard_etf(n_classes, device):
    ans = torch.eye(n_classes) - torch.ones((n_classes, n_classes)) / n_classes
    ans = ans * np.sqrt(n_classes / (n_classes - 1))
    return ans.to(device = device)
def get_general_etf(n_classes, n_features, device):
    ans = get_standard_etf(n_classes, device)
    ans = torch.matmul(
        torch.tensor(ortho_group.rvs(n_features)[:, :n_classes]).to(dtype=torch.float32, device=device), ans)
    return ans.T

class ETFClassifier(nn.Module):
    def __init__(self, in_features, out_features, norm_rate: float = 1, train_norm = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        weight_vec = get_general_etf(out_features, in_features, "cpu")
        self.weight_vec = nn.Parameter(weight_vec)
        self.norm_rate = norm_rate
        norm_tensor = torch.ones((out_features, 1)) -\
                    (norm_rate - 1)/((out_features-1) * norm_rate) *  torch.arange(out_features).unsqueeze(dim=1)
        self.weight_norm = nn.Parameter(norm_tensor.to(torch.float32))    
        
        bias = torch.empty(out_features).uniform_(-1 / in_features, 1 / in_features)
        self.bias = nn.Parameter(bias)

    def forward(self, x):
        
        return torch.matmul(x, self.get_weight().T) + self.bias
    
    # return the weight as a linear layer
    def get_weight(self):
        return self.weight_vec * self.weight_norm
    def set_weight(self, tensor: torch.Tensor):
        norm = torch.linalg.norm(tensor, dim=1, keepdim=True)
        self.weight_vec.data = tensor/norm
        self.weight_norm.data = norm
        return self
    
    @classmethod
    def convert_etf_weights_to_linear(cls, model, model_dict):
        # change vec + norm -> weight
        vec_tensor: torch.Tensor
        norm_tensor: torch.Tensor
        vec_name: str
        for name in list(model_dict.keys()):
            if model.belongs_to_fc_layers(name):
                if name.endswith("weight_vec"):
                    vec_tensor = model_dict.pop(name)
                    vec_name = name
                elif name.endswith("weight_norm"):
                    norm_tensor = model_dict.pop(name)
        weight_tensor = vec_tensor * norm_tensor
        model_dict[vec_name[:-4]] = weight_tensor
        return model_dict
    
    @classmethod
    def convert_linear_weights_to_etf(cls, model, model_dict):
        # change vec + norm -> weight
        weights_tensor: torch.Tensor
        weights_name: str
        for name in list(model_dict.keys()):
            if model.belongs_to_fc_layers(name):
                if name.endswith("weight"):
                    weights_tensor = model_dict.pop(name)
                    weights_name = name
        norm_tensor = torch.linalg.norm(weights_tensor, dim=1, keepdim=True)
        vec_tensor = weights_tensor / norm_tensor
        model_dict[weights_name + "_vec"] = vec_tensor
        model_dict[weights_name + "_norm"] = norm_tensor
        return model_dict

        
class sparse_ETF_Classifier(nn.Module):
    def __init__(self, feat_in, num_classes, fix_bn=False, LWS=False, reg_ETF=False):
        """
        初始化ETF分类器。

        参数:
        feat_in: 输入特征的数量。
        num_classes: 类别数量。
        fix_bn: 是否固定批量归一化层的参数。
        LWS: 是否使用可学习的权重标准化。
        reg_ETF: 是否正则化ETF。
        """
        super(sparse_ETF_Classifier, self).__init__()
        # 初始化后三个矩阵P、I、M都是num_classes*num_classes维的
        # 生成随机正交矩阵P
        P = self.generate_random_orthogonal_matrix(feat_in, num_classes)
        # 生成单位矩阵I和全1矩阵one
        I = torch.eye(num_classes)
        one = torch.ones(num_classes, num_classes)
        # 初始化的ETF矩阵，feat_in * num_classes
        M = np.sqrt(num_classes / (num_classes-1)) * torch.matmul(P, I-((1/num_classes) * one))
        self.ori_M = M.cuda()

        self.LWS = LWS
        self.reg_ETF = reg_ETF
#        if LWS:
#            self.learned_norm = nn.Parameter(torch.ones(1, num_classes))
#            self.alpha = nn.Parameter(1e-3 * torch.randn(1, num_classes).cuda())
#            self.learned_norm = (F.softmax(self.alpha, dim=-1) * num_classes)
#        else:
#            self.learned_norm = torch.ones(1, num_classes).cuda()

        self.BN_H = nn.BatchNorm1d(feat_in)
        if fix_bn:
            self.BN_H.weight.requires_grad = False
            self.BN_H.bias.requires_grad = False

    def generate_random_orthogonal_matrix(self, feat_in, num_classes):
        a = np.random.random(size=(feat_in, num_classes))
        # 进行QR分解，分解后形成一个正交矩阵P和后续不再使用的上三角矩阵_
        P, _ = np.linalg.qr(a)
        P = torch.tensor(P).float()
        # 确保P是正交矩阵
        assert torch.allclose(torch.matmul(P.T, P), torch.eye(num_classes), atol=1e-07)\
            , torch.max(torch.abs(torch.matmul(P.T, P) - torch.eye(num_classes)))
        return P

    def forward(self, x):
        # 对特征归一化
        if x.size(0) > 1:  # Only apply BatchNorm if batch size > 1
            x = self.BN_H(x)
            
        x = x / torch.clamp(
            torch.sqrt(torch.sum(x ** 2, dim=1, keepdims=True)), min=1e-8)
        
        return x

    
    def gen_sparse_ETF(self, feat_in=512, num_classes=100, beta=0.6):
        # Initialize ETF
        etf = copy.deepcopy(self.ori_M)
        # Sparsify ETF
        num_zero_elements = int(beta * feat_in * num_classes)
        zero_indices = np.random.choice(feat_in * num_classes, num_zero_elements, replace=False)
        etf_flatten = etf.flatten()
        etf_flatten[zero_indices] = 0
        sparse_etf = etf_flatten.reshape(feat_in, num_classes)
        
        # Adjust non-zero elements
        sparse_etf = torch.tensor(sparse_etf, requires_grad=True)
        
        
        # Create a mask where the initial tensor is non-zero
        mask = (sparse_etf != 0).float()

        # Optimizer
        optimizer = optim.Adam([sparse_etf], lr=0.0001)

        # Number of optimization steps
        n_steps = 10000
        loss_log = []
        for step in range(n_steps):
            optimizer.zero_grad()
            
            # Constraint 1: L2 norm of each row should be 1
            # 每一行代表一个特征维度，每一列代表一个类别的向量表示，这里计算的是每列的L2范数，不是每行的
            row_norms = torch.norm(sparse_etf, p=2, dim=0)
            norm_loss = torch.sum((row_norms - 0.1)**2)
            
            # Constraint 2: Maximize the angle between vectors (minimize cosine similarity)
            normalized_etf = sparse_etf / row_norms
            cos_sim = torch.mm(normalized_etf.t(), normalized_etf)
            # 将对角线元素填充为-1，因为余弦相似度在对角线上是1，避免自己与自己的相似性
            torch.diagonal(cos_sim).fill_(-1)
            angle_loss = -torch.acos(cos_sim.max(dim=1)[0].clamp(-0.99999, 0.99999)).mean()
            # angle_loss = -torch.sum(torch.acos(torch.clamp(cos_sim, -0.9999999, 0.9999999))) 
            # Total loss
            loss = norm_loss + angle_loss
    
            # Backpropagation
            loss.backward()
            
            # Apply the mask to the gradients,将掩码应用到梯度上，确保稀疏的部分保持不变(原本为0的一直是0)
            if sparse_etf.grad is not None:
                sparse_etf.grad *= mask
                
                
            optimizer.step()
            loss_log.append(loss)
            if step % 100 == 0:
                print(f"Step {step}, Loss {loss.item()}")
                
        self.test_etf(sparse_etf)     
                
        return sparse_etf
    
    def test_etf(self, sparse_etf):
        # Normalize each column to have L2 norm = 1
        col_norms = torch.norm(sparse_etf, p=2, dim=0, keepdim=True)
        normalized_etf = sparse_etf / col_norms

        # Compute cosine similarities
        cosine_similarities = torch.mm(normalized_etf.t(), normalized_etf)

        # Zero out the diagonal (we don't want to compare vectors with themselves)
        torch.diagonal(cosine_similarities).fill_(float('nan'))

        # Compute angles in radians
        angles_radians = torch.acos(torch.clamp(cosine_similarities, -1, 1))

        # Convert angles from radians to degrees
        angles_degrees = angles_radians * (180 / np.pi)

        # Convert to numpy array
        angles_degrees_numpy = angles_degrees.cpu().detach().numpy()

        # Calculate mean and variance of angles, ignoring NaNs
        angle_mean = np.nanmean(angles_degrees_numpy)
        angle_variance = np.nanvar(angles_degrees_numpy)

        # Calculate mean and variance of norms
        col_norms_numpy = col_norms.cpu().detach().numpy()
        norm_mean = np.mean(col_norms_numpy)
        norm_variance = np.var(col_norms_numpy)

        print(f"Angle Mean: {angle_mean}, Angle Variance: {angle_variance}")
        print(f"Norm Mean: {norm_mean}, Norm Variance: {norm_variance}")   

#########################
## Abstract class 
#########################      
        
# Abstract class for NN model
class MyModel(nn.Module, metaclass = ABCMeta):
    def __init__(self):
        super().__init__()
        self.outFeature = False
        self.outAllFeatures = False
        self.fineGrainedFeatures = False
    # input parameter's name and return whether it belongs to final fc layer
    @abstractclassmethod
    def belongs_to_fc_layers(cls, name: str):
        raise NotImplementedError()
    @abstractclassmethod
    def belongs_to_bn_bias(cls, name: str):
        raise NotImplementedError()
    @abstractclassmethod
    def belongs_to_bn_weight(cls, name: str):
        raise NotImplementedError()
    # judge whether the batch normalization maps identity feature (not residual)
    # be careful, cause this function does not judge wheteher the module is batch normalization.
    @abstractclassmethod
    def belongs_to_identity_bn(cls, name: str):
        raise NotImplementedError()
    @abstractmethod
    # return fc_weights in tensor
    def get_fc_weight(self):
        raise NotImplementedError()
    @abstractmethod
    # set fc_weights by tensor
    def set_fc_weight(self, tensor: torch.Tensor):
        raise NotImplementedError()
    @abstractmethod
    def get_fc_bias(self):
        raise NotImplementedError()
    @abstractmethod
    def set_fc_bias(self, tensor: torch.Tensor):
        raise NotImplementedError()
    # return params belonging to the fc layer
    @abstractmethod
    def get_fc_params_list(self):
        raise NotImplementedError()
    @abstractmethod
    def forward_last_layer(self, x: torch.Tensor):
        raise NotImplementedError()
    # when wd's target is restricted, check if the module is target or not
    @abstractmethod
    def check_valid_wd_target(self, name: str, args):
        raise NotImplementedError()
    # the length of features list when outAllFeatures is true, 
    @abstractmethod
    def get_features_n(self):
        raise NotImplementedError()
    @abstractmethod
    def get_last_layer_weights_names_list(self):
        raise NotImplementedError()
    
    # append feature tensor to self.features and return 
    def f_append(self, x: torch.Tensor):
        self.features.append(x)
        return x
    
    def get_trained_params_list(self, args):
        return [param for name, param in self.named_parameters() 
                if (args.train_bn_bias or not self.belongs_to_bn_bias(name)) and
                    (args.train_bn_weight or not self.belongs_to_bn_weight(name))]
        # elif args.project_name == "second":
        #     return [param for name, param in self.named_parameters() if self.belongs_to_fc_layers(name)]
    def get_trained_params_names_list(self, args):
        return [name for name, param in self.named_parameters()
                if (args.train_bn_bias or not self.belongs_to_bn_bias(name)) and
                    (args.train_bn_weight or not self.belongs_to_bn_weight(name))]
        # elif args.project_name == "second":
        #     return [name for name, param in self.named_parameters() if self.belongs_to_fc_layers(name)]
    # this can load a model which has a different ETFClassifier
    # def load_pretrained_model_dict(self, args):
    #     pretrained_path = os.path.join(args.model.pretrained_path, str(args.seeds), BEST_PARAM_FILE_NAME)
    #     model_dict = torch.load(pretrained_path, map_location=args.device)
    #     loaded_args_path = os.path.join(args.model.pretrained_path, str(args.seeds), ARGS_FILE_NAME)
    #     if args.model.use_etf:
    #         # Linear -> ETF
    #         model_dict = ETFClassifier.convert_linear_weights_to_etf(self, model_dict)
    #     else:
    #         # ETF -> Linear
    #         model_dict = ETFClassifier.convert_etf_weights_to_linear(self, model_dict)
        
    #     if args.model.load_only_representation:
    #         for name in list(model_dict.keys()):
    #             if self.belongs_to_fc_layers(name):
    #                 model_dict.pop(name)
    #     self.load_state_dict(model_dict, strict = (not args.model.load_only_representation) \
    #                          and (not args.model.ignore_restrict_load ))
    #     return self
    
# Used to maintain consistency in the relationship between the magnitude of the gradient and that of the impact on Weight Decay
class copy_and_aggregate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, num_features: int):
        return x * torch.ones((num_features,), device = x.device)
    
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out.mean(0,keepdim = True), None

# Mdofied version of the Pytorch's implementation
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html#BatchNorm2d
class OneParamBN(torch.nn.BatchNorm2d):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )
        self.weight = torch.nn.Parameter(torch.empty(1, **factory_kwargs), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.empty(1, **factory_kwargs), requires_grad=True)
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            copy_and_aggregate.apply(self.weight, self.num_features),
            copy_and_aggregate.apply(self.bias, self.num_features),
            
            bn_training,
            exponential_average_factor,
            self.eps,
        )

        
def is_batchnorm(module: nn.Module):
    return isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d) or isinstance(module, OneParamBN)
