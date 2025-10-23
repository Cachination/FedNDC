import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pretrainedmodels
from .efficientnet import efficientnet_b0
from .efficientnet import efficientnet_b1
from .efficientnet import efficientnet_b2
from .efficientnet import efficientnet_b3
from .efficientnet import efficientnet_b4
from .efficientnet import efficientnet_b5
from .efficientnet import efficientnet_b6
from .efficientnet import efficientnet_b7
from .network_arch_resnet import ResnetEncoder
from .model_units import ETFClassifier


def get_model(args, outFeature=True, get_residual = False):
    if args.model == 'Resnet34' or args.model == 'IMResNet34':
        model = ResnetEncoder(args, 34, False,
            n_classes=args.n_classes, poolSize=4, outFeature=outFeature, useETF=args.use_etf, 
            getResidual = get_residual)
    elif args.model == 'Resnet18' or args.model == 'IMResNet18':
        if args.dataset == "speech" :
            model = ResnetEncoder(args, 18, False,
                n_classes=args.n_classes, poolSize=4, outFeature=outFeature, useETF=args.use_etf, 
                getResidual = get_residual, input_channel=1)
        else:
            model = ResnetEncoder(args, 18, False,
                n_classes=args.n_classes, poolSize=4, outFeature=outFeature, useETF=args.use_etf, 
                getResidual = get_residual, input_channel=3)

    model.to(args.device)
    return model, args


def modify_last_layer(model, args, normed=False, bias=True):
    """modify the last layer of the CNN model to fit the num_classes
    Args:
      model_name: model name
      model: CNN model
      num_classes: class number
    Returns:
      model: the desired model
    """
    num_ftrs = model.encoder.fc.in_features
    fc_bias = model.encoder.fc.bias
    model.encoder.fc = ETFClassifier(num_ftrs, args.n_classes, args.etf_norm_rate)
    model.encoder.fc.bias = fc_bias
    model.to(args.device)
    return model


def classifier(num_features, num_classes):
    last_linear = nn.Linear(num_features, num_classes)
    return last_linear



def get_feature_length(model_name, model):
    """get the feature length of the last feature layer
    Args:
      model_name: model name
      model: CNN model
    Returns:
      num_ftrs: the feature length of the last feature layer
    """
    if 'Vgg' in model_name:
        num_ftrs = model.classifier._modules['6'].in_features
    elif 'Dense' in model_name:
        num_ftrs = model.classifier.in_features
    elif 'Resnet' in model_name:
        num_ftrs = model.fc.in_features
    elif 'Efficient' in model_name:
        num_ftrs = model._fc.in_features
    elif 'RegNet' in model_name:
        num_ftrs = model.head.fc.in_features
    else:
        num_ftrs = model.last_linear.in_features

    return num_ftrs