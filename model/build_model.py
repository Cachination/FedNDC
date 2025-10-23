import torch.nn as nn
from .all_models import get_model, modify_last_layer, get_model


def set_model(args):
    # choose different Neural network model for different args
    model, args = get_model(args)
    return model, args
    # else:
    #     model = get_model(args)
    #     model, _ = modify_last_layer(args.model, model, args.n_classes)
    #     model = model.to(args.device)
    #     return model


