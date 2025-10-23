import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # system setting
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--seed', type=int,  default=0, help='random seed')
    parser.add_argument('--gpu', type=str,  default="1", help='GPU to use')

    # model setting
    parser.add_argument('--use_dr_balancer', type=bool, default=False, help="true: \
                        false: ours, fedloge, fednoro")
    parser.add_argument('--use_additive', type=bool, default=True, help="true: \
                        false: ours, fedloge, fednoro")
    parser.add_argument('--loss_type', type=str, default="ce", help="ce, dr_loss, focal_loss, nc_loss")
    parser.add_argument('--etf_norm_rate', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=3e-4)
    parser.add_argument('--norm_type', type=str, default="none")
    parser.add_argument('--train_bn_bias', type=bool, default=True)
    parser.add_argument('--train_bn_weight', type=bool, default=True)
    parser.add_argument('--features_regularization', type=float, default=0.01)
    parser.add_argument('--optimizer', type=str, default="Adam")
    parser.add_argument('--temperature', type=float, default=2)
    parser.add_argument('--threshold', default=0.5, type=float, help='pseudo label threshold')
    parser.add_argument('--batch_size_auxiliary', default=128, type=int, help='batch size for client selection')


    # basic setting
    parser.add_argument('--exp', type=str,
                        default='ours', help='experiment name')
    parser.add_argument('--dataset', type=str,
                        default='cifar10', help='dataset name')
    parser.add_argument('--model', type=str,
                        default='Resnet18', help='model name')
    parser.add_argument('--batch_size', type=int,
                        default=16, help='batch_size per gpu')
    parser.add_argument('--first_lr', type=float,  default=3e-4,
                        help='base learning rate')
    parser.add_argument('--second_lr', type=float,  default=5e-4,
                        help='base learning rate')
    parser.add_argument('--weights', type=int,  default=None)

    # for FL
    parser.add_argument('--n_clients', type=int,  default=20,
                        help='number of users') 
    parser.add_argument('--iid', type=int, default=0, help="i.i.d. or non-i.i.d.")
    parser.add_argument('--non_iid_prob_class', type=float,
                        default=0.9, help='parameter for non-iid')
    parser.add_argument('--alpha_dirichlet', type=float,
                        default=0.9, help='parameter for non-iid')
    parser.add_argument('--local_ep', type=int, default=5, help='local epoch')
    parser.add_argument('--rounds', type=int,  default=100, help='rounds')

    parser.add_argument('--s1', type=int,  default=10, help='stage 1 rounds')
    parser.add_argument('--warm', type=int,  default=0)

    # noise & imbalance
    parser.add_argument('--level_n_system', type=float, default=0.4, help="fraction of noisy clients")
    parser.add_argument('--level_n_lowerb', type=float, default=0.2, help="lower bound of noise level")
    parser.add_argument('--level_n_upperb', type=float, default=0.4, help="upper bound of noise level")
    parser.add_argument('--n_type', type=str, default="symmetric", help="type of noise, \
                        symmetric, asymmetric")
    parser.add_argument('--imb_ratio', type=float, default=0.1, help="imbalance ratio")
    
    # fednoro
    parser.add_argument('--begin', type=int,  default=10, help='ramp up begin')
    parser.add_argument('--end', type=int,  default=49, help='ramp up end')
    parser.add_argument('--a', type=float,  default=0.8, help='a')
    
    # fedcorr
    
    
    args = parser.parse_args()
    return args