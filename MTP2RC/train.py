import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn


from trainer_NPY_3D_C_Transformer import trainer_synapse


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='D:/_0_code/ICIAR2018-master/dataset/for128/train_slice', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=300000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=250, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=8, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--base_lr', type=float,  default=0.000005,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=128, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50+ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--volume_path', type=str,
                    default='D:/project/Relabeled_data/Relabeled_test_2.txt', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--CROP_SIZE', type=list,
                    default=[128, 128])  # for acdc volume_path=root_dir
parser.add_argument('--resize', type=bool,
                    default=True, )  # for acdc volume_path=root_dir
parser.add_argument('--val_path', type=str,
                    default='D:/project/Relabeled_data/Relabeled_val.txt', help='root dir for validation volume data')  # for acdc volume_path=root_dir


args = parser.parse_args()


if __name__ == "__main__":
    cudnn.benchmark = False
    cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset


    dataset_config = {
        'Synapse': {
            'root_path': "",
            'volume_path': "",
            'num_classes': 1,
        },
    }

    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.is_pretrain = True
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "./model_rectal/"


    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)


    trainer = {'Synapse': trainer_synapse,}
    trainer[dataset_name](args, snapshot_path)