import argparse
import logging
import os
import random
import sys
import time
from utils import test_single_volume
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored
from datasets.dataset_synapse import *
from datasets.data_process import *
import torch.optim as optim
# from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from utils import *
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
import pandas as pd
from networks.unet3D_M_transformer.unet_model import UNETR_clinical
from utils import _neg_partial_log


def trainer_synapse(args, snapshot_path):

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s,', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    max_iterations = args.max_iterations

    heart_dataset_R = pd.read_excel('',
                                    sheet_name='INTEGRAT')


    cat_dims, cat_idxs, con_idxs, X,nan_mask,y,DFS, train_mean, train_std,names = clinical_embd(heart_dataset_R)



    train_path = args.root_path
    train_list = create_list(train_path)
    db_train = ImageDataset(data_list = train_list,transform = [Resize3D([128,128],check=True),TorchIOAugmentation()], clinical = [X,nan_mask,y,DFS,cat_idxs,train_mean, train_std,names])
    cat_dims = np.append(np.array([1]),np.array(cat_dims)).astype(int)
    model = UNETR_clinical(cat_dims = cat_dims).cuda()



    test_path = args.volume_path
    test_list = create_list(test_path)
    db_test = ImageDataset(data_list = test_list,transform = [Resize3D([128,128],check=True),TorchIOAugmentation()],clinical = [X,nan_mask,y,DFS,cat_idxs,train_mean, train_std,names])



    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    testloader = DataLoader(db_test, batch_size=1, pin_memory=True)



    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()


    seg_loss_func = Dice_and_FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=0.001)
    iter_num = 0
    max_epoch = args.max_epochs
    best_dice = 0
    iterator = tqdm(range(max_epoch), ncols=70)
    loss_list = []
    dice_score_lst = []
    mean_loss_list = []
    mean_dice_list = []
    performance_list = []

    for epoch_num in iterator:
        epoch_num = epoch_num + 1
        total_acc = 0.

        model.train()
        for i_batch, (data, _) in enumerate(trainloader):
            torch.cuda.empty_cache()

            data_img = Variable(data[0].cuda())  # A Variable wraps a Tensor. It supports nearly all the API’s defined by a Tensor.
            data_clinical = data[1]# A Variable wraps a Tensor. It supports nearly all the API’s defined by a Tensor.
       


            x_categ, x_cont, label_censor, DFS, cat_mask, con_mask = data_clinical[0].cuda(), data_clinical[1].cuda(), data_clinical[2].cuda().unsqueeze(-1), \
                                                         data_clinical[3].cuda(), data_clinical[4].cuda().long(), data_clinical[5].cuda()

            _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, False)


            classify,lbl_pred_all = model(data_img,x_categ_enc, x_cont_enc,True)



            loss_surv = _neg_partial_log(lbl_pred_all, DFS, label_censor)+nn.BCELoss()(lbl_pred_all,label_censor.float())
    
            classify_loss = nn.BCELoss()(classify,label_censor.float())

            loss = classify_loss + loss_surv



            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_


            loss_list.append(loss.item())



        total_acc = total_acc/len(trainloader)

        mean_loss = np.mean(loss_list)
        mean_loss_list.append(mean_loss)

        mean_dice = np.mean(dice_score_lst)
        mean_dice_list.append(mean_dice)


        print('================================================================================')
        print('epoch {}   ||   slice_mean_loss:{}   ||  mean_dice: :{:3f} ||  total_acc:{:3f} '.format(epoch_num,mean_loss,mean_dice,total_acc))

        model.eval()
        best_dice = 0.
        acc_total = 0.
        all_event_times = []
        all_censorships = []
        pre_censorship_lst = []
        all_risk_scores = []
        for i_batch, (data, infm) in tqdm(enumerate(testloader)):
            torch.cuda.empty_cache()

            data_img = Variable(
                data[0].cuda())  # A Variable wraps a Tensor. It supports nearly all the API’s defined by a Tensor.
            data_clinical = data[1]  # A Variable wraps a Tensor. It supports nearly all the API’s defined by a Tensor.


            x_categ, x_cont, test_label_censor, test_label_dfs, cat_mask, con_mask = data_clinical[0].cuda(), data_clinical[1].cuda(), \
                                                                data_clinical[2].cuda().unsqueeze(-1), \
                                                                data_clinical[3].cuda(), data_clinical[4].cuda().long(), data_clinical[5].cuda()



            __, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, False)

            pre_censorship,risk = model(data_img,x_categ_enc, x_cont_enc)  # outputs[8,1,256,256]





            pre_censorship[pre_censorship>=0.5]=1
            pre_censorship[pre_censorship<0.5]=0


            all_risk_scores.append(risk.cpu().detach().numpy())
            all_event_times.append(test_label_dfs.cpu().detach().numpy())
            all_censorships.append(test_label_censor.cpu().detach().numpy())
            pre_censorship_lst.append(pre_censorship.cpu().detach().numpy())




        acc_total = acc_total / len(testloader)


        all_risk_scores = np.concatenate(
            (np.stack(all_risk_scores[:-1], axis=1).reshape(-1), (all_risk_scores[-1]).reshape(-1)), axis=0)
        all_event_times = np.concatenate(
            (np.stack(all_event_times[:-1], axis=1).reshape(-1), (all_event_times[-1]).reshape(-1)), axis=0)
        all_censorships = np.concatenate(
            (np.stack(all_censorships[:-1], axis=1).reshape(-1), (all_censorships[-1]).reshape(-1)), axis=0)
        pre_censorship_lst = np.concatenate(
            (np.stack(pre_censorship_lst[:-1], axis=1).reshape(-1), (pre_censorship_lst[-1].reshape(-1))), axis=0)

        censor_auc, recall, precision, accuracy, jaccard = classification_evalution(pre_censorship_lst, all_censorships)


        cindex = concordance_index_censored(all_censorships.astype(bool), all_event_times, all_risk_scores)

        print('censor_auc: {:.4f} , recall: {:.4f} , precision: {:.4f} , accuracy: {:.4f} , jaccard: {:.4f} '.format(censor_auc, recall, precision, accuracy, jaccard))
        print('Survival: cindex: {:.4f} '.format(cindex[0]))

        performance = accuracy
        performance_list.append(accuracy)
        print('train loss list : ', mean_loss_list)
        print('val performance list:', performance_list)


        if performance>=best_dice:
            best_dice = performance
            best_dice_ = "{:.3f}".format(performance)
            save_mode_path = os.path.join(snapshot_path, 'best_epoch_' + str(epoch_num)+'_'+ best_dice_+ '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))


    return mean_loss_list,performance_list