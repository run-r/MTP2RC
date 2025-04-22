import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
from torch import Tensor
from datasets.data_process import Eliminate_false_positives
from typing import Iterable,Set
from scipy.ndimage import distance_transform_edt as distance
import torch.nn.functional as F
from torch.autograd import Variable
from datasets.data_process import resize_image_itk
from scipy.ndimage import morphology, gaussian_filter
from sklearn.metrics import roc_auc_score,recall_score,precision_score,accuracy_score,jaccard_score
from sklearn.preprocessing import LabelBinarizer, StandardScaler
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from lifelines.utils import concordance_index as ci
import xlsxwriter as xw



def IoU(seg, gt, ratio=0.5):
    """
    function to calculate the dice score
    """

    seg = seg.flatten()
    gt = gt.flatten()
    if type(seg) == torch.Tensor:
        seg[seg > ratio] = 1.0
        seg[seg < ratio] = 0.0
    else:
        seg[seg > ratio] = np.float32(1)
        seg[seg < ratio] = np.float32(0)

    mix = (gt * seg).sum()
    iou = mix/(gt.sum() + seg.sum()+1e-5-mix)
    return iou


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = 1e-3

    def forward(self, input, target):
        input = input.clamp(self.eps, 1 - self.eps)
        loss = - (target * torch.pow((1 - input), self.gamma) * torch.log(input) +
                  (1 - target) * torch.pow(input, self.gamma) * torch.log(1 - input))
        return loss.mean()



class Dice_and_FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(Dice_and_FocalLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.dice_loss(input, target) + self.focal_loss(input, target)
        return loss


class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, input, target):
        assert input.size() == target.size(), 'predict {} & target {} shape do not match'.format(input.size(),
                                                                                                  target.size())
        pred = input.reshape(-1)
        truth = target.reshape(-1)

        # BCE loss
        bce_loss = nn.BCELoss()(pred, truth).double()
        smooth = 1e-5
        # Dice Loss
        dice_coef = (2.0 * (pred * truth).double().sum()) / (
                (pred**2).double().sum() + (truth**2).double().sum() + smooth
        )

        return bce_loss,(1 - dice_coef)



class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target)
        z_sum = torch.sum(score)
        dice = (2 * intersect) / (z_sum + y_sum + smooth)
        loss = 1 - dice
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        dice_1 = self._dice_loss(inputs, target)
        return dice_1





def classification_evalution(all_censorships,pre_censorship_lst):

    censor_auc = roc_auc_score(np.array(pre_censorship_lst), np.array(all_censorships))
    recall = recall_score(np.array(pre_censorship_lst), np.array(all_censorships))
    precision = precision_score(np.array(pre_censorship_lst), np.array(all_censorships))
    accuracy = accuracy_score(np.array(pre_censorship_lst), np.array(all_censorships))
    jaccard = jaccard_score(np.array(pre_censorship_lst), np.array(all_censorships))


    return censor_auc,recall,precision,accuracy,jaccard

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd_score = metric.binary.asd(pred, gt)
        return dice, hd95, asd_score
    else:
        return 0, 50,10


def dice_coeff(seg, gt, ratio=0.5):
    """
    function to calculate the dice score
    """

    seg = seg.flatten()
    gt = gt.flatten()
    if type(seg) == torch.Tensor:
        seg[seg > ratio] = 1.0
        seg[seg < ratio] = 0.0
    else:
        seg[seg > ratio] = np.float32(1)
        seg[seg < ratio] = np.float32(0)
    dice = (2 * (gt * seg).sum())/(gt.sum() + seg.sum()+1e-5)
    return dice





def _neg_partial_log(prediction, T, E):

    current_batch_len = len(prediction)
    R_matrix_train = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_matrix_train[i, j] = T[j] >= T[i]

    train_R = torch.FloatTensor(R_matrix_train)
    train_R = train_R.cuda()

    # train_ystatus = torch.tensor(np.array(E),dtype=torch.float).to(prediction.device)
    train_ystatus = E

    theta = prediction.reshape(-1)

    exp_theta = torch.exp(theta)
    loss_nn = - torch.mean((theta - torch.log(torch.sum(exp_theta * train_R, dim=1))) * train_ystatus)

    return loss_nn



def xw_toExcel_OnlyCSL(data, fileName):
    workbook = xw.Workbook(fileName)
    worksheet1 = workbook.add_worksheet("sheet1")
    worksheet1.activate()
    title = ['name', 'CLS_GT','CLS_pre',]
    # title = ['name', 'Risk',]
    worksheet1.write_row('A1', title)
    i = 2  # 从第二行开始写入数据
    for j in range(len(data)):
        insertData = [data[j]["name"], data[j]["CLS_GT"], data[j]["CLS_pre"]]
        # insertData = [data[j]["name"], data[j]["Risk"]]
        row = 'A' + str(i)
        worksheet1.write_row(row, insertData)
        i += 1
    workbook.close()


def xw_toExcel(data, fileName):
    workbook = xw.Workbook(fileName)
    worksheet1 = workbook.add_worksheet("sheet1")
    worksheet1.activate()
    title = ['name','risk', 'CLS_GT','CLS_pre','mean_dice','hd95','asd','iou']
    worksheet1.write_row('A1', title)
    i = 2
    for j in range(len(data)):
        insertData = [data[j]["name"],data[j]["risk"], data[j]["CLS_GT"], data[j]["CLS_pre"],
                     data[j]["mean_dice"], data[j]["hd95"],
                      data[j]["asd"], data[j]["iou"]]
        row = 'A' + str(i)
        worksheet1.write_row(row, insertData)
        i += 1
    workbook.close()


def clinical_embd(heart_dataset_R):
    # 2,3
    names = heart_dataset_R[heart_dataset_R.columns[2]]

    X = heart_dataset_R[heart_dataset_R.columns[3:-2]]


    categorical_indicator = [True, False, True, True, True, True, True, True, True, True, True, True, True, True, True,
                             True, True]

    # for pre_clinical_indicator
    # categorical_indicator = [False, True, True, True, True, True, True, True, True, True]

    y = heart_dataset_R[heart_dataset_R.columns[-2]]
    DFS = heart_dataset_R[heart_dataset_R.columns[-1]]
    categorical_columns = X.columns[list(np.where( np.array(categorical_indicator) == True)[0])].tolist()
    cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))

    cat_idxs = list(np.where(np.array(categorical_indicator) == True)[0])
    con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))

    for col in categorical_columns:
        X[col] = X[col].astype("object")


    temp = X.fillna("10000")
    nan_mask = temp.ne("10000").astype(int)

    cat_dims = []
    for col in categorical_columns:
        X[col] = X[col].fillna(2)
        l_enc = LabelEncoder()
        print(col)
        X[col] = l_enc.fit_transform(X[col].values)
        cat_dims.append(len(l_enc.classes_))
    for col in cont_columns:
        X[col].fillna("MissingValue",inplace=True)
        X.fillna(X[col].mean(), inplace=True)
    X = X.values
    nan_mask = nan_mask.values
    y = y.values
    l_enc = LabelEncoder()
    y = l_enc.fit_transform(y)


    train_mean, train_std = np.array(X[:,con_idxs], dtype=np.float32).mean(0), np.array(
       X[:,con_idxs], dtype=np.float32).std(0)
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)
    return cat_dims, cat_idxs, con_idxs,X,nan_mask,y,DFS.values, train_mean, train_std,names




def embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset=False):
    device = x_cont.device
    x_categ = x_categ + model.saint_process.categories_offset.type_as(x_categ)
    x_categ_enc = model.saint_process.embeds(x_categ)
    n1, n2 = x_cont.shape
    _, n3 = x_categ.shape
    if model.saint_process.cont_embeddings == 'MLP':
        x_cont_enc = torch.empty(n1, n2, model.saint_process.dim)
        for i in range(model.saint_process.num_continuous):
            x_cont_enc[:, i, :] = model.saint_process.simple_MLP[i](x_cont[:, i])
    else:
        raise Exception('This case should not work!')

    x_cont_enc = x_cont_enc.to(device)
    cat_mask_temp = cat_mask + model.saint_process.cat_mask_offset.type_as(cat_mask)
    con_mask_temp = con_mask + model.saint_process.con_mask_offset.type_as(con_mask)

    cat_mask_temp = model.saint_process.mask_embeds_cat(cat_mask_temp)
    con_mask_temp = model.saint_process.mask_embeds_cont(con_mask_temp)
    x_categ_enc[cat_mask == 0] = cat_mask_temp[cat_mask == 0]
    x_cont_enc[con_mask == 0] = con_mask_temp[con_mask == 0]

    if vision_dset:
        pos = np.tile(np.arange(x_categ.shape[-1]), (x_categ.shape[0], 1))
        pos = torch.from_numpy(pos).to(device)
        pos_enc = model.saint_process.pos_encodings(pos)
        x_categ_enc += pos_enc

    return x_categ, x_categ_enc, x_cont_enc




def generate_data(heart_dataset,start, end):
    dic_ = {}
    print("Keys of heartstd_dataset: \n{}".format(heart_dataset.keys()))
    df = heart_dataset

    x = df[df.columns[start:end-2]].values
    x = StandardScaler().fit(x).transform(x)
    dfs = df[df.columns[-1]].values
    dfstd = pd.DataFrame(x)

    features = list(set(dfstd.columns))
    y_bin_labels = []

    y_bin_labels.append('y' + str(0))
    dfstd['y' + str(0)] = df['censorship']

    y_bin_labels.append('y' + str(1))
    dfstd['y' + str(1)] = dfs

    data = dfstd['y0'][dfstd['y0'] > 0]
    uncensored_df = dfstd['y1'][data.index]

    disc_labels, q_bins = pd.qcut(uncensored_df, q=10, retbins=True, labels=False)
    q_bins[-1] = dfs.max() + 1e-5
    q_bins[0] = dfs.min() - 1e-5

    disc_labels, q_bins = pd.cut(dfs, bins=q_bins, retbins=True, labels=False, right=False,
                                 include_lowest=True)

    dfstd['y' + str(2)] = disc_labels
    y_bin_labels.append('y' + str(2))

    names = df['id_name'].values

    for item in range(len(names)):
        dic_[names[item]] = dfstd.values[item]
    return dic_, features, y_bin_labels

def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, img_information=None):
    if img_information:
        Direction = img_information[0]
        Origin = img_information[1]
        Spacing = img_information[2]
        name = img_information[3][0].strip().split('/')[-1]

    image = image.cuda()
    net.eval()
    with torch.no_grad():
        image = image.squeeze(0)
        image = image.permute(3,0,1,2)

        outputs = net(image)
        out= outputs.cpu().detach().numpy()

        out[out > 0.5] = np.float32(1)
        out[out < 0.5] = np.float32(0)

    label = label.squeeze(0)
    label = label.permute(3,0,1,2)
    label = label.squeeze(0).cpu().detach().numpy()
    dice_score = calculate_metric_percase(out, label)

    iou = IoU(out, label)

    dice_pre = dice_score[0]
    hd_95 = dice_score[1]
    asd_score = dice_score[2]

    if test_save_path is not None:
        path = test_save_path+'/Sample_test.txt'
        with open(path,'a',encoding='utf-8') as f:
            string ="{}: ||   Dice {:.4f} || HD95 {:.4f}".format(name,dice_pre,hd_95)
            f.write(string+'\n')

        prediction = out.squeeze(1)
        prediction = np.transpose(prediction, (0, 2, 1))

        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        prd_itk = resize_image_itk(prd_itk , [256,256] ,interpolator = 'linear' )

        prd_itk.SetDirection([float(Direction[0][0]),float(Direction[1][0]),float(Direction[2][0]),
                             float(Direction[3][0]),float(Direction[4][0]),float(Direction[5][0]),
                             float(Direction[6][0]),float(Direction[7][0]),float(Direction[8][0])])
        prd_itk.SetOrigin([float(Origin[0][0]),float(Origin[1][0]),float(Origin[2][0])])
        prd_itk.SetSpacing([float(Spacing[0][0]),float(Spacing[1][0]),float(Spacing[2][0])])


        label = label.squeeze(1)
        label = np.transpose(label, (0, 2, 1))


        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        lab_itk = resize_image_itk(lab_itk , [256,256] ,interpolator = 'linear' )


        lab_itk.SetDirection([float(Direction[0][0]),float(Direction[1][0]),float(Direction[2][0]),
                             float(Direction[3][0]),float(Direction[4][0]),float(Direction[5][0]),
                             float(Direction[6][0]),float(Direction[7][0]),float(Direction[8][0])])
        lab_itk.SetOrigin([float(Origin[0][0]),float(Origin[1][0]),float(Origin[2][0])])
        lab_itk.SetSpacing([float(Spacing[0][0]),float(Spacing[1][0]),float(Spacing[2][0])])

        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+case + "_gt.nii.gz")

    return dice_score,hd_95,iou,asd_score,0,0,0,0

