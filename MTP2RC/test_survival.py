import argparse
import logging
import random
import sys
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.data_process import *
from datasets.dataset_synapse import ImageDataset
from sksurv.metrics import concordance_index_censored
from networks.unet3D_M_transformer.unet_model import UNet
from utils import *
import pandas as pd



import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='D:/project/Relabeled_data/2D_slices_256/Relabeled_test.txt', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')

parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=30, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=10,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=128, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true",default=True, help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,   default='R50+ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='./predictions007', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0002, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()

def inference_slices(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="Relabeled_test", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=8, shuffle=False, num_workers=0)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    acc_total = 0
    dice_epual_0 = []
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name']



        metric_i,acc = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)

        metric_list += np.array(metric_i)
        acc_total += acc
        each_vol_dice = metric_list

        logging.info('idx %d || case %s || mean_dice %f || acc %f' % (i_batch, case_name, metric_i,acc))
        if float(each_vol_dice) < 0.0001:
            dice_epual_0.append(case_name)
    metric_list = metric_list / i_batch
    acc_total = acc_total/ i_batch
    for i in range(0, args.num_classes):
        logging.info('Mean class %d mean_dice %f acc  %f' % (i, metric_list,acc_total))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    print('total manmber : {} || The detail name : {}'.format(len(dice_epual_0),dice_epual_0))
    return "Testing Finished!"


def cam_generate(case_name,t2,cams,index):
    t2_np = sitk.GetArrayFromImage(t2)

    cams = cams.cpu().detach().numpy()
    cams = np.sum(cams, axis=1).squeeze()
    cam_img = (cams - cams.min()) / (cams.max() - cams.min())  # Normalize

    cam_img_mean = cam_img.transpose((2,0,1))
    cam_img_mean = cam_img_mean
    import cv2 as cv

    for slice_id in range(cam_img_mean.shape[0]):
        plt.figure()
        mix_img = (255 * cam_img_mean[slice_id])
        mix_img = np.uint8((mix_img / mix_img.max()) * 255).squeeze()

        mix_img = cv.applyColorMap(np.uint8(mix_img), cv.COLORMAP_JET)
        mix_img = cv.cvtColor(mix_img, cv.COLOR_BGR2RGB)



        mix_img = mix_img.transpose((1, 0, 2)).transpose((1, 0, 2)).transpose((1, 0, 2))

        plt.imshow(t2_np[slice_id])
        plt.imshow(mix_img, alpha=0.4, cmap='rainbow')
        # plt.imshow()
        plt.axis('off')

        f = plt.gcf()
        save_path = '_08_mulitiTask/' + case_name
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_path = save_path + '/slice' + index+'_'+str(slice_id + 1) + '.jpg'
        f.savefig(save_path)
        f.clear()
        plt.close()


def inference_volume(args, test_save_path=None):

    test_path = args.volume_path

    heart_dataset_R = pd.read_excel('',
                                    sheet_name='')

    cat_dims, cat_idxs, con_idxs, X, nan_mask, y, DFS, train_mean, train_std,names = clinical_embd(heart_dataset_R)
    cat_dims = np.append(np.array([1]), np.array(cat_dims)).astype(int)

    model = UNet(cat_dims=cat_dims).cuda()
    snapshot = ''

    model.load_state_dict(torch.load(snapshot))



    train_list = create_list(test_path)
    db_test = ImageDataset(data_list = train_list,transform = [Resize3D([128,128],check=True)],clinical = [X,nan_mask,y,DFS,cat_idxs,train_mean, train_std,names])

    testloader = DataLoader(db_test, batch_size=1, shuffle=False)


    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    hd_list = 0.0
    iou_list = 0.
    asd_list = 0.

    acc_total = 0
    dice_epual_0 = []

    txt = test_save_path+'/_details.txt'

    all_event_times = []
    all_censorships = []
    pre_censorship_lst = []
    all_risk_scores = []

    detailed_infm = []
    with open(txt,'a',encoding='utf-8') as f:
        for i_batch, (data, label, infm) in enumerate(testloader):
            data_img = Variable(
                data[0].cuda())  # A Variable wraps a Tensor. It supports nearly all the API’s defined by a Tensor.
            data_clinical = data[1]  # A Variable wraps a Tensor. It supports nearly all the API’s defined by a Tensor.
            test_label_seg = Variable(label[0].cuda())
            Direction, Origin, Spacing,name,Size = infm
            x_categ, x_cont, test_label_censor, test_label_dfs, cat_mask, con_mask = data_clinical[0].cuda(), data_clinical[1].cuda(), \
                                                                     data_clinical[2].cuda().unsqueeze(-1), \
                                                                 data_clinical[3].cuda(), data_clinical[4].cuda().long(), data_clinical[5].cuda()

            __, x_categ_enc, x_cont_enc= embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, False)
            case_name = infm[-2][0].strip().split('/')[-1]

            outputs, pre_censorship,risk = model(data_img, x_categ_enc, x_cont_enc, True)

            test_label_seg = test_label_seg.cpu().detach().numpy()

            outputs = outputs.cpu().detach().numpy()
            outputs[outputs > 0.5] = 1
            outputs[outputs < 0.5] = 0

            dice_score = calculate_metric_percase(outputs, test_label_seg)
            dice_pre = dice_score[0]
            hd95 = dice_score[1]
            asd = dice_score[2]
            iou = IoU(outputs, test_label_seg)

            print(case_name,'||',test_label_censor.item(),'||',pre_censorship.item())
            subject_infm = {'name': case_name,'risk':risk.cpu().detach().numpy(), 'CLS_GT': test_label_censor.cpu().detach().numpy(),
                            'CLS_pre': pre_censorship.cpu().detach().numpy(), 'mean_dice': dice_pre,
                            'hd95': hd95, 'asd': asd,
                            'iou': iou}
            detailed_infm.append(subject_infm)


            pre_censorship[pre_censorship>=0.5]=1
            pre_censorship[pre_censorship<0.5]=0

            all_risk_scores.append(risk.cpu().detach().numpy())
            all_event_times.append(test_label_dfs.cpu().detach().numpy())
            all_censorships.append(test_label_censor.cpu().detach().numpy())
            pre_censorship_lst.append(pre_censorship.cpu().detach().numpy())


            metric_list += np.array(dice_pre)
            hd_list += np.array(hd95)
            asd_list += np.array(asd)
            iou_list += np.array(iou)

            info = 'idx %d || case %s || mean_dice %3f || hd95 %3f || asd  %3f || iou  %3f' % (i_batch, case_name, dice_pre, hd95,asd,iou)
            print(info)




    metric_list = metric_list / len(testloader)
    hd_list = hd_list / len(testloader)

    asd_list = asd_list / len(testloader)
    iou_list = iou_list / len(testloader)




    all_risk_scores = np.concatenate(
        (np.stack(all_risk_scores[:-1], axis=1).reshape(-1), (all_risk_scores[-1]).reshape(-1)), axis=0)
    all_event_times = np.concatenate(
        (np.stack(all_event_times[:-1], axis=1).reshape(-1), (all_event_times[-1]).reshape(-1)), axis=0)
    all_censorships = np.concatenate(
        (np.stack(all_censorships[:-1], axis=1).reshape(-1), (all_censorships[-1]).reshape(-1)), axis=0)
    pre_censorship_lst = np.concatenate(
        (np.stack(pre_censorship_lst[:-1], axis=1).reshape(-1), (pre_censorship_lst[-1].reshape(-1))), axis=0)

    censor_auc, recall, precision, accuracy, jaccard = classification_evalution(pre_censorship_lst, all_censorships)


    c_index = concordance_index_censored(all_censorships.astype(bool), all_event_times, all_risk_scores)

    print('Survival: c_index {:.3f} '.format(c_index[0]))
    print('classification: accuracy {:.3f}, censor_auc {:.3f}, recall {:.3f}, precision {:.3f}, jaccard{:.3f}'.format(accuracy,censor_auc, recall, precision,jaccard))
    final_infm = 'Segmentation: Final results: mean_dice %3f  hd95  %3f  || asd  %3f  || iou  %3f  ' % (metric_list, hd_list,asd_list,iou_list)
    logging.info(final_infm)
    # xw_toExcel(detailed_infm, 'new_risk_res/img_PreClinical_external2.xls')
    with open(txt, 'a', encoding='utf-8') as f:
        f.write(final_infm+'\n')

    print('total manmber : {} || The detail name : {}  || '.format(len(dice_epual_0), dice_epual_0))
    return "Testing Finished!"


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    dataset_config = {
        'Synapse': {
            'Dataset': ImageDataset,
            'volume_path': '',
            'list_dir':'',
            'num_classes': 1,
            'z_spacing': 1,
        },
    }



    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    # name the same snapshot defined in train script!
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "./model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name


    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = 'seg_results/_02_multi_task'
        test_save_path = os.path.join(args.test_save_dir, 'imgs')

        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference_volume(args, test_save_path)


