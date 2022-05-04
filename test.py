import torch
import numpy as np
import math
from glob import glob
from VNet import VNet
#from medpy.metric import binary
#from dataset import read_h5
from transforms import RandomCrop, CenterCrop, RandomRotFlip, ToTensor
from dataset import LAHeart
from torchvision import transforms
from tqdm import tqdm
import h5py
import torch.nn.functional as F
from medpy import metric
import nibabel as nib
import os


def test_all_case(net, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True, test_save_path=None, preproc_fn=None):
    total_metric = 0.0
    for image_path in tqdm(image_list):
        id = image_path.split('/')[-2]
        #print('id:', id)
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)

        if np.sum(prediction)==0:
            single_metric = (0,0,0,0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
        total_metric += np.asarray(single_metric)
        print('single_metric:', single_metric)
        
        if save_result:
            if not os.path.exists(test_save_path + id + '/'):
                os.makedirs(test_save_path + id + '/')
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path + id + '/' + id + "_pred.nii.gz")
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + id + '/' + id + "_img.nii.gz")
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + id + '/' + id + "_gt.nii.gz")
    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))

    return avg_metric

def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                y1 = net(test_patch)
                y = F.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0,:,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = np.argmax(score_map, axis = 0)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map

def calculate_metric_percase(pred, gt):
    print('pred:', pred.shape)
    print('gt:', gt.shape)
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd

if __name__ == '__main__':
    base_dir = 'drive/MyDrive/ELEC4010N_project/data/'
    save_model_path = 'drive/MyDrive/ELEC4010N_project/saved_models/saved_174.pth'
    test_save_path = './predictions/'
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
    with open(base_dir+'test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [base_dir+"2018LA_Seg_Training Set/"+item.replace('\n','')+"/mri_norm2.h5" for item in image_list]
    
    model =  VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=False).cuda() # load your model here
    model.load_state_dict(torch.load(save_model_path)) #Load back saved model
    print("init weight from {}".format(save_model_path))
    model.eval()
    avg_metric = test_all_case(model, image_list, num_classes=2,
                            patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                            save_result=True, test_save_path=test_save_path)