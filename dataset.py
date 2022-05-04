import torch
import tensorflow as tf
import h5py
from torch.utils.data import Dataset
from torchvision import transforms
from transforms import RandomCrop, CenterCrop, RandomRotFlip, ToTensor
from torch.utils.data import DataLoader
import numpy as np

class LAHeart(Dataset):
    """ LA Dataset """
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        if split=='train':
            with open(self._base_dir+'train.list', 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(self._base_dir+'test.list', 'r') as f:
                self.image_list = f.readlines()
        self.image_list = [item.replace('\n','') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir+"2018LA_Seg_Training Set/"+image_name+"/mri_norm2.h5", 'r')
        #print(self._base_dir+"2018LA_Seg_Training Set/"+image_name+"/mri_norm2.h5")
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample

'''
train_data_path = './data/'
patch_size = (112, 112, 80)
batch_size = 4

if __name__ == '__main__':
    db_train = LAHeart(base_dir=train_data_path, 
                split='train',
                transform = transforms.Compose([
                            RandomRotFlip(),
                            RandomCrop(patch_size),
                            ToTensor(),
                            ]))
    for i in range(len(db_train)):
        get_item = LAHeart.__getitem__(db_train, 0)
        get_item['label'] = get_item['label'].numpy()
        zeros = np.count_nonzero(get_item['label'])
        #print(zeros)
        if zeros != 0:
            print(i)
'''