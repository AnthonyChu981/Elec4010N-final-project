from dataset import LAHeart
from torch.utils.data import DataLoader
from torchvision import transforms
from transforms import RandomCrop, CenterCrop, RandomRotFlip, ToTensor
from VNet import VNet
import torch
from torch import nn
from tqdm import tqdm
#from tensorboardX import SummaryWriter
import torch.nn.functional as F
#from medpy.metric import binary
import os
import shutil
import logging
import sys
from losses import dice_loss
import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':
    log_path = "./run/"  

    use_cuda = True
    #print(torch.cuda.is_available())
    max_epoch = 200
    train_data_path = 'drive/MyDrive/ELEC4010N_project/data/'
    crop_size = (112, 112, 80)
    batch_size = 4
    transform = transforms.Compose([
                    RandomRotFlip(),
                    RandomCrop(crop_size),
                    ToTensor()
                    ])

    #model = UNet(in_channel=1, out_channel=2, training=True)
    model = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True)

    if use_cuda:
        model = model.cuda()
    train_dst = LAHeart(base_dir=train_data_path,
                       split='train',
                       num=16,
                       transform = transform
                       )

    train_loader = DataLoader(
        train_dst, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4, 
        pin_memory=True
    )

    # loss function
    criterion = nn.CrossEntropyLoss()

    base_lr = 0.01
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    #writer = SummaryWriter(log_path+'/log')
    logging.info("{} itertations per epoch".format(len(train_loader)))

    loss_list = []
    acc_train_list = []
    acc_test_list = []

    lr_ = base_lr
    model.train()
    best_loss = 0.0
    for epoch in range(max_epoch):
        running_loss = 0.0
        running_correct = 0
        print(" -- Epoch {}/{}".format(epoch + 1, max_epoch))
        model.train()
        for batch in tqdm(train_loader):
            # set all gradients to zero
            optimizer.zero_grad()

            # fetch data
            images, labels = batch['image'], batch['label']
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()
            
            # model forward
            outputs = model(images)

            # calculate loss
            #loss = criterion(outputs, labels)
            loss_seg = F.cross_entropy(outputs, labels)
            outputs_soft = F.softmax(outputs, dim=1)
            loss_seg_dice = dice_loss(outputs_soft[:, 1, :, :, :], labels == 1)
            loss = 0.5*(loss_seg+loss_seg_dice) 

            # backward and optimize parameters
            loss.backward()
            optimizer.step()

            #pred = F.softmax(outputs, dim=1)
            running_loss += loss.item()
            #running_correct += torch.sum(pred == labels)
            
        if epoch % 40 == 0:
            lr_ = base_lr * 0.1 ** (epoch // 40) 
        # record loss, accuracy
        #loss = running_loss / len(train_dst)
        #writer.add_scalar('training loss', loss, epoch+1)
        loss_list.append(running_loss)
        #acc_train = running_correct / len(train_dst)
        #acc_train_list.append(acc_train.item())
        print('loss:', running_loss)
        #save_mode_path = os.path.join(log_path, 'iter_'+str(max_epoch+1)+'.pth')
        if epoch == 0:
            best_loss = running_loss
        if running_loss <= best_loss:
            torch.save(model.state_dict(), 'saved_{}.pth'.format(epoch+1))
            best_loss = running_loss

    print('loss:', loss_list)
    #print('train_acc:', acc_train_list)
    #writer.close()
    
    x_axis = list(range(1, max_epoch+1))
    fig = plt.figure()
    plt.plot(x_axis, loss_list, '-')
    plt.xlabel('Epochs')
    plt.ylabel('Training loss')
    plt.show()