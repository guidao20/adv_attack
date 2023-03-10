import os
import torch
import torchvision.transforms as T
import torch.nn as nn
import argparse
import torchvision
from torch.utils.data import Dataset
import csv
import PIL.Image as Image
from torch.backends import cudnn
import numpy as np
import pretrainedmodels
import ssl
import datetime
from attack_method import *
from config import *
import sys
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CUDA_VISIBLE_DEVICES']= sys.argv[2]

# Selected imagenet. The .csv file format:
# class_index, class, image_name
# 0,n01440764,ILSVRC2012_val_00002138.JPEG
# 2,n01484850,ILSVRC2012_val_00004329.JPEG
# ...
class SelectedImagenet(Dataset):
    def __init__(self, imagenet_val_dir, selected_images_csv, transform=None):
        super(SelectedImagenet, self).__init__()
        self.imagenet_val_dir = imagenet_val_dir
        self.selected_images_csv = selected_images_csv
        self.transform = transform
        self._load_csv()
    def _load_csv(self):
        reader = csv.reader(open(self.selected_images_csv, 'r'))
        next(reader)
        self.selected_list = list(reader)
    def __getitem__(self, item):
        image_name, target  = self.selected_list[item]
        image = Image.open(os.path.join(self.imagenet_val_dir, image_name))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, int(target) - 1
    def __len__(self):
        return len(self.selected_list)


if __name__ == '__main__':
    starttime = datetime.datetime.now()
    cudnn.benchmark = False

    attack_name = sys.argv[1]

    attack_type, model_name, target_attack, batch_size, save_dir  = get_config_info(attack_name)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    if model_name == 'squeezenet':
            model = torchvision.models.squeezenet1_0(pretrained=True)
    elif model_name == 'vgg':
            model = torchvision.models.vgg19(pretrained=True)
    elif model_name == 'inception':
            model = torchvision.models.inception_v3(pretrained=True)
    elif model_name == 'senet':
            model =  pretrainedmodels.__dict__['senet154'](num_classes=1000, pretrained='imagenet')
    elif model_name == 'resnet':
            model = torchvision.models.resnet50(pretrained=True)
    elif model_name == 'densenet':
            model = torchvision.models.densenet121(pretrained=True)
    elif model_name == 'inceptionv4':
            model =  pretrainedmodels.__dict__['inceptionv4'](num_classes=1000, pretrained='imagenet')
    elif model_name == 'inceptionv3':
            model =  pretrainedmodels.__dict__['inceptionv3'](num_classes=1000, pretrained='imagenet')
    elif model_name == 'inceptionresv2':
            model =  pretrainedmodels.__dict__['inceptionresnetv2'](num_classes=1000, pretrained='imagenet')
    elif model_name == 'resnet152':
            model = pretrainedmodels.__dict__['resnet152'](num_classes=1000, pretrained='imagenet')
    else:
            print('No implemation')




    if model_name not in ['inception', 'inceptionv4', 'inceptionresv2', 'inceptionv3']:
                mean = (0.485, 0.456, 0.406)
                std = (0.229, 0.224, 0.225)
                input_size = [3, 224, 224]
    else:
                mean = (0.5, 0.5, 0.5)
                std = (0.5, 0.5, 0.5)
                input_size = [3, 299, 299]

    
    channel1_low_bound = (0 - mean[0]) / std[0]   
    channel1_hign_bound = (1 - mean[0]) / std[0]   
    channel2_low_bound = (0 - mean[1]) / std[1]   
    channel2_hign_bound = (1 - mean[1]) / std[1]   
    channel3_low_bound = (0 - mean[2]) / std[2]   
    channel3_hign_bound = (1 - mean[2]) / std[2]  
    
    bound = [channel1_low_bound, channel1_hign_bound, channel2_low_bound, channel2_hign_bound, channel3_low_bound, channel3_hign_bound]

    norm = T.Normalize(tuple(mean), tuple(std))
    resize = T.Resize(tuple((input_size[1:])))

    print(std)

    trans = T.Compose([
            resize,
            T.ToTensor(),
            norm
            ])



    dataset = SelectedImagenet(imagenet_val_dir='data/imagenet/ILSVRC2012_img_val',
                               selected_images_csv='data/imagenet/val_rs.csv',
                               transform=trans
                               )

    ori_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers = 0, pin_memory = False)


    model.eval()
    model.to(device)
    if target_attack:
        label_switch = torch.tensor(list(range(500,1000))+list(range(0,500))).long()
    label_ls = []
    correct = 0


    for ind, (ori_img, label)in enumerate(ori_loader):
        label_ls.append(label)
        if target_attack:
            label = label_switch[label]
        ori_img = ori_img.to(device)
        img = ori_img.clone()
        label = label.to(device)
        img_adv = attack_type.attack(model,img,label)
        predict_adv = model(img_adv)
        predicted_adv = torch.max(predict_adv.data, 1)[1]
        print(label, predicted_adv)
        correct += (predicted_adv == label).sum().cpu().numpy()
        print(correct)
        print('successful')
        img_adv[:,0,:,:] = img_adv[:,0,:,:] * std[0] + mean[0]
        img_adv[:,1,:,:] = img_adv[:,1,:,:] * std[1] + mean[1]
        img_adv[:,2,:,:] = img_adv[:,2,:,:] * std[2] + mean[2]

        np.save(save_dir + '/batch_{}.npy'.format(ind), torch.round(img_adv.data*255).cpu().numpy().astype(np.uint8()))

        del img, ori_img, img_adv
        print('batch_{}.npy saved'.format(ind))
        ### end
    label_ls = torch.cat(label_ls)
    np.save(save_dir + '/labels.npy', label_ls.numpy())
    print('images saved')
    endtime = datetime.datetime.now()
    cost_time = (endtime - starttime).seconds
    print('Computation time: ', cost_time)
