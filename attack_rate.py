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



if torch.cuda.is_available():
        device = torch.device('cuda')
else:
        device = torch.device('cpu')

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
        target, target_name, image_name = self.selected_list[item]
        image = Image.open(os.path.join(self.imagenet_val_dir, image_name))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, int(target) - 1
    def __len__(self):
        return len(self.selected_list)



class AdvImagenet(Dataset):
    def __init__(self, imagepath, labelpath, transform=None):
        super(AdvImagenet, self).__init__()
        self.imagepath = imagepath
        self.labelpath = labelpath
        self.transform = transform
        self._load_data()
    def _load_data(self):
        self.images = np.load(self.imagepath)
        self.labels = np.load(self.labelpath)
    def __getitem__(self, item):
        image = Image.fromarray(self.images[item].transpose(1,2,0))
        label = self.labels[item]
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    def __len__(self):
        return len(self.images)



def get_model(model_name):
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
        elif model_name == 'inception_v4':
                model =  pretrainedmodels.__dict__['inceptionv4'](num_classes=1000, pretrained='imagenet')
        elif model_name == 'inception_Resv2':
                model =  pretrainedmodels.__dict__['inceptionresnetv2'](num_classes=1000, pretrained='imagenet')
        elif model_name == 'resnet152':
                model = pretrainedmodels.__dict__['resnet152'](num_classes=1000, pretrained='imagenet')
        else:
                print('No implemation')
        return model



batch_size = 4

model_resnet152 = get_model('resnet152')
model_inceptionv3 = get_model('inception')
model_inceptionv4 = get_model('inception_v4')
model_inception_resv2 = get_model('inception_Resv2')


model_resnet152.eval()
model_resnet152.to(device)
model_inceptionv3.eval()
model_inceptionv3.to(device)
model_inceptionv4.eval()
model_inceptionv4.to(device)
model_inception_resv2.eval()
model_inception_resv2.to(device)




#if model_name in ['squeezenet', 'resnet', 'densenet', 'senet' ,'vgg', 'resnet152']:
#        input_size = [3, 224, 224]
#        mean = (0.485, 0.456, 0.406)
#        std = (0.229, 0.224, 0.225)
#else:
#        input_size = [3, 299, 299]
#        mean = (0.5, 0.5, 0.5)
#        std = (0.5, 0.5, 0.5)

input_size1 = [3, 299, 299]
mean1 = (0.5, 0.5, 0.5)
std1 = (0.5, 0.5, 0.5)

norm1 = T.Normalize(tuple(mean1), tuple(std1))
resize1 = T.Resize(tuple((input_size1[1:])))


trans_inc = T.Compose([
            resize1,
            T.ToTensor(),
            norm1
    ])




correct_inceptionv3 = 0
correct_inceptionv4 = 0
correct_inception_res = 0

advpath = 'DIM.npy'
imagepath = os.path.join('result_resnet152', advpath)
labelpath = os.path.join('result_resnet152', 'labels.npy')

advdataset1 = AdvImagenet(imagepath = imagepath, labelpath = labelpath, transform = trans_inc)
adv_loader1 = torch.utils.data.DataLoader(advdataset1, batch_size=batch_size, shuffle=False, num_workers = 0, pin_memory = False)

for ind, (adv_img , label) in enumerate(adv_loader1):
        adv_img = adv_img.to(device)
        label = label.to(device)
        predict_inceptionv3 = model_inceptionv3(adv_img)
        predict_inceptionv4 = model_inceptionv4(adv_img)
        predict_inception_resv2 = model_inception_resv2(adv_img)
        predicted_inceptionv3 = torch.max(predict_inceptionv3.data, 1)[1]
        predicted_inceptionv4 = torch.max(predict_inceptionv4.data, 1)[1]
        predicted_inception_resv2 = torch.max(predict_inception_resv2.data, 1)[1]
        correct_inceptionv3 += (predicted_inceptionv3 == label).sum()
        correct_inceptionv4 += (predicted_inceptionv4 == label).sum()
        correct_inception_res += (predicted_inception_resv2 == label).sum()                
        print(correct_inceptionv3.item(), correct_inceptionv4.item(), correct_inception_res.item())  



input_size2 = [3, 224, 224]
mean2 = (0.485, 0.456, 0.406)
std2 = (0.229, 0.224, 0.225)


norm2 = T.Normalize(tuple(mean2), tuple(std2))
resize2 = T.Resize(tuple((input_size2[1:])))


trans_res = T.Compose([
            resize2,
            T.ToTensor(),
            norm2
    ])




advdataset2 = AdvImagenet(imagepath = imagepath, labelpath = labelpath, transform = trans_res)
adv_loader2 = torch.utils.data.DataLoader(advdataset2, batch_size=batch_size, shuffle=False, num_workers = 0, pin_memory = False)

correct_resnet152 = 0
for ind, (adv_img , label) in enumerate(adv_loader2):
        adv_img = adv_img.to(device)
        label = label.to(device)
        predict_resnet152 = model_resnet152(adv_img)
        predicted_resnet152 = torch.max(predict_resnet152.data, 1)[1]
        correct_resnet152 += (predicted_resnet152 == label).sum()
        print(correct_resnet152.item())


asc_resnet152 = (1000 - correct_resnet152.item()) / 10.0
asc_inceptionv3 = (1000 - correct_inceptionv3.item()) / 10.0
asc_inceptionv4 = (1000 - correct_inceptionv4.item()) / 10.0
asc_inception_res = (1000 - correct_inception_res.item()) / 10.0

print('Attack Success Rate for resnet152: {}%'.format(asc_resnet152))
print('Attack Success Rate for inceptionv3: {}%'.format(asc_inceptionv3))
print('Attack Success Rate for inceptionv4: {}%'.format(asc_inceptionv4))
print('Attack Success Rate for inception_resnetv2: {}%'.format(asc_inception_res))




