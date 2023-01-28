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
import torch.nn.functional as F
import datetime
import cv2
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
ssl._create_default_https_context = ssl._create_unverified_context



class SIM(object):
    def __init__(self, epsilon, T, mu, m, device, bound):
        self.epsilon = epsilon
        self.mu = mu
        self.T = T
        self.m = m
        self.device = device
        self.bound = bound

    def attack(self, model, ori_img, labels):
        x_adv = ori_img.detach()
        g_t = torch.zeros_like(ori_img)
        loss_fn = nn.CrossEntropyLoss()
        alpha = self.epsilon / self.T
        x_ori = ori_img.detach()
        for t in range(self.T):
            g = torch.zeros_like(x_adv)
            # x_nes = x_adv + alpha * self.mu * g_t
            x_nes = x_adv.detach()
            for i in range(self.m):
                x_nes.requires_grad = True
                x_temp = (x_nes / (2**i))
                #x_temp.requires_grad = True # M1: x_temp_grad
                outputs_temp = model(x_temp)
                loss_temp = loss_fn(outputs_temp, labels)
                loss_temp.backward()
                #g += x_temp.grad.detach()
                g += x_nes.grad.detach()
                x_nes = x_nes.detach()
            g = g / self.m
            g_t = self.mu * g_t + g / torch.norm(g, p=1, dim=(1,2,3), keepdim = True)

            x_adv = x_adv + alpha * torch.sign(g_t)
            x_adv = torch.where(x_adv > ori_img + self.epsilon, ori_img + self.epsilon, x_adv)
            x_adv = torch.where(x_adv < ori_img - self.epsilon, ori_img - self.epsilon, x_adv)
            x_adv[:,0,:,:]= torch.clamp(x_adv[:,0,:,:], self.bound[0], self.bound[1])
            x_adv[:,1,:,:]= torch.clamp(x_adv[:,1,:,:], self.bound[2], self.bound[3])
            x_adv[:,2,:,:]= torch.clamp(x_adv[:,2,:,:], self.bound[4], self.bound[5])
            x_adv = x_adv.detach()
        return x_adv


class TAIGO(object):
    def __init__(self, epsilon, steps, T_niters, gamma, mu,  bound, device):
        self.epsilon = epsilon
        self.steps = steps
        self.T_niters = T_niters
        self.device = device
        self.mu = mu
        self.gamma = gamma
        self.bound = bound


    def attack(self, model, ori_img, label):
        x_adv = ori_img.to(self.device)
        baseline = torch.zeros_like(ori_img).to(self.device)
        g_t = torch.zeros_like(ori_img).to(self.device)
        alpha = self.epsilon / self.T_niters
        for t in range(self.T_niters):
            x_temp = x_adv.detach()
            gradients = []
            for i in range(0, self.steps+1):
                    img_tmp = baseline + (float(i) / self.steps) * (x_temp - baseline)
                    img_tmp.requires_grad_(True)
                    output = model(img_tmp)
                    output_label = output.gather(1, label.view(-1,1)).squeeze(1)
                    grad_first = torch.autograd.grad(output_label, img_tmp, grad_outputs = torch.ones_like(output_label), retain_graph = True, create_graph=True)
                    gradients.append(grad_first[0])
            Intergrated_Attenton = (sum(gradients) / len(gradients)).detach()
            x_adv.requires_grad_(True)
            attribution = x_adv * Intergrated_Attenton
            blank = torch.zeros_like(attribution)
            positive = torch.where(attribution >= 0, attribution, blank)
            negative = torch.where(attribution < 0,  attribution, blank)
            balance_attribution = positive + self.gamma * negative
            loss = torch.mean(torch.sum(balance_attribution, dim = [1, 2, 3]))
            loss.backward()
            grad = x_adv.grad

            g_t = self.mu * g_t + grad / torch.norm(grad, p=1, dim = [1,2,3], keepdim = True)
            x_adv = x_adv - alpha * torch.sign(g_t)

            x_adv = torch.where(x_adv > ori_img + self.epsilon, ori_img + self.epsilon, x_adv)
            x_adv = torch.where(x_adv < ori_img - self.epsilon, ori_img - self.epsilon, x_adv)

            x_adv[:,0,:,:]= torch.clamp(x_adv[:,0,:,:], self.bound[0], self.bound[1])
            x_adv[:,1,:,:]= torch.clamp(x_adv[:,1,:,:], self.bound[2], self.bound[3])
            x_adv[:,2,:,:]= torch.clamp(x_adv[:,2,:,:], self.bound[4], self.bound[5])
            x_adv = x_adv.detach()
        return x_adv


class VMI(object):
    def __init__(self, epsilon,  beta, N, T, mu,device, bound):
        self.epsilon = epsilon
        self.beta = beta
        self.mu = mu
        self.N = N
        self.T = T
        self.device = device
        self.bound = bound

    def attack(self, model, images, lables):
        g = torch.zeros_like(images)
        v = torch.zeros_like(images)
        x_adv = images.detach()
        loss_fn = nn.CrossEntropyLoss()
        alpha = self.epsilon / self.T

        for i in range(self.T):
            x_adv.requires_grad = True
            outputs = model(x_adv)
            loss = loss_fn(outputs, lables)
            loss.backward()
            g_prime = x_adv.grad.detach()
            g = self.mu * g + (g_prime + v) / torch.norm(g_prime + v, p = 1, dim = (1, 2, 3), keepdim = True)
            grad_temp = torch.zeros_like(x_adv)
            for k in range(self.N):
                x_temp = x_adv.detach() + (torch.rand(x_adv.shape).to(self.device) - 0.5) * 2 * self.beta * self.epsilon
                x_temp.requires_grad = True
                output_temp = model(x_temp)
                loss_temp  = loss_fn(output_temp, lables)
                loss_temp.backward()
                grad_temp += x_temp.grad.detach()
            v = grad_temp / self.N  - g_prime
            x_adv = x_adv +  torch.clamp(alpha * torch.sign(g), - self.epsilon, self.epsilon)
            x_adv[:,0,:,:]= torch.clamp(x_adv[:,0,:,:], self.bound[0], self.bound[1])
            x_adv[:,1,:,:]= torch.clamp(x_adv[:,1,:,:], self.bound[2], self.bound[3])
            x_adv[:,2,:,:]= torch.clamp(x_adv[:,2,:,:], self.bound[4], self.bound[5])
            #x_adv = torch.clamp(x_adv + self.alpha * torch.sign(g), 0, 1)
            x_adv = x_adv.detach()
        return x_adv


class TAIGOR(object):
    def __init__(self, epsilon, steps, T_niters, gamma, mu,  bound, device):
        self.epsilon = epsilon
        self.steps = steps
        self.T_niters = T_niters
        self.device = device
        self.mu = mu
        self.gamma = gamma
        self.bound = bound


    def attack(self, model, ori_img, label):
        x_adv = ori_img.to(self.device)
        baseline = torch.zeros_like(ori_img).to(self.device)
        g_t = torch.zeros_like(ori_img).to(self.device)
        alpha = self.epsilon / self.T_niters
        for t in range(self.T_niters):
            x_temp = x_adv.detach()
            gradients = []
            for i in range(0, self.steps+1):
                if i==0 or i == self.steps:
                    random_noise = (torch.zeros_like(x_temp)).to(self.device)
                else:
                    random_noise = ((torch.rand(x_temp.shape) - 0.5) * 2 * self.epsilon).to(self.device)
                img_tmp = baseline + (float(i) / self.steps) * (x_temp - baseline) + random_noise
                img_tmp.requires_grad_(True)
                output = model(img_tmp)
                output_label = output.gather(1, label.view(-1,1)).squeeze(1)
                grad_first = torch.autograd.grad(output_label, img_tmp, grad_outputs = torch.ones_like(output_label), retain_graph = True, create_graph=True)
                gradients.append(grad_first[0])

            Intergrated_Attenton = (sum(gradients) / len(gradients)).detach()
            x_adv.requires_grad_(True)
            attribution = x_adv * Intergrated_Attenton
            blank = torch.zeros_like(attribution)
            positive = torch.where(attribution >= 0, attribution, blank)
            negative = torch.where(attribution < 0,  attribution, blank)
            balance_attribution = positive + self.gamma * negative
            loss = torch.mean(torch.sum(balance_attribution, dim = [1, 2, 3]))
            loss.backward()
            grad = x_adv.grad

            g_t = self.mu * g_t + grad / torch.norm(grad, p=1, dim = [1,2,3], keepdim = True)
            x_adv = x_adv - alpha * torch.sign(g_t)

            x_adv = torch.where(x_adv > ori_img + self.epsilon, ori_img + self.epsilon, x_adv)
            x_adv = torch.where(x_adv < ori_img - self.epsilon, ori_img - self.epsilon, x_adv)

            x_adv[:,0,:,:]= torch.clamp(x_adv[:,0,:,:], self.bound[0], self.bound[1])
            x_adv[:,1,:,:]= torch.clamp(x_adv[:,1,:,:], self.bound[2], self.bound[3])
            x_adv[:,2,:,:]= torch.clamp(x_adv[:,2,:,:], self.bound[4], self.bound[5])
            x_adv = x_adv.detach()
        return x_adv


class AoA(object):
    def __init__(self, epsilon, eta, lambda_, T, device, bound):
        self.epsilon = epsilon
        self.eta = eta
        self.lambda_ = lambda_
        self.T = T
        self.device = device
        self.bound = bound

    def attack(self, model, inputs, labels):
        x_ori = inputs.detach()
        x_adv = inputs.detach()
        x_shape = x_ori.shape
        # N = float(x_shape[0] * x_shape[1] * x_shape[2] * x_shape[3])
        N = float(x_shape[1] * x_shape[2] * x_shape[3])
        k = 0
        alpha = self.epsilon / self.T
        #while torch.sqrt(torch.norm(x_adv-x_ori, p=2)) < self.eta and k < self.T:  ## 3.3591
        while k < self.T:
            x_adv.requires_grad = True  # shape: [1,1,28,28]
            outputs = model(x_adv)
            loss1 = nn.CrossEntropyLoss()(outputs, labels)
            outputs_max, _ = torch.max(outputs, dim=1)
            #outputs_max = outputs.gather(1, labels.view(-1,1)).squeeze(1)
            grad1 = torch.autograd.grad(outputs_max, x_adv, grad_outputs = torch.ones_like(outputs_max), retain_graph = True, create_graph=True) # source map
            one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)
            outputs_sec, ind  = torch.max((1-one_hot_labels)*outputs, dim=1)
            grad2 = torch.autograd.grad(outputs_sec, x_adv, grad_outputs = torch.ones_like(outputs_sec), retain_graph = True, create_graph=True) # second map 
            # Compute Log Loss
            loss2 = (torch.log(torch.norm(grad1[0], p=1, dim=[1,2,3])) - torch.log(torch.norm(grad2[0], p=1,dim=(1,2,3)))).sum() / x_shape[0]
            # AOA loss
            loss = loss2 - self.lambda_ * loss1
            delta = torch.autograd.grad(loss, x_adv, retain_graph = True)
            x_adv = x_adv - alpha * torch.sign(delta[0] /(torch.norm(delta[0], p = 1, dim = [1, 2, 3], keepdim = True) / N))
            x_adv = torch.where(x_adv > x_ori + self.epsilon, x_ori + self.epsilon, x_adv)
            x_adv = torch.where(x_adv < x_ori - self.epsilon, x_ori - self.epsilon, x_adv)
            x_adv[:,0,:,:] = torch.clamp(x_adv[:,0,:,:], self.bound[0], self.bound[1])
            x_adv[:,1,:,:] = torch.clamp(x_adv[:,1,:,:], self.bound[2], self.bound[3])
            x_adv[:,2,:,:] = torch.clamp(x_adv[:,2,:,:], self.bound[4], self.bound[5])
            x_adv = x_adv.detach()
            k = k + 1
        return x_adv


class NAAO(object):
    def __init__(self, epsilon, steps, T_niters, mu, gamma, target_layer, path_type, device, bound):
        self.epsilon = epsilon
        self.steps = steps
        self.T_niters = T_niters
        self.mu = mu
        self.gamma = gamma
        self.target_layer = target_layer
        self.path_type = path_type
        self.bound = bound
        self.device = device 

    def attack(self, model, ori_img, label):
        
        x_adv = ori_img.to(self.device)        
        baseline = torch.zeros_like(ori_img).to(self.device)
        g_t = torch.zeros_like(ori_img).to(self.device)
        alpha = self.epsilon / self.T_niters

        for t in range(self.T_niters):
            gradients = []

            def hook_backward(module, grad_input, grad_output):
                gradients.append(grad_output[0].clone().detach())

            #Handle_backward = []

            for layer_name, layer in model.named_children(): 
                if layer_name == self.target_layer:
                    # backward_hook = layer.register_full_backward_hook(hook_backward)
                    backward_hook = layer.register_backward_hook(hook_backward)
                    # Handle_backward.append(backward_hook)
            
            for i in range(0, self.steps+1):
                if self.path_type == 's':
                    img_tmp = baseline + (float(i) / self.steps) * (x_adv - baseline) 
                else:
                    if i==0 or i == self.steps:
                        random_noise = (torch.zeros_like(x_adv)).to(self.device)
                    else:
                        random_noise = ((torch.rand(x_adv.shape) - 0.5) * 2 * self.epsilon).to(self.device)
                    img_tmp = baseline + (float(i) / self.steps) * (x_adv - baseline) + random_noise
                img_tmp.requires_grad_(True)
                output = model(img_tmp)
                output_label = output.gather(1, label.view(-1,1)).squeeze(1)
                grad = torch.autograd.grad(output_label, img_tmp, grad_outputs = torch.ones_like(output_label), retain_graph = True, create_graph = True)

            Intergrated_Attention = (sum(gradients) / len(gradients)).detach()
            backward_hook.remove()
            x_adv.requires_grad_(True)
            model_children = torchvision.models._utils.IntermediateLayerGetter(model, {self.target_layer: 'feature'})
    
            base_feature = model_children(baseline)['feature']
            x_feature = model_children(x_adv)['feature']
            attribution = (x_feature - base_feature) * Intergrated_Attention
            blank = torch.zeros_like(attribution)
            positive = torch.where(attribution >= 0, attribution, blank)
            negative = torch.where(attribution < 0,  attribution, blank)
            balance_attribution = positive + self.gamma * negative
            loss = torch.mean(torch.sum(balance_attribution, dim = [1, 2, 3]))
            loss.backward()
            grad = x_adv.grad

            g_t = self.mu * g_t + grad / torch.norm(grad, p=1, dim = [1,2,3], keepdim = True)
            x_adv = x_adv - alpha * torch.sign(g_t)

            x_adv = torch.where(x_adv > ori_img + self.epsilon, ori_img + self.epsilon, x_adv)
            x_adv = torch.where(x_adv < ori_img - self.epsilon, ori_img - self.epsilon, x_adv)

            x_adv[:,0,:,:]= torch.clamp(x_adv[:,0,:,:], self.bound[0], self.bound[1])
            x_adv[:,1,:,:]= torch.clamp(x_adv[:,1,:,:], self.bound[2], self.bound[3])
            x_adv[:,2,:,:]= torch.clamp(x_adv[:,2,:,:], self.bound[4], self.bound[5])
            x_adv = x_adv.detach()
        return x_adv 


class BLURIGO(object):
    def __init__(self, epsilon, steps, sigma_max, grad_step, T_niters, gamma, mu,  device, bound):
        self.epsilon = epsilon
        self.steps = steps
        self.T_niters = T_niters
        self.device = device
        self.mu = mu
        self.gamma = gamma
        self.bound = bound
        self.sigma_max = sigma_max
        self.grad_step = grad_step

    def GaussianBlurConv(self, img, sigma, kernel_size = 5, channels=3):
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        kernel = kernel * kernel.transpose(1,0)  
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, channels, axis=0)  # 
        kernel = kernel.to(self.device)
        weight = nn.Parameter(data=kernel, requires_grad=False)
        output = F.conv2d(img, weight, padding = 2, groups = channels)
        return output


    def attack(self, model, ori_img, label):
        x_adv = ori_img.to(self.device)
        baseline = torch.zeros_like(ori_img).to(self.device)
        g_t = torch.zeros_like(ori_img).to(self.device)
        alpha = self.epsilon / self.T_niters
        for t in range(self.T_niters):
            x_temp = x_adv.detach()
            gradients_mid = []
            for i in range(0, self.steps):
                sigma = i * self.sigma_max / float(self.steps)
                img_tmp = self.GaussianBlurConv(x_temp, sigma)
                img_tmp.requires_grad_(True)
                output = model(img_tmp)
                output_label = output.gather(1, label.view(-1,1)).squeeze(1)
                grad_first = torch.autograd.grad(output_label, img_tmp, grad_outputs = torch.ones_like(output_label), retain_graph = True, create_graph=True)
                    
                gradients_mid.append(grad_first[0])   
            
            x_adv.requires_grad_(True)
            attribution = torch.zeros_like(x_adv)
            for i in range(0, self.steps):
                sigma = i * self.sigma_max / float(self.steps)
                gaussian_gradient = (self.GaussianBlurConv(x_adv, sigma + self.grad_step) - self.GaussianBlurConv(x_adv, sigma)) / float(self.grad_step)
                attribution += gaussian_gradient * gradients_mid[i]
           
            attribution = (self.sigma_max * attribution) / self.steps

            blank = torch.zeros_like(attribution)
            positive = torch.where(attribution >= 0, attribution, blank)
            negative = torch.where(attribution < 0,  attribution, blank)
            balance_attribution = positive + self.gamma * negative
            loss = torch.mean(torch.sum(balance_attribution, dim = [1, 2, 3]))
            loss.backward()
            grad = x_adv.grad

            g_t = self.mu * g_t + grad / torch.norm(grad, p=1, dim = [1,2,3], keepdim = True)
            x_adv = x_adv - alpha * torch.sign(g_t)

            x_adv = torch.where(x_adv > ori_img + self.epsilon, ori_img + self.epsilon, x_adv)
            x_adv = torch.where(x_adv < ori_img - self.epsilon, ori_img - self.epsilon, x_adv)

            x_adv[:,0,:,:]= torch.clamp(x_adv[:,0,:,:], self.bound[0], self.bound[1])
            x_adv[:,1,:,:]= torch.clamp(x_adv[:,1,:,:], self.bound[2], self.bound[3])
            x_adv[:,2,:,:]= torch.clamp(x_adv[:,2,:,:], self.bound[4], self.bound[5])
            x_adv = x_adv.detach()
        return x_adv



class NRDM(object):
    def __init__(self, epsilon, T_niters, target_layer, device, bound):
        self.epsilon = epsilon
        self.T_niters = T_niters
        self.target_layer = target_layer
        self.bound = bound
        self.device = device
    
    def attack(self, model, ori_img, label):
        x_adv = torch.zeros_like(ori_img)
        alpha = self.epsilon / self.T_niters
        for t in range(self.T_niters):
            x_adv.requires_grad_(True)
            model_children = torchvision.models._utils.IntermediateLayerGetter(model, {self.target_layer: 'feature'})
            feature_ori = model_children(ori_img)['feature']
            feature_adv = model_children(x_adv)['feature']
            loss = torch.sum(torch.norm(feature_adv - feature_ori, p=2, dim = [1,2,3])) / x_adv.shape[0]
            loss.backward()
            grad = x_adv.grad
            x_adv = x_adv + alpha * torch.sign(grad)
            x_adv = torch.where(x_adv > ori_img + self.epsilon, ori_img + self.epsilon, x_adv)
            x_adv = torch.where(x_adv < ori_img - self.epsilon, ori_img - self.epsilon, x_adv)
            x_adv[:,0,:,:]= torch.clamp(x_adv[:,0,:,:], self.bound[0], self.bound[1])
            x_adv[:,1,:,:]= torch.clamp(x_adv[:,1,:,:], self.bound[2], self.bound[3])
            x_adv[:,2,:,:]= torch.clamp(x_adv[:,2,:,:], self.bound[4], self.bound[5])
            x_adv = x_adv.detach()
        return x_adv



class FDA(object):
    def __init__(self, epsilon,  T_niters, mu,  target_layer, device, bound):
        self.epsilon = epsilon
        self.T_niters = T_niters
        self.bound = bound
        self.device = device
        self.mu = mu
        self.target_layer = target_layer
        self.bound = bound

    def attack(self, model, ori_img, label):

        x_input = ori_img.clone().detach()

        g_t = torch.zeros_like(x_input)

        x_adv = x_input.clone().detach()

        alpha = self.epsilon / self.T_niters
        for t in range(self.T_niters):
            grad = torch.zeros_like(x_adv)
            for idx, layer in enumerate(self.target_layer):
                model_children = torchvision.models._utils.IntermediateLayerGetter(model, {layer: 'feature'})
                x_adv.requires_grad_(True)
                x_feature = model_children(x_adv)['feature']
                channel_feature = torch.mean(x_feature, 1, keepdim=True)
                blank = torch.zeros_like(x_feature)
                positive = torch.where(x_feature >= channel_feature, x_feature, blank)
                negative = torch.where(x_feature < channel_feature, x_feature, blank)
                loss = torch.mean(torch.log(torch.norm(negative, p = 2, dim = [1,2,3])) - torch.log(torch.norm(positive, p = 2, dim = [1, 2, 3])))
                loss.backward()
                grad += x_adv.grad
                x_adv = x_adv.detach()
            g_t = self.mu * g_t + grad / torch.norm(grad, p=1, dim = [1,2,3], keepdim = True)
            x_adv = x_adv + alpha * torch.sign(g_t)
            x_adv = torch.where(x_adv > ori_img + self.epsilon, ori_img + self.epsilon, x_adv)
            x_adv = torch.where(x_adv < ori_img - self.epsilon, ori_img - self.epsilon, x_adv)
            x_adv[:,0,:,:]= torch.clamp(x_adv[:,0,:,:], self.bound[0], self.bound[1])
            x_adv[:,1,:,:]= torch.clamp(x_adv[:,1,:,:], self.bound[2], self.bound[3])
            x_adv[:,2,:,:]= torch.clamp(x_adv[:,2,:,:], self.bound[4], self.bound[5])
            x_adv = x_adv.detach()
        return x_adv


class FIA(object):
    def __init__(self, epsilon, steps, T_niters, mu, pro,  target_layer, device, bound):
        self.epsilon = epsilon
        self.steps = steps 
        self.T_niters = T_niters
        self.bound = bound
        self.device = device
        self.mu = mu
        self.pro = pro
        self.target_layer = target_layer

    def attack(self, model, ori_img, label):
        loss_fn = nn.CrossEntropyLoss()         
        gradients = []
        def hook_backward(module, grad_input, grad_output):
            gradients.append(grad_output[0].clone().detach())

        alpha = self.epsilon / self.T_niters
        Handle_backward = []
        for layer_name, layer in model.named_children():
            if layer_name ==  self.target_layer:
                backward_hook = layer.register_backward_hook(hook_backward)
                Handle_backward.append(backward_hook)

           
        for n in range(self.steps):
            # mask_x = torch.bernoulli(torch.ones_like(ori_img.shape) * self.pro)
            x_mask = torch.bernoulli(torch.ones_like(ori_img) * self.pro)
            x_rest = ori_img * x_mask
            x_rest.requires_grad_(True) 
            predict_y = model(x_rest)
            loss = loss_fn(predict_y, label)
            loss.backward()

        grad_features = sum(gradients) / float(len(gradients))
        print('gradients:' ,len(gradients))
        grad_features = grad_features / torch.norm(grad_features, p = 2, dim = [1,2,3], keepdim = True)


        x_adv = torch.zeros_like(ori_img)
        g_t = torch.zeros_like(ori_img)


        for t in range(self.T_niters):
            model_children = torchvision.models._utils.IntermediateLayerGetter(model, {self.target_layer: 'feature'})
            x_adv.requires_grad_(True)
            feature_adv = model_children(x_adv)['feature']
            ojb_loss = torch.sum(grad_features * feature_adv) / x_adv.shape[0]
            ojb_loss.backward()
            grad = x_adv.grad
            g_t = self.mu * g_t + grad / torch.norm(grad, p = 1, dim =[1, 2, 3], keepdim = True)
            x_adv = x_adv - alpha * torch.sign(grad)
            x_adv = torch.where(x_adv > ori_img + self.epsilon, ori_img + self.epsilon, x_adv)
            x_adv = torch.where(x_adv < ori_img - self.epsilon, ori_img - self.epsilon, x_adv)
            x_adv[:,0,:,:]= torch.clamp(x_adv[:,0,:,:], self.bound[0], self.bound[1])
            x_adv[:,1,:,:]= torch.clamp(x_adv[:,1,:,:], self.bound[2], self.bound[3])
            x_adv[:,2,:,:]= torch.clamp(x_adv[:,2,:,:], self.bound[4], self.bound[5])
            x_adv = x_adv.detach()
        return x_adv   


class DIM(object):
    def __init__(self, epsilon, T_niters, mu, p_value, device,  bound):
        self.epsilon = epsilon
        self.T_niters = T_niters
        self.mu = mu 
        self.p_value = p_value
        self.device = device  		
        self.bound = bound

    def input_diversity(self, input_tensor, low_bound, high_bound, prob):
        image_shape = input_tensor.shape[3]
        rnd = torch.randint(low_bound, high_bound, ())
        rescaled = F.interpolate(input_tensor, size = [rnd, rnd], mode = 'bilinear', align_corners=True)
        h_rem = high_bound - rnd
        w_rem = high_bound - rnd
        pad_top = torch.randint(0, h_rem, ())
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(0, w_rem, ())
        pad_right = w_rem - pad_left
        pad_list = (pad_left, pad_right, pad_top, pad_bottom)
        padded = nn.ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom), 0.)(rescaled)
        padded = nn.functional.interpolate(padded, [image_shape, image_shape])
        return padded if torch.rand(()) < prob else input_tensor


    def attack(self, model, ori_img, label):

        loss_fn = nn.CrossEntropyLoss()

        g_t = torch.zeros_like(ori_img)

        x_adv = ori_img.clone().detach()

        alpha = self.epsilon / self.T_niters

        for t in range(self.T_niters):
            x_adv.requires_grad_(True)
            x_div = self.input_diversity(x_adv, 299, 330, self.p_value)
            predict = model(x_div)
            loss = loss_fn(predict, label)
            loss.backward()
            grad = x_adv.grad
            g_t = self.mu * g_t + grad / torch.norm(grad, p=1, dim = [1,2,3], keepdim = True)
            x_adv = x_adv + alpha * torch.sign(g_t)
            x_adv = torch.where(x_adv > ori_img + self.epsilon, ori_img + self.epsilon, x_adv)
            x_adv = torch.where(x_adv < ori_img - self.epsilon, ori_img - self.epsilon, x_adv)
            x_adv[:,0,:,:]= torch.clamp(x_adv[:,0,:,:], self.bound[0], self.bound[1])
            x_adv[:,1,:,:]= torch.clamp(x_adv[:,1,:,:], self.bound[2], self.bound[3])
            x_adv[:,2,:,:]= torch.clamp(x_adv[:,2,:,:], self.bound[4], self.bound[5])
            x_adv = x_adv.detach()
        return x_adv

