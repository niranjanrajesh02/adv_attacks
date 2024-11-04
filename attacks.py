import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Subset, DataLoader

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from model_utils import load_pretrained_model, load_data, get_ds_labels, denorm_img, get_test_acc



def pgd_attack(model, x, y, attack_params):
    # args
    # model: nn.Module - model to be attacked
    # x: torch.Tensor - batched image input
    # y: torch.Tensor - batched ground truth labels (or target labels if targeted is True)
    # attack_params: dict
            # eps: float - epsilon budget
            # targeted: bool - whether to use targeted attack
            # iters: int - number of attack iterations
            # step_size: float - step size per iteration
    
    eps = attack_params['eps']
    targeted = attack_params['targeted']
    iters = attack_params['iters']
    step_size = attack_params['step_size']
    if targeted:
        y = attack_params['target_class']
    model.eval()
    assert len(x.shape) == 4 # (N, C, H, W)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if targeted:
        loss_function = lambda output, target: - nn.CrossEntropyLoss()(output, target) # minimise loss (max accuracy) for targeted class
    else:
        loss_function = nn.CrossEntropyLoss() # maximise general loss function

    x, y = x.to(device), y.to(device)

    pert_x = x.clone().detach()
    pert_x.requires_grad = True

    for i in range(iters):
        output = model(pert_x)
        model.zero_grad()
        loss = loss_function(output, y)
        loss.backward()

        pert_x = pert_x + step_size * torch.sign(pert_x.grad)
        pert_x = torch.clamp(pert_x, 0, 1)
        pert_x = torch.clamp(pert_x, x - eps, x + eps).detach_()
        pert_x.requires_grad = True

    return pert_x

def fgsm_attack(model, x, y, attack_params): 
    # args
    # model: nn.Module - model to be attacked
    # x: torch.Tensor - batched image input
    # attack_params: dict
    #       eps: float - epsilon budget
    #       targetted: bool - whether to use targeted attack
    # 
    # returns: torch.Tensor - perturbed image (batched)


    eps = attack_params['eps']
    targeted = attack_params['targeted']
    if targeted:
        y = attack_params['target_class']
    model.eval()

    assert len(x.shape) == 4 # (N, C, H, W)

    if targeted:
        loss_function = lambda output, target: - nn.CrossEntropyLoss()(output, target) # minimise loss (max accuracy) for targeted class
    else:
        loss_function = nn.CrossEntropyLoss() # maximise general loss function
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x, y = x.to(device), y.to(device)

    pert_x = x.clone().detach()
    pert_x.requires_grad = True

    output = model(pert_x)
    model.zero_grad()
    loss = loss_function(output, y)
    loss.backward()

    pert_x = pert_x + eps * torch.sign(pert_x.grad) # single step towards increasing loss
    pert_x = torch.clamp(pert_x, 0, 1)
    pert_x = torch.clamp(pert_x, x - eps, x + eps).detach_()


    return pert_x


def eval_attack(model, test_ds, attack_params, attack_algorithm='fgsm', ds_name='cifar10'):
    # args
    # model: nn.Module - model to be adv evaluated
    # test_ds: torch.utils.data.Dataset - test dataset
    # eps: float - epsilon budget
    #
    # returns: float - attack accuracy
    correct = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    attacks={
        'fgsm': fgsm_attack,
        'pgd': pgd_attack
    }
    attack = attacks[attack_algorithm]
    eps = attack_params['eps']
    if eps != 0:
        for imgs,labels in tqdm(test_ds):
            imgs, labels = imgs.to(device), labels.to(device)
            adv_imgs = attack(model, imgs, labels, attack_params)
            output = model(adv_imgs)
            preds = output.argmax(dim=1, keepdim=True)
            correct += preds.eq(labels.view_as(preds)).sum().item()

           

        adv_acc = correct / len(test_ds.dataset)
        return adv_acc

    else:
        acc = get_test_acc(model, test_ds)
        return acc

def visualize_attack(model, test_ds, attack_params, attack_algorithm='fgsm', ds_name='cifar10'):
    # args
    # model: nn.Module - model to be adv evaluated
    # test_ds: torch.utils.data.Dataset - test dataset
    # attack_params: dict
    # 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    attacks={
        'fgsm': fgsm_attack,
        'pgd': pgd_attack
    }
    attack = attacks[attack_algorithm]

    plt.figure(figsize=(4, 4))
    for batch_i, (imgs, labels) in enumerate(test_ds):
            adv_imgs = attack(model, imgs, labels, attack_params)
            imgs, labels = imgs.to(device), labels.to(device)

            predicted = model(imgs).argmax(dim=1, keepdim=True)
            predicted_adv = model(adv_imgs).argmax(dim=1, keepdim=True)

            for img_i in range(2):
                # Plot clean and adv
                plt.subplot(2, 2, 2*img_i + 1)

                label = labels[img_i].cpu().detach().numpy()
                img = imgs[img_i].cpu().detach()
                img = denorm_img(img, ds_name)[0]

               
                img = img.permute(1, 2, 0)
                plt.imshow(img)
                plt.axis(False)
                plt.title(f'Label: {get_ds_labels(ds_name)[label.item()]},\n Prediction: {get_ds_labels(ds_name)[predicted[img_i].item()]}', fontsize=8)
                plt.subplot(2, 2, 2*img_i + 2)
                adv_img = adv_imgs[img_i].cpu().detach()
                adv_img = denorm_img(adv_img, ds_name)[0]
                adv_img = adv_img.permute(1, 2, 0)
                plt.imshow(adv_img)
                plt.title(f'Epsilon={attack_params['eps']},\n Prediction: {get_ds_labels(ds_name)[predicted_adv[img_i].item()]}', fontsize=8)
                plt.axis(False)

            break
    
    plt.savefig(f'./example_imgs/{ds_name}_{attack_algorithm}.png')

    return
