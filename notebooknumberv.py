#%%
#Load data

import torch
import torchvision
from torch.optim import Adam
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Subset

transform = transforms.Compose([            
            transforms.Resize(28),                    
            transforms.CenterCrop(24),                
            transforms.ToTensor(),                     
            transforms.Normalize(                      
            mean=[0.485],                #, 0.456, 0.406
            std=[0.229]                  #, 0.224, 0.225
            )])

batch_size = 100



trainset1 = torchvision.datasets.EMNIST(root='./data' ,split="balanced", train=True, download=True, transform=transform)

testset1 = torchvision.datasets.EMNIST(root='./data', split="balanced", train=False, download=True, transform=transform)

########SHUFFEL MUSS FALSE SEIN !!!!!
trainloader1 = torch.utils.data.DataLoader(trainset1, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

testloader1 = torch.utils.data.DataLoader(testset1, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
indezesToTrain=[]
indezesToTest=[]
positionInDataset =0
for inputs,labels in enumerate(trainloader1):
    for x in range(len(labels[1])):
        if labels[1][x].item() <= 9 or labels[1][x].item() ==36:
            #if labels[1][x] == 36:
            #    testloader1[inputs][x] = 10
            indezesToTrain.append(positionInDataset)
            #print(positionInDataset)
            #print(labels[1][x].item())
            #print("\n")
        positionInDataset +=1

positionInDataset =0
for inputs,labels in enumerate(testloader1):
    for x in range(len(labels[1])):
        if labels[1][x].item() <= 9 or labels[1][x].item() ==36:
            #if labels[1][x] == 36:
            #    testloader1[inputs][x] = 10
            indezesToTest.append(positionInDataset)
            #print(positionInDataset)
            #print(labels[1][x].item())
            #print("\n")
        positionInDataset +=1


trainset = Subset(trainset1, indezesToTrain)
testset = Subset(testset1, indezesToTest)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
dataloaders = {
    "train": trainloader,
    "val": testloader,
}


# show/save image
import matplotlib.pyplot as plt
import numpy as np

batch = next(iter(trainloader))
print(batch[0].shape)
plt.imshow(batch[0][0].permute(1, 2, 0)) # image an der ersten stelle
print(batch[1][0]) # label an der ersten stelle
plt.savefig("./data/img.png")
#plt.show()



# %%
#Train the model  
import time
import os
#HELPER function we need
def relu_evidence(y):
    return F.relu(y)

def one_hot_embedding(labels, num_classes):
    # Convert to One Hot Encoding
    ###functioniert aber hässlich### corregiert labels
    for x in range(len(labels)):
        if labels[x] == 36:
            labels[x] = 10
    #print(num_classes)
    #print(labels)
    y = torch.eye(num_classes)
    return y[labels]

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device

def calculate_evidence(preds, labels, outputs, num_classes):
    match = torch.reshape(torch.eq( preds, labels).float(), (-1, 1))
    acc = torch.mean(match)
    evidence = relu_evidence(outputs)
    alpha = evidence + 1
    u = num_classes / torch.sum(alpha, dim=1, keepdim=True)
    u = u.mean()
    total_evidence = torch.sum(evidence, 1, keepdim=True)
    mean_evidence = torch.mean(total_evidence)
    mean_evidence_succ = torch.sum(
    torch.sum(evidence, 1, keepdim=True) * match) / torch.sum(match + 1e-20)
    mean_evidence_fail = torch.sum(
    torch.sum(evidence, 1, keepdim=True) * (1 - match)) / (torch.sum(torch.abs(1 - match)) + 1e-20)

    return u, mean_evidence , mean_evidence_succ , mean_evidence_succ
###### LOSSES
import torch
import torch.nn.functional as F


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)

#The Kullback-Leibler Divergence score, or KL divergence score,
#quantifies how much one probability distribution differs from 
#another probability distribution.
def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device = get_device()
    beta = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    #Sum dirchlet distribution
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    #Sum (number of labers)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    # ln (gammafunct) | gammafunc = (n-1)!
    lnB = torch.lgamma(S_alpha) -         torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    # ist das  nicht immer 0? | torch.sum(torch.lgamma(beta)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1,
                        keepdim=True) - torch.lgamma(S_beta)
    #ableitung gammafunc(x) / gammafunc(x)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    # sum( dirchlet - 1 *(digamma(Sum_alpha) -(digamma(alpha)) ) 
    # + ???  
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1,
                   keepdim=True) + lnB + lnB_uni
    return kl

#???
def loglikelihood_loss(y, alpha, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    
    loglikelihood_err = torch.sum(
        (y - (alpha / S)) ** 2, dim=1, keepdim=True)
    
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood
#???
def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    annealing_coef = torch.min(torch.tensor(
        1.0, dtype=torch.float32), torch.tensor(epoch_num / annealing_step, dtype=torch.float32))

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef *         kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div

# EQ 4 mit func = digamma
def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(torch.tensor(
        1.0, dtype=torch.float32), torch.tensor(epoch_num / annealing_step, dtype=torch.float32))

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef *         kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(mse_loss(target, alpha, epoch_num,
                               num_classes, annealing_step, device=device))
    return loss


def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(edl_loss(torch.log, target, alpha,
                               epoch_num, num_classes, annealing_step, device))
    return loss


def edl_digamma_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(edl_loss(torch.digamma, target, alpha,
                               epoch_num, num_classes, annealing_step, device))
    return loss


device = torch.device("cuda:0")
model_directory = ""
def train_model(model, dataloaders, criterion, optimizer, model_directory ,device, num_classes, num_epochs, is_train=True, uncertainty=False):
    print("im using:" + str(device)) # see if using GPU cuda

    since = time.time()
    
    acc_history = []
    loss_history = []
    evidence_history = []

    best_acc = 0.0
    best_evidence = 0.0

    directory = './results/models/' + model_directory
    if not os.path.exists(directory):
        os.makedirs(directory)    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders:
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward

            #with uncertainty
            if uncertainty:
                y = one_hot_embedding(labels, num_classes)
                y = y.to(device)
                outputs = model(inputs)
                # save the gradients to _ and predictions in preds
                _, preds = torch.max(outputs, 1)
                loss = criterion(
                            outputs, y.float(), epoch, num_classes, 10, device)
            
                ############## evidence calculations ##########################
                # U = uncertainty ?
                u, mean_evidence , mean_evidence_succ , mean_evidence_succ = calculate_evidence(preds, labels, outputs, num_classes)
                
            
            #without uncertainty
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                ############## evidence calculations ##########################
                # U = uncertainty ?
                u, mean_evidence , mean_evidence_succ , mean_evidence_succ = calculate_evidence(preds, labels, outputs, num_classes)
            # backward
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders.dataset)
        epoch_acc = running_corrects.double() / len(dataloaders.dataset)
        ###me
        epoch_evidence1 =  mean_evidence #total_evidence , ean_evidence_succ ,mean_evidence_fail

        ###me 
        print('Loss: {:.4f} Acc: {:.4f} Uncertainty_mean: {:.4f} Evidence_mean: {:.4f} '.format(epoch_loss, epoch_acc, u.item() ,epoch_evidence1.item()))
        #### herausfinden wie ich uncertainty bekomme und was unterschied zu evidenze ist ???#####
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc

        if epoch_evidence1 > best_evidence:
            best_evidence = epoch_evidence1
            
        acc_history.append(epoch_acc.item())
        loss_history.append(epoch_loss)
        evidence_history.append(epoch_evidence1.item())

        # speichert jede Epoche
        torch.save(model.state_dict(), os.path.join(directory, '{0:0=2d}.pth'.format(epoch)))
        print(f"Saved: " + directory + '{0:0=2d}.pth'.format(epoch))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Acc: {:4f} Best Evidence: {:4f}'.format(best_acc, best_evidence))
    
    
    return acc_history, loss_history , evidence_history

import copy
import glob
def eval_model(model, dataloaders, model_directory, device, num_classes):
    since = time.time()
    
    acc_history = []
    best_acc = 0.0
    best_evidence = 0.0

    # placeholder for saving the best model
    best_model = copy.deepcopy(model.state_dict())

    directory = './results/models/' + model_directory
    
    if not os.path.exists(directory):
        os.makedirs(directory)    

    saved_models = glob.glob(directory + '*.pth')
    saved_models.sort()
    print('saved_model', saved_models)

    for model_path in saved_models:
        print('Loading model', model_path)

        model.load_state_dict(torch.load(model_path))
        model.eval()
        model.to(device)

        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            ############## evidence calculations ##########################
            # U = uncertainty ?
            u, mean_evidence , mean_evidence_succ , mean_evidence_succ = calculate_evidence(preds, labels, outputs, num_classes)
        epoch_acc = running_corrects.double() / len(dataloaders.dataset)
        epoch_evidence1 = mean_evidence 
        print('Acc: {:.4f}'.format(epoch_acc))
        print('Evidence: {:.4f}'.format(epoch_evidence1))
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model = copy.deepcopy(model.state_dict())

        if epoch_evidence1 > best_evidence:
            best_evidence = epoch_evidence1

        acc_history.append(epoch_acc.item())
        #evidence history ???
        
        print()
    
    torch.save(model.state_dict(), os.path.join(directory , 'bestmodel.pth'))
    print(f"Saved the best model after eval" + directory + 'best_model.pth')

    time_elapsed = time.time() - since
    print('Validation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Acc: {:4f} Best Evidenz: {:4f}'.format(best_acc , best_evidence))
    
    return acc_history # evidenz/uncertainty history ?



def hist_plot(train_loss_hist,train_evidence_hist,val_acc_hist, model_directory):
    
    directory = './results/models/' + model_directory
    if not os.path.exists(directory):
        os.makedirs(directory)    
    # save the plots
    plt.figure(0)
    plt.plot(val_acc_hist)
    plt.plot(train_loss_hist)
    #plt.savefig(directory + 'trainHistoAccuracyLoss.png')
    plt.show()

    plt.figure(1)
    plt.plot(train_evidence_hist)
    #plt.savefig(directory + 'trainHistoEvidence.png')
    plt.show()
    print()
    print("saved TrainHisto" + model_directory)
    print()

# because we have a pretrained model the weights arent randomly but allready set 
# to match a concrete Problem 
#
# so we want to use this to our advantage , 
# by making the learningrate of the pretrained "resuable" small 
# so it mostly keeps it and just learns most of it in our new linear layer

#Load used Model
import torchvision.models as models

#resnet18 = models.resnet18(pretrained= True)
resnet18 = models.resnet18(pretrained= True)

# we can se in the output that the resnet18 we have has 
# 1000 output layers but we have 10 Outputclasses for CIFAR100
#print(resnet18)

#%%
# so we need to change the last layer !! keep the input the same !!
# well add an linar layer wit 47 Outpuclasses

import torch.nn as nn
################################
num_classes = 11

resnet18.conv1 = torch.nn.Conv2d(1,64,
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), bias=False)


resnet18.fc = nn.Linear(512, num_classes)

#here are all parameters like: linear1.. liner2..conv2d...
all_parameters = list(resnet18.parameters())

#we want last layer to have a faster learningrate 
without_lastlayer =all_parameters[0: len(all_parameters) -1]
#te
#so we extract it
last_param = resnet18.fc.parameters()

num_epochs = 25
criterion = edl_digamma_loss
uncertainty = True
#criterion = nn.CrossEntropyLoss()
#criterion = edl_log_loss
#passing a nested dict for different learningrate with differen params
optimizer = Adam(resnet18.parameters()) #Adam([
                #{'params': without_lastlayer},
                #{'params': resnet18.fc(), 'lr': 1e-3}
                #], lr=1e-2) 


########################



train_acc_hist, train_loss_hist , train_evidence_hist = train_model(resnet18, trainloader, criterion, optimizer,model_directory, device,num_classes =num_classes, num_epochs =num_epochs, uncertainty= uncertainty)
val_acc_hist = eval_model(resnet18, testloader, model_directory ,device, num_classes=num_classes)
    
# saves the histogramms 
hist_plot(train_loss_hist,train_evidence_hist,val_acc_hist, model_directory)