#%%
#Load data

import torch
import torchvision

from torchvision import transforms

transform = transforms.Compose([            
            transforms.Resize(28),                    
            transforms.CenterCrop(224),                
            transforms.ToTensor(),                     
            transforms.Normalize(                      
            mean=[0.485, 0.456, 0.406],                
            std=[0.229, 0.224, 0.225]                  
            )])

batch_size = 100
trainset = torchvision.datasets.EMNIST(root='./data', train=True,
                                    download=True, transform=transform)

testset = torchvision.datasets.EMNIST(root='./data', train=False,
                                    download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
