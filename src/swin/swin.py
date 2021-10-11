import csv
import numpy as np
import torch
import torch.nn.functional as F
import timm 

from torch import nn
from torchvision import datasets,transforms
from tqdm import tqdm

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    batch_size = 16
    epochs = 1
    dataset = 'tiny'  # cifar, tiny, or stl
    pretrained = True
    
    csv_file = 'swin_' + dataset + ('_pretrained' if pretrained else '') +'_e'+str(epochs)+'.csv'
    
    if dataset == 'cifar':
        img_size = 32
    elif dataset ==  'stl':
        img_size =  96
    elif dataset ==  'tiny':
        img_size =  64
    
    if pretrained:
        model_in = 224
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=True).to(device)
        model.head = nn.Linear(model.head.in_features, 200).to(device)
    else:
        model_in = img_size
        if dataset == 'cifar':
            model = timm.models.swin_transformer.SwinTransformer(img_size  = 32, num_classes = 10,  embed_dim = 64, depths = (2,2,8,2), num_heads=(2,4,8,16), patch_size=4, window_size = 8).to(device)
        elif dataset ==  'stl':
            model = timm.models.swin_transformer.SwinTransformer(img_size  = 96, num_classes = 10,  embed_dim = 64, depths = (2,2,8,2), num_heads=(2,4,8,16), patch_size=4, window_size = 12).to(device)
        elif dataset ==  'tiny':
            model = timm.models.swin_transformer.SwinTransformer(img_size  = 64, num_classes = 200,  embed_dim = 64, depths = (2,2,8,2), num_heads=(2,4,8,16), patch_size=4, window_size = 8).to(device)

    
   
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(img_size, padding=4),
        transforms.Resize(model_in),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    
    if dataset == 'cifar':
        training_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        validation_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif dataset ==  'stl':
        training_dataset = datasets.STL10(root='./data', split='train', download=True, transform=transform_train)
        validation_dataset = datasets.STL10(root='./data', split='test', download=True, transform=transform_test)
    elif dataset ==  'tiny':
        train_dir = "./tiny-imagenet-200/train"
        training_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
        test_dir = "./tiny-imagenet-200/val"
        validation_dataset = datasets.ImageFolder(test_dir, transform=transform_test)
        
    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers = 8)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size =batch_size, shuffle=False, num_workers = 8)

    criterion = nn.CrossEntropyLoss()
    params = model.parameters()
    
    # Default hparams for Adam
    optimizer = torch.optim.Adam(params, lr=5e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    
    results =  []
    
    for e in range(epochs):
        running_loss = 0.0
        running_corrects = 0.0
        val_running_loss = 0.0
        val_running_corrects = 0.0
        
        for inputs, labels in tqdm(training_loader):
        #for inputs, labels in training_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            outputs = model(inputs)
    
            loss = criterion(outputs, labels) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data).item()
    
        else:
            with torch.no_grad(): # No gradient for validation
                for val_inputs, val_labels in validation_loader:
                    val_inputs = val_inputs.to(device)
                    val_labels = val_labels.to(device)
    
                    val_outputs = model(val_inputs)
    
                    val_loss = criterion(val_outputs, val_labels)
                    
                    _, val_preds = torch.max(val_outputs, 1)
                    val_running_loss += val_loss.item()
                    val_running_corrects += torch.sum(val_preds == val_labels.data).item()
                
            epoch_loss = running_loss/len(training_loader)
            epoch_acc = running_corrects/ len(training_dataset)
            
            val_epoch_loss = val_running_loss/len(validation_loader)
            val_epoch_acc = val_running_corrects/ len(validation_dataset)
            
            
            epoch_results = {
                'Epoch': e,
                'Training Loss': round(epoch_loss, 3),
                'Validation Loss': round(val_epoch_loss, 3),
                'Training Accuracy': round(epoch_acc,3),
                'Validation Accuracy': round(val_epoch_acc,3)
                }
            
            print(epoch_results)
            results.append(epoch_results)
    
    csv_columns = ['Epoch', 'Training Loss', 'Validation Loss', 'Training Accuracy',  'Validation Accuracy']
    try:
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in results:
                writer.writerow(data)
    except IOError:
        print("I/O error")
        