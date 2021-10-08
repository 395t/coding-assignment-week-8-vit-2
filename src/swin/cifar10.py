import csv
import numpy as np
import torch
import torch.nn.functional as F
import timm 

from torch import nn
from torchvision import datasets,transforms
from tqdm import tqdm

csv_file = "swin_cifar10.csv"
batch_size = 1024

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.Resize((32,32)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])

training_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
validation_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
 
training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size =batch_size, shuffle=False)

#model = timm.models.swin_transformer.SwinTransformer(img_size  = 32, num_classes = 10,  embed_dim = 64, depths = (4,8,4,2), num_heads=(2,4,4,2), window_size = 4).to(device)
model = timm.models.swin_transformer.SwinTransformer(img_size  = 32, num_classes = 10,  embed_dim = 64, depths = (2,2,8,2), num_heads=(2,4,4,8), window_size = 4).to(device)
#model = timm.models.swin_transformer.SwinTransformer(img_size  = 32, num_classes = 10,  embed_dim = 128, depths = (2, 2, 16, 2), num_heads=(4, 8, 16, 32), window_size = 4).to(device)
#model = timm.models.swin_transformer.SwinTransformer(img_size  = 32, num_classes = 10,  embed_dim = 16, depths = (2,4), num_heads=(2,2), window_size = 4).to(device)

epochs = 40
criterion = nn.CrossEntropyLoss()
params = model.parameters()

# Default hparams for Adam
optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

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
            
        epoch_loss = running_loss/len(training_dataset)
        epoch_acc = running_corrects/ len(training_dataset)
        
        val_epoch_loss = val_running_loss/len(validation_dataset)
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
        
