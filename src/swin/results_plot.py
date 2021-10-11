import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('C:/Users/Daniel/Desktop/School/swin_vit/swin_cifar_e30.csv')
df.plot(x="Epoch",  y=['Training Accuracy', 'Validation Accuracy'], title='Swin CIFAR-10')

df2 = pd.read_csv('C:/Users/Daniel/Desktop/School/swin_vit/swin_stl_e30.csv')
df2.plot(x="Epoch",  y=['Training Accuracy', 'Validation Accuracy'], title='Swin STL-10')

df3 = pd.read_csv('C:/Users/Daniel/Desktop/School/swin_vit/swin_tiny_e30.csv')
df3.plot(x="Epoch",  y=['Training Accuracy', 'Validation Accuracy'], title='Swin Tiny Imagenet')


df4 = pd.read_csv('C:/Users/Daniel/Desktop/School/swin_vit/swin_cifar_loss_e5.csv')
df4.plot(x="Epoch",  y=[ 'training_0.005', 'training_0.0005', 'training_5e-05'], title='Swin CIFAR-10 LRs')

df4 = pd.read_csv('C:/Users/Daniel/Desktop/School/swin_vit/swin_stl_loss_e5.csv')
df4.plot(x="Epoch",  y=[ 'training_0.005', 'training_0.0005', 'training_5e-05'], title='Swin STL-10 LRs')

df5 = pd.read_csv('C:/Users/Daniel/Desktop/School/swin_vit/swin_tiny_loss_e5.csv')
df5.plot(x="Epoch",  y=[ 'training_0.005', 'training_0.0005', 'training_5e-05'], title='Swin Tiny Imagenet LRs')
