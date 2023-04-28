import os
from datetime import datetime
import json

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import csv
from model import get_UNet_model, get_deepUNet_model
from losses import *
# from dataset import get_train_set
from codigo.data import FloatingSeaObjectDataset
from codigo.transforms import get_transform


def train(argumentos):
    # dataset de Jamila
    train_set = FloatingSeaObjectDataset('./data', fold="train", transform=get_transform("train", intensity=0,
                                         add_fdi_ndvi=(argumentos['Input channels'] == 14)),
                                         output_size=argumentos['Input size'], seed=argumentos['Seed'])

    # mi dataset
    # train_set = FSODataset(root='./data', fold='train', seed=0, output_size=argumentos['Input size'],
    #                       transform=get_transform('train', intensity=0, add_fdi_ndvi=(argumentos['Input channels'] == 14)), use_l2a_probability=0.5)

    batch_size = argumentos['Batch size']
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)

    if argumentos['Modelo'] == 'UNet':
        model = get_UNet_model(argumentos['Input channels'], argumentos['Hidden channels'], argumentos['Output channels'], BN=argumentos['BN'])
    elif argumentos['Modelo'] == 'deepUNet':
        model = get_deepUNet_model(argumentos['Input channels'], argumentos['Hidden channels'], argumentos['Output channels'], BN=argumentos['BN'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.train()

    if argumentos['Loss'][0] == 'BCE':
      loss_fn = torch.nn.BCEWithLogitsLoss()
    elif argumentos['Loss'][0] == 'FL':
      loss_fn = FocalLoss(gamma=argumentos['Loss'][1])
    elif argumentos['Loss'][0] == 'DL':
      loss_fn = DiceLoss()
    elif argumentos['Loss'][0] == 'Combo':
      loss_fn = DiceBCELoss()
    elif argumentos['Loss'][0] == 'IoU':
      loss_fn = IoULoss()


    if argumentos['Optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=argumentos['Learning rate'])
    elif argumentos['Optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=argumentos['Learning rate'])

    for epoch in range(argumentos['Epochs']):
        epoch_loss = []
        with tqdm(enumerate(train_loader), total=len(train_loader), leave=True) as pbar:
            for idx, batch in pbar:
                optimizer.zero_grad()

                images, targets = batch
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)

                loss = loss_fn(outputs, targets)
                epoch_loss.append(loss.item())
                loss.backward()
                optimizer.step()

                pbar.set_description(f'Epoch: {epoch+1}; Training Loss {np.array(epoch_loss).mean():.6f}')

        argumentos['Train losses'].append(np.array(epoch_loss).mean())
        if (epoch + 1) % 4 == 0:
            torch.save(model.state_dict(), argumentos['Path save'] + f'/epoch{epoch + 1}.pt')

    torch.save(model.state_dict(), argumentos['Path save'] + '/last_net.pt')
    argumentos['Last train loss'] = np.array(epoch_loss).mean()


    with open('../floatingobjects/entrenos/train log.csv', 'a', newline='\n') as f:
        writer = csv.DictWriter(f, fieldnames=list(argumentos.keys()))
        writer.writerow(argumentos)


now = datetime.now()
date = now.strftime("%Y_%m_%d_%H_%M")

args = {'Path save': f'entrenos/{date}',
        'Dataset': 'FSO - all',
        'Modelo': 'deepUNet',
        'BN': True,
        'Loss': ['Combo', None],
        'Seed': 0,
        'Input channels': 14,
        'Hidden channels': 64,
        'Output channels': 1,
        'Input size': 64,
        'Optimizer': 'Adam',
        'Batch size': 4,
        'Learning rate': 1e-4,
        'Epochs': 20,
        'Last train loss': None,
        'Train losses': []}

if not os.path.isdir(args['Path save']):
    os.mkdir(args['Path save'])

with open(f'{args["Path save"]}/params.json', 'w') as file:
    json.dump(args, file)

train(args)
