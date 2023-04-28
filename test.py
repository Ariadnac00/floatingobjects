from model import get_UNet_model, get_deepUNet_model
from tqdm import tqdm
import json
import csv
import torch
from codigo.data import FloatingSeaObjectDataset
from codigo.transforms import get_transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from codigo.train import predict_images
from sklearn.metrics import classification_report


def dice_coefficient(inputs, targets, smooth=1):
    # flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    dice = (2. * intersection + smooth) / (torch.sum(inputs) + torch.sum(targets) + smooth)
    return dice


def intersection_over_union(inputs, targets, smooth=1):
    # flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    # intersection is equivalent to True Positive count
    # union is the mutually inclusive area of all labels & predictions
    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection

    IoU = (intersection + smooth) / (union + smooth)

    return IoU


def test_model(model, test_set):
    with torch.no_grad():
        model.eval()
        dice = []
        iou = []
        predicciones = []

        for idx in tqdm(range(len(test_set))):
            image, target = test_set[idx]
            images = torch.unsqueeze(image, axis=0)
            images = images.to(device)
            targets = torch.unsqueeze(target, axis=0)
            targets = targets.to(device)
            outputs = model(images)
            predicciones.append(outputs.cpu().detach().numpy()[0])

            dice.append(dice_coefficient(torch.tensor(np.where(np.exp(outputs.cpu()) > 1, 1., 0.)).to(device), targets).cpu())
            iou.append(intersection_over_union(torch.tensor(np.where(np.exp(outputs.cpu()) > 1, 1., 0.)).to(device), targets).cpu())

        print('')
        print(f"Dice Coefficient = {np.array(dice).mean():.4f}")
        print(f"Intersection over Union = {np.array(iou).mean():.4f}")
        return predicciones, np.array(dice).mean(), np.array(iou).mean()

################################################
################ CAMBIAR FECHA #################
################################################

path_save = '../floatingobjects/entrenos/2023_04_27_09_36/'

with open(f'{path_save}params.json') as json_file:
    params = json.load(json_file)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if params['Modelo'] == 'UNet':
  model = get_UNet_model(params['Input channels'], params['Hidden channels'], params['Output channels'], params['BN'])
elif params['Modelo'] == 'deepUNet':
  model = get_deepUNet_model(params['Input channels'], params['Hidden channels'], params['Output channels'], params['BN'])

model.load_state_dict(torch.load(path_save + 'last_net.pt', map_location=device))
model.to(device)

test_set = FloatingSeaObjectDataset('./data', fold="test",
                                    transform=get_transform("test", intensity=0, add_fdi_ndvi=(params['Input channels'] == 14)),
                                    output_size=64, seed=0, hard_negative_mining=False)

predicciones, DC, IoU = test_model(model, test_set)

y_true = []
y_pred_exp = []
for idx in tqdm(range(len(predicciones))):
  for pix in np.where(np.exp(predicciones[idx]) > 0.5, 1, 0).reshape(-1):
    y_pred_exp.append(pix)
  for pix in test_set[idx][1].detach().numpy().reshape(-1):
    y_true.append(pix)

class_report = classification_report(y_true, y_pred_exp, zero_division=0, output_dict=True)
print(class_report)
metrics = {'Path save': params["Path save"],
           'Water p': class_report['0.0']['precision'],
           'Water r': class_report['0.0']['recall'],
           'Water f1': class_report['0.0']['f1-score'],
           'Plastic p': class_report['1.0']['precision'],
           'Plastic r': class_report['1.0']['recall'],
           'Plastic f1': class_report['1.0']['f1-score'],
           'Average p': class_report['macro avg']['precision'],
           'Average r': class_report['macro avg']['recall'],
           'Average f1': class_report['macro avg']['f1-score'],
           'Accuracy': class_report['accuracy'],
           'DC': DC,
           'IoU': IoU}

with open('../floatingobjects/entrenos/metrics.csv', 'a', newline='\n') as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writerow(metrics)


np.random.seed(params['Seed'])
torch.manual_seed(params['Seed'])
torch.cuda.manual_seed_all(params['Seed'])
np.random.seed(params['Seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

test_loader = DataLoader(test_set, batch_size=5, shuffle=True)

for i in range(5):
    fig = predict_images(test_loader, model, device)
    fig.suptitle(f'{params["Modelo"]}, {params["Input channels"]} in_channels, {params["Hidden channels"]} hidden channels, BN {params["BN"]}, {params["Loss"][0]} loss')
    plt.savefig(f'{path_save}test{i+1}.png')
