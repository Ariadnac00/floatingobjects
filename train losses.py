import matplotlib.pyplot as plt
import csv

plt.figure(figsize=(5, 6))
with open('../floatingobjects/entrenos/train log.csv', 'r', newline='\n') as f:
    reader = list(csv.DictReader(f))
    last_train = reader[-1]
    train_losses = last_train['Train losses']
    optim = last_train['Optimizer']
    hidd_ch = last_train['Hidden channels']
    lr = last_train['Learning rate']
    batch_size = last_train['Batch size']

    plt.plot(eval(train_losses), color='green')
    plt.axis([0, 20, 0, 1])
    plt.suptitle(f'Training loss over epochs')
    plt.title(f'With {hidd_ch} hidden channels, {lr} learning rate\n Batch size of: {batch_size}, {optim} optimizer')

    plt.show()