import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import transforms, datasets, models
from torch.utils.data.sampler import SubsetRandomSampler
import pandocfilters as f
import os

from glob import glob
from PIL import Image

from sklearn.metrics import f1_score

print(os.listdir("Original Images"))
train_transforms = transforms.Compose([transforms.Resize(size=(224, 224)),
                                       transforms.ToTensor()])
validation_transforms = transforms.Compose([transforms.Resize(size=(224, 224)),
                                            transforms.ToTensor()
                                            ])
img_dir = 'Original Images'
train_data = datasets.ImageFolder(img_dir, transform=train_transforms)
# number of subprocesses to use for data loading
num_workers = 0
# percentage of training set to use as validation
valid_size = 0.1
test_size = 0.1

# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([
    transforms.ToTensor()
])
# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
valid_split = int(np.floor((valid_size) * num_train))
test_split = int(np.floor((valid_size + test_size) * num_train))
valid_idx, test_idx, train_idx = indices[:valid_split], indices[valid_split:test_split], indices[test_split:]

print(len(valid_idx), len(test_idx), len(train_idx))

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
test_sampler = SubsetRandomSampler(test_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=20,
                                           sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=20,
                                           sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(train_data, batch_size=20,
                                          sampler=test_sampler, num_workers=num_workers)




from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


def test(model, criterion, use_cuda):
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.
    pre = 0.0
    count = 0
    rec = 0.0
    f1_m = 0.0
    au = 0.0
    y_pred = []
    y_target = []

    for batch_idx, (data, target) in enumerate(test_loader):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        y_pred_list = torch.tensor([a.squeeze().tolist() for a in pred])
        targetlist = torch.tensor([t.squeeze().tolist() for t in target])

        y_pred.append(y_pred_list)
        y_target.append(targetlist)
        # y_target = torch.stack(target,1)

        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
        test_accuracy = 100. * correct / total
    print('Test Loss: {:.6f}\n'.format(test_loss))
    print('\nTest Accuracy: %2.5f%% (%2d/%2d)' % (
        100. * correct / total, correct, total))
    y_pred_list = torch.cat(y_pred)
    targetlist = torch.cat(y_target)
    y_predd = y_pred_list.numpy()
    targett = targetlist.numpy()
    aucc = recall_score(targett, y_predd, average='macro')

    rec = recall_score(targett, y_predd, average='macro')
    pre = precision_score(targett, y_predd, average='macro')
    f_m = f1_score(targett, y_predd, average='macro')

    # auu=float(au/count)
    print("Precision=" + str(pre))
    print("Recall=" + str(rec))
    print("F1=" + str(f_m))
    return test_accuracy

def train(n_epochs, model, optimizer, criterion, use_cuda, save_path):
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf
    for epoch in range(1, n_epochs + 1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            # initialize weights to zero
            optimizer.zero_grad()
            output = model(data)
            # calculate loss
            loss = criterion(output, target)
            # back prop
            loss.backward()
            # grad
            optimizer.step()
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            if batch_idx % 100 == 0:
                print('Epoch %d, Batch %d loss: %.6f' %
                      (epoch, batch_idx + 1, train_loss))
        ######################
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(valid_loader):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            output = model(data)
            loss = criterion(output, target)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
        ))
        # test(model, criterion, use_cuda)
        if valid_loss < valid_loss_min:
            torch.save(model.state_dict(), save_path)
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            valid_loss_min = valid_loss
    return model
model = models.alexnet(pretrained=True)
#12 30
for param in model.parameters():
    param.requires_grad = False
num_ftrs=model.classifier[6].in_features
#model.classifier[1]=nn.Conv2d(512,2,kernel_size=(1,1), stride=(1,1))
model.classifier[6]= nn.Linear(num_ftrs, 6)
fc_parameters = model.classifier[6].parameters()
for param in fc_parameters:
    param.requires_grad = True
use_cuda = torch.cuda.is_available()
if use_cuda:
    model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.classifier[6].parameters(), lr=0.0001,momentum=0.9)
train(30, model, optimizer, criterion, use_cuda, 'alexnet.pt')
model.load_state_dict(torch.load('alexnet.pt'))
test(model, criterion, use_cuda)