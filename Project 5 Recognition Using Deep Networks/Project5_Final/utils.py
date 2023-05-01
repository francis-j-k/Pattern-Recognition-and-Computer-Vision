'''
Code by Francis Jacob Kalliath 
Project 5: Recognition using Deep Networks
'''

import csv
from collections import Counter

import cv2
import torch
from numpy import linalg
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision

# defining the hyper-parameters
N_EPOCHS = 5
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 1000
LEARNING_RATE = 0.01
MOMENTUM = 0.5
LOG_INTERVAL = 10


# model definition
"""
The first convolution layer has ten 5x5 filters, and the second convolution layer has twenty 5x5 filters. 
The max pooling layers use a 2x2 window. The dropout layer has a dropout rate of 0.5 (50%). 
The first fully connected Linear layer has 50 nodes, and the second fully connected Linear layer has 10 nodes. 
The log_softmax function is applied to the output of the final layer.
"""

class MyNetwork(nn.Module):
    # initialize the network layers
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    # compute a forward pass for the network
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # relu on max pooled results of conv1
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))  # relu on max pooled results of dropout of conv2
        x = x.view(-1, 320)  # flatten operation
        x = F.relu(self.fc1(x))  # relu on fully connected linear layer with 50 nodes
        x = self.fc2(x)  # fully connect linear layer with 10 nodes
        return F.log_softmax(x, 1)  # apply log_softmax()


class MyNetwork1(nn.Module):
    # initialize the network layers
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=5)
        self.conv4 = nn.Conv2d(30, 40, kernel_size=5)
        self.conv5 = nn.Conv2d(40, 50, kernel_size=5)
        self.conv6_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(2560, 1280)
        self.fc2 = nn.Linear(1280, 640)
        self.fc3 = nn.Linear(640, 320)
        self.fc4 = nn.Linear(320, 50)
        self.fc5 = nn.Linear(50, 10)

    # compute a forward pass for the network
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # relu on max pooled results of conv1
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))  # relu on max pooled results of dropout of conv2
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv3(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv4(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv5(x)), 2))
        x = x.view(-1, 320)  # flatten operation
        x = F.relu(self.fc1(x))  # relu on fully connected linear layer with 50 nodes
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc3(x))
        x = self.fc5(x)  # fully connect linear layer with 10 nodes
        return F.log_softmax(x, 1)  # apply log_softmax()


import torch.nn as nn
import torch.nn.functional as F

class MyNetwork2(nn.Module):
    # initialize the network layers
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=5)
        self.conv4 = nn.Conv2d(30, 40, kernel_size=5)
        self.conv5 = nn.Conv2d(40, 50, kernel_size=5)
        self.conv6_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(2560, 1280)
        self.fc2 = nn.Linear(1280, 640)
        self.fc3 = nn.Linear(640, 320)
        self.fc4 = nn.Linear(320, 50)
        self.fc5 = nn.Linear(50 * 4 * 4, 10)

    # compute a forward pass for the network
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv3(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv4(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv5(x)), 2))
        x = x.view(-1, 50 * 4 * 4)  # flatten operation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return F.log_softmax(x, 1)

class MyNetwork5(nn.Module):
    # initialize the network layers
    def __init__(self):
        super(MyNetwork5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(256*3*3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    # compute a forward pass for the network
    def forward(self, x):
        x = F.relu(self.conv1(x))  # relu on conv1 results
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))  # max pooled results of dropout of conv2
        x = F.relu(F.max_pool2d(self.conv3(x), 2)) # max pooled results of conv3
        x = F.relu(F.max_pool2d(self.conv4(x), 2)) # max pooled results of conv4
        x = x.view(-1, 256*3*3)  # flatten operation
        x = F.relu(self.fc1(x))  # relu on fully connected linear layer with 512 nodes
        x = F.dropout(x, p=0.5, training=self.training) # dropout layer
        x = F.relu(self.fc2(x)) # relu on fully connected linear layer with 256 nodes
        x = F.dropout(x, p=0.25, training=self.training) # dropout layer
        x = self.fc3(x)  # fully connect linear layer with 10 nodes
        return F.log_softmax(x, dim=1)  # apply log_softmax()





#function plots images

def plot_images(data, row, col):
    examples = enumerate(data)
    batch_idx, (example_data, example_targets) = next(examples)
    for i in range(row * col):
        plt.subplot(row, col, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()



#training of the model is done in this method and saves the model and optimizer

def train(epoch, model, optimizer, train_loader, train_loss, train_count, a):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 2 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item()))
            train_loss.append(loss.item())
            train_count.append(
                (batch_idx * a) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(model.state_dict(), 'results/model.pth')
            torch.save(optimizer.state_dict(), 'results/optimizer.pth')



#testing of the model is performed in this methods and prints the accuracy information

def test(model, test_loader, test_loss):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_loss.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



#function plots curves of the training loses and testing losses

def plot_curve(train_count, train_loss, test_count, test_loss):
    plt.plot(train_count, train_loss, color='blue')
    plt.scatter(test_count, test_loss, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()


#The function applies a model to a dataset and retrieves the first ten data points along with their corresponding labels.

def first_ten_output(data, model):
    first_ten_data = []
    first_ten_label = []

    count = 0
    for data, target in data:
        if count < 10:
            squeeze_data = np.transpose(torch.squeeze(data, 1).numpy(), (1, 2, 0))
            first_ten_data.append(squeeze_data)
            with torch.no_grad():
                output = model(data)
                print(f'{count + 1} - Orginal value: {output}')
                print(f'{count + 1} - Actual value: {output.argmax().item()}')
                label = output.data.max(1, keepdim=True)[1].item()
                print(f'{count + 1} - Predicted Value: {label}')
                first_ten_label.append(label)
                count += 1

    return first_ten_data, first_ten_label

#The function applies a model to a dataset and retrieves the first three data points along with their corresponding labels for the greek dataset.
def first_three_output1(data, model):
    first_three_data = []
    first_three_label = []
    count = 0
    for data, target in data:
        if count < 3:
            squeeze_data = np.transpose(torch.squeeze(data, 1).numpy(), (1, 2, 0))
            first_three_data.append(squeeze_data)
            with torch.no_grad():
                output = model(data)
                print(f'{count + 1} - Orginal value: {output}')
                print(f'{count + 1} - Actual value: {output.argmax().item()}')
                label = output.data.max(1, keepdim=True)[1].item()
                print(f'{count + 1} - Predicted Value: {label}')
                first_three_label.append(label)
                count += 1

    return first_three_data, first_three_label


#The function generates a plot displaying the input image along with its corresponding prediction values gor the greek dataset.

def plot_prediction1(data_set, label_set, total, row, col):
    string_array = ['alpha', 'beta', 'gamma']
    for i in range(total):
        plt.subplot(row, col, i + 1)
        plt.tight_layout()
        plt.imshow(data_set[i][:,:,0], cmap='gray', interpolation='none')
        plt.title('Pred: {}'.format(string_array[label_set[i]]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


#The function generates a plot displaying the input image along with its corresponding prediction values

def plot_prediction(data_set, label_set, total, row, col):
    for i in range(total):
        plt.subplot(row, col, i + 1)
        plt.tight_layout()
        plt.imshow(data_set[i], cmap='gray', interpolation='none')
        plt.title('Pred: {}'.format(label_set[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


#The function plots some filters

def plot_filters(conv, total, row, col):
    filters = []
    with torch.no_grad():
        for i in range(total):
            plt.subplot(row, col, i + 1)
            plt.tight_layout()
            curr_filter = conv.weight[i, 0]
            filters.append(curr_filter)
            print(f'filter {i + 1}')
            print(curr_filter)
            print(curr_filter.shape)
            print('\n')
            plt.imshow(curr_filter)
            plt.title(f'Filter {i + 1}')
            plt.xticks([])
            plt.yticks([])
        plt.show()
    return filters


#The function plots filters and filtered images

def plot_filtered_images(filters, image, n, total, row, col):
    with torch.no_grad():
        items = []
        for i in range(n):
            items.append(filters[i])
            filtered_image = cv2.filter2D(np.array(image), ddepth=-1, kernel=np.array(filters[i]))
            items.append(filtered_image)
        for i in range(total):
            plt.subplot(row, col, i + 1)
            plt.tight_layout()
            plt.imshow(items[i],cmap='gray')
            plt.xticks([])
            plt.yticks([])
        plt.show()


# greek data set transform
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        x = torchvision.transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        return torchvision.transforms.functional.invert( x )
