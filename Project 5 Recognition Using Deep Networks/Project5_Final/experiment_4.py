'''
Code by Francis Jacob Kalliath 
Project 5: Recognition using Deep Networks
'''

import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import utils

batch_size_test = 64


#Class to build the network
class BasicNetwork(nn.Module):
    def __init__(self, num_of_conv, conv_filter_size, dropout_rate):
        super(BasicNetwork, self).__init__()
        self.input_size = 28 
        self.num_of_conv = num_of_conv
        self.conv1 = nn.Conv2d(1, 10, kernel_size=conv_filter_size, padding='same')
        self.conv2 = nn.Conv2d(10, 20, kernel_size=conv_filter_size, padding='same')
        self.conv = nn.Conv2d(20, 20, kernel_size=conv_filter_size, padding='same')
        self.conv2_drop = nn.Dropout2d(dropout_rate)
        self.fc1 = nn.Linear(self.get_fc1_input_size(), 50)
        self.fc2 = nn.Linear(50, 10)

    
    #function to obtain the input size of first fc layer
    
    def get_fc1_input_size(self):
        fc1_size = self.input_size / 2
        fc1_size = fc1_size / 2
        fc1_size = fc1_size * fc1_size * 20
        return int(fc1_size)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        for i in range(self.num_of_conv):
            x = F.relu(self.conv(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, 1)

#the experiment class to accomplist one full round of training with the passed hyper parameters from the main method
def experiment(num_epochs, batch_size_train, num_of_conv, conv_filter_size, dropout_rate, filename):
    # loading the test and training data
    train_data_loader = DataLoader(
        torchvision.datasets.MNIST('experiment_data', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_train)

    test_data_loader = DataLoader(
        torchvision.datasets.MNIST('experiment_data', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_test)

    network = BasicNetwork(num_of_conv, conv_filter_size, dropout_rate)
    optimizer = optim.SGD(network.parameters(), lr=utils.LEARNING_RATE,
                          momentum=utils.MOMENTUM)
    train_loss = []
    train_count = []
    test_loss = []
    test_count = [i * len(train_data_loader.dataset) for i in range(num_epochs + 1)]

    # training the model
    utils.test(network, test_data_loader, test_loss)
    for epoch in range(1, num_epochs + 1):
        utils.train(epoch, network, optimizer, train_data_loader, train_loss, train_count,batch_size_train)
        utils.test(network, test_data_loader, test_loss)

    # function call to plot the graphs
    plot_curve(train_count, train_loss, test_count, test_loss, filename)


'''
The function to plot the graphs
'''
def plot_curve(train_count, train_loss, test_count, test_loss, filename):
    plt.plot(train_count, train_loss, color='blue')
    plt.scatter(test_count, test_loss, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training transversed')
    plt.ylabel('negative log likelihood loss')
    plt.savefig(filename)


# main function

'''
Loop 108 times with different hyper-parameters
epoch sizes: 1, 3, 5
training batch sizes: 64, 128, 256
the number of convolution layers: 1, 2, 3, 4
convolution layer filter size: 3, 5, 7
dropout rate: 0.3, 0.5
'''
def main():
    for num_epochs in [1, 3, 5]:
        for batch_size_train in [64, 128, 256]:
            for num_of_conv in range(1, 4):
                for conv_filter_size in [3, 5, 7]:
                    for dropout_rate in [0.3, 0.5]:
                        filename = f'curve_{num_epochs}_{batch_size_train}_{num_of_conv}_{conv_filter_size}_{dropout_rate}.png'
                        print('*********************************')
                        print(f'Number of Epochs: {num_epochs}')
                        print(f'Train Batch Size: {batch_size_train}')
                        print(f'Number of Convolution Layer: {num_of_conv}')
                        print(f'Convolution Filter Size: {conv_filter_size}')
                        print(f'Dropout Rate: {dropout_rate}')
                        print('**********************************')
                        experiment(num_epochs, batch_size_train, num_of_conv, conv_filter_size, dropout_rate, filename)


if __name__ == "__main__":
    main()
