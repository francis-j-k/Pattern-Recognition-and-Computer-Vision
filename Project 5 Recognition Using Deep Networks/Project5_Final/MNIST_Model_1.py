'''
Code by Francis Jacob Kalliath 
Project 5: Recognition using Deep Networks
'''
#imports
import utils
import ssl
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision

ssl._create_default_https_context = ssl._create_unverified_context

def main(argv):
    torch.backends.cudnn.enabled = False
    torch.manual_seed(42)
    #loading the train data
    train_data_loader = DataLoader(
        torchvision.datasets.MNIST('data2', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),batch_size=utils.BATCH_SIZE_TRAIN, shuffle=True)
    #loading the test data
    test_data_loader = DataLoader(
        torchvision.datasets.MNIST('data2', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),batch_size=utils.BATCH_SIZE_TEST, shuffle=True)

    utils.plot_images(train_data_loader, 2, 3)
    #importing the network
    network = utils.MyNetwork()
    optimizer = optim.SGD(network.parameters(), lr=utils.LEARNING_RATE,
                          momentum=utils.MOMENTUM)
    train_loss = []
    train_count = []
    test_loss = []
    test_count = [i * len(train_data_loader.dataset) for i in range(utils.N_EPOCHS + 1)]
    utils.test(network, test_data_loader, test_loss)
    #training the model
    for epoch in range(1, utils.N_EPOCHS + 1):
        utils.train(epoch, network, optimizer, train_data_loader, train_loss, train_count, 64)
        utils.test(network, test_data_loader, test_loss)
    #plotting the graphs
    utils.plot_curve(train_count, train_loss, test_count, test_loss)
    #saving the network
    torch.save(network, 'model10.pth')
    torch.save(network.state_dict(), 'model_state_dict10.pth')
    return

if __name__ == "__main__":
    main(sys.argv)
