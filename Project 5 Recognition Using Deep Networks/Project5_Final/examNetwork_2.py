'''
Code by Francis Jacob Kalliath 
Project 5: Recognition using Deep Networks
'''

# imports
import utils
from utils import MyNetwork
import numpy as np
import sys
import torch
from torch.utils.data import DataLoader
import torchvision

def main(argv):
    # loading the model
    model = torch.load('model10.pth')
    print(model)

    filters = utils.plot_filters(model.conv1, 10, 3, 4)

    # loading training data
    train_data_loader = DataLoader(
        torchvision.datasets.MNIST('data2', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])))

    #obtain the first image
    first_image, first_label = next(iter(train_data_loader))
    squeezed_image = np.transpose(torch.squeeze(first_image, 1).numpy(), (1, 2, 0))

    # plot the first images from 1st layer filtered by the 10 filters
    utils.plot_filtered_images(filters, squeezed_image, 10, 20, 5, 4)

if __name__ == "__main__":
    main(sys.argv)
