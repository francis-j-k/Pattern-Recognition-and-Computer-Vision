'''
Code by Francis Jacob Kalliath 
Project 5: Recognition using Deep Networks
'''

# imports
import utils
from utils import MyNetwork
import sys
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
torch.set_printoptions(precision=2)

#importing the model from the MNIST_Model_1
def main(argv):
    # loading the model
    model = torch.load('model10.pth')
    model.eval()

    # loading the data for testing
    test_data_loader = DataLoader(
        torchvision.datasets.MNIST('data2', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])))

    # obtain the labels
    data_to_plot, label_to_plot = utils.first_ten_output(test_data_loader, model)

    #plotting the image predictions
    utils.plot_prediction(data_to_plot, label_to_plot, 9, 3, 3)

    # custon data is being loaded
    image_dir = '/home/francis-kalliath/PRCV_work/Project5_all/Project5_3/Handwritten_data'
    custom_images = datasets.ImageFolder(image_dir,
                                         transform=transforms.Compose([transforms.Resize((28, 28)),
                                                                       transforms.Grayscale(),
                                                                       transforms.functional.invert,
                                                                       transforms.ToTensor(),
                                                                       transforms.Normalize((0.1307,), (0.3081,))]))
    custom_loader = DataLoader(custom_images)
    first_ten_custom_data, first_ten_custom_label = utils.first_ten_output(custom_loader, model)
    utils.plot_prediction(first_ten_custom_data, first_ten_custom_label, 10, 3, 4)
    return

if __name__ == "__main__":
    main(sys.argv)