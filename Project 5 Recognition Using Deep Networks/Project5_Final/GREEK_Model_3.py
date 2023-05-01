'''
Code by Francis Jacob Kalliath 
Project 5: Recognition using Deep Networks
'''

import utils
import ssl
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from torch import nn
import matplotlib.pyplot as plt


ssl._create_default_https_context = ssl._create_unverified_context
#finetunes the model and replaces the last layer with 3 output channels tests the model on custon images
def main(argv):

    torch.backends.cudnn.enabled = False
    torch.manual_seed(42)
    model = torch.load('model10.pth')
    # freeze the network weights
    for param in model.parameters():
        param.requires_grad = False
    # replace the last layer with a new layer with 3 outputs
    model.fc2 = nn.Linear(50, 3)
    print(model)

    train_data_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder( '/home/francis-kalliath/PRCV_work/Project5_all/Project5_1/greek_train/greek_train',
                                        transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),utils.GreekTransform(),
                                                                                       torchvision.transforms.Normalize(
                                                                                           (0.1307,), (0.3081,) ) ] ) ),batch_size = 5,shuffle = True )
    train_losses = []
    train_counter = []
    optimizer = optim.SGD(model.parameters(), lr=utils.LEARNING_RATE,
                          momentum=utils.MOMENTUM)

    for epoch in range(1, 25):
        utils.train(epoch, model, optimizer, train_data_loader, train_losses, train_counter,5)
    # plotting the graph
    plt.plot(train_counter, train_losses, color='blue')    
    plt.legend(['Train Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()
    
    # saving the model that is trained
    torch.save(model, 'model20.pth')
    torch.save(model.state_dict(), 'model_state_dict11.pth')

    model.eval()

    # load custom digit data, apply the model, and plot the ten results
    print("1")
    image_dir = '/home/francis-kalliath/PRCV_work/Project5_all/Project5_2/greek_dataset/'
    custom_images = datasets.ImageFolder(image_dir,
                                         transform=transforms.Compose([transforms.Resize((28, 28)),
                                                                       transforms.Grayscale(),
                                                                       transforms.functional.invert,
                                                                       transforms.ToTensor(),
                                                                       transforms.Normalize((0.1307,), (0.3081,))]))
    custom_loader = DataLoader(custom_images)
    first_three_custom_data, first_three_custom_label = utils.first_three_output1(custom_loader, model)
    utils.plot_prediction1(first_three_custom_data, first_three_custom_label, 3, 3, 4)
    return


if __name__ == "__main__":
    main(sys.argv)
