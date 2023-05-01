'''
Code by Francis Jacob Kalliath 
Project 5: Recognition using Deep Networks
'''
    #Extension
import ssl
import sys
import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import utils
import urllib
from PIL import Image
def main(argv):

    url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
    filename = "dog.jpg"

    try: urllib.URLopener().retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename)
    
    for i in range(1, 5):
        if i == 1:
            #importing the vgg11 model
            model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
            model.eval()
            convolution1 = model.features[0]
            convolution2 = model.features[3]
            convolution3 = model.features[18]
        elif i == 2:
            #importing the alexnet model
            model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
            model.eval()
            convolution1 = model.features[0]
            convolution2 = model.features[3]
        elif i == 3:
            #importing the densenet121 model
            model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
            model.eval()
            convolution1 = model.features.conv0
            convolution2 = model.features.denseblock1.denselayer1.conv1

        elif i == 4:
            #importing the mobilenet_v2 model
            model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
            model.eval()
            convolution1 = model.features[0][0]
            convolution2 = model.features[1].conv[0]

        # Define the model
        new_model = nn.Sequential(*list(model.features.children())[:2])

        # Preprocessing the image
        picture = Image.open(filename)
        preprocess = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        transformed_picture = preprocess(picture)
        processed_picture = np.transpose(torch.squeeze(transformed_picture,1).numpy(),(1,2,0))
    
        analyze_filters(convolution1, processed_picture)
        analyze_filters(convolution2, processed_picture)
        #analyze_filters(convolution3, processed_picture)

    
def analyze_filters(conv, first_image):
    # plot the first 10 filters of convolutional layer
    filters = []
    with torch.no_grad():
        for i in range(16):
            plt.subplot(4, 4, i+1)
            plt.tight_layout()
            fil_curr = conv.weight[i,0]
            filters.append(fil_curr)
            print (f'Filter {i+1} has shape {fil_curr.shape}')
            print (fil_curr)
            plt.imshow(fil_curr, interpolation='none')
            plt.title(f'Filter {i+1}')
            plt.xticks([])
            plt.yticks([])
        plt.show()

if __name__ == "__main__":
    main(sys.argv)