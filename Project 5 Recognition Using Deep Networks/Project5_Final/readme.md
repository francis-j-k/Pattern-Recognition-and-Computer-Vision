# Foobar

Code by Francis Jacob Kalliath 
Project 5: Recognition using Deep Networks

## Operating system and IDE

Ubuntu 20.04.5 LTS
Visual Studio Code 1.74.3


## Instructions for running the Calibration and projection of the objects for all tasks

```python




//In the Terminal type 

open the directory Project5
mkdir results  (create a directory named resukts inside Project5)


python MNIST_Model_1.py  #--  for Task 1, get data, train model, save model and plot graphs
python MNIST_Test_1.py  #--  for Task 1, load the model from MNIST_Model_1 and test on test dataset and custom written dataset
python examNetwork_2.py  #--  for Task 2, examine the first layer of the model and plot the first layer and the custon image through first layer filter
python GREEK_Model_3.py  #-- for Task 3, load the model from MNIST_Model_1 modify the last layer, train the model and plot the graphs
python experiment_4.py  #-- for Task 4, model to examine the different deep networks by modifying different aspects

#Extention
python pre-trained_first-layer.py #loadind different pretrained models including vgg11, alexnet, densenet121 and mobilenet_v2. comparing the first and second layers using a convolution plots. comparing a deep convolution layer of vgg11


# util.py - is a helper function with models, functions and hyper-parameters





```


## I have used 3 Time Travel Days out of 4 remaining Time Travel Days
