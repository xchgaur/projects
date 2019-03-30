import pandas as pd
import numpy as np

import torch
from torch import nn, optim
from torch.optim import lr_scheduler

import torchvision
from torchvision import datasets, transforms, models

from collections import OrderedDict
from PIL import Image
from os import listdir
import json
import argparse

vgg16 = models.vgg16(pretrained=True)
densenet = models.densenet121(pretrained=True)

model_dict = {"vgg16": vgg16, "densenet": densenet}
input_dict = {"vgg16": 25088, "densenet": 1024}


###################################
#
# Get comman dline args from
# argparse
#
###################################
def get_args():
    parser = argparse.ArgumentParser(description="Train Deep Learning Model for Flower Classification Model")
    parser.add_argument('data_dir', type=str, help="Directory containing images to be used for training, validation and testing")
    parser.add_argument('--save_dir', default='checkpoint_pymodel.pth', type=str, help="Location to save checkpoints")
    parser.add_argument('--arch', default='vgg16', help='Transfer learning options: vgg16, densenet')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--hidden_layers', default=[5000,500], type=list, help='list of hidden layers')
    parser.add_argument('--output_size', default=102, type=int, help='number of output categories')
    parser.add_argument('--drop', default=0.1, type=float, help='dropout probability')
    parser.add_argument('--epochs', default=5, type=int, help='number of epochs for training')
    parser.add_argument('--gpu', default=False, action='store_true', help='Use Gpu')
    return parser.parse_args()


###################################
#
# build layers of model to be trained
#
#
###################################
def build_network(input_size, hidden_layers, drop, output_size):
    classifier = nn.Sequential(OrderedDict([
                           ('fc1',nn.Linear(input_size,hidden_layers[0])),
                           ('ReLu1',nn.ReLU()),
                           ('Dropout1',nn.Dropout(p=drop)),
                           ('fc2',nn.Linear(hidden_layers[0], hidden_layers[1])),
                           ('ReLu2',nn.ReLU()),
                           ('Dropout2',nn.Dropout(p=drop)),
                           ('fc3',nn.Linear(hidden_layers[1],output_size)),
                           ('output',nn.LogSoftmax(dim=1))
                           ]))
    
    return classifier


###################################
#
# This will train the model on our train data
# and also run validation on validation set
# printing the validation loss and accuracy
###################################
def train_model(model_img, dataloaders, criterion, optimizer, epochs, gpu):
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in dataloaders['train']:
            steps += 1
            # Move input and label tensors to GPU
            if gpu==True:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            else:
                inputs, labels = inputs.to('cpu'), labels.to('cpu')

            optimizer.zero_grad()

            logps = model_img.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model_img.eval()
                with torch.no_grad():
                    for inputs, labels in dataloaders['valid']:
                        logps = model_img.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(dataloaders['valid']):.3f}.. "
                      f"Test accuracy: {accuracy/len(dataloaders['valid']):.3f}")
                running_loss = 0
                model_img.train()
                      
    return model_img


###################################
#
# Calculate accuracy of the model
# on test data.
#
#
###################################
def accuracy(testloader, model, criterion, gpu):
    accuracy = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            if gpu==True:
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')
            else:
                inputs, labels = inputs.to('cpu'), labels.to('cpu')

            logps = model.forward(inputs)
        
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
    print (f"Test accuracy: {accuracy/len(dataloaders['test']):.3f}")
    return accuracy

###################################
#
# Save the model to a checkpoint
#
#
###################################
def save_model(model, save_to, input_size, output_size, hidden_layers, drop, learning_rate, optimizer, image_datasets):
    model.cpu()

    checkpoint = {
            'arch': 'vgg16',
            'input_size': input_size,
            'output_size': output_size,
            'hidden_layers': hidden_layers,
            'drop': drop,
            'learning_rate': learning_rate,              
            'optimizer': optimizer,
            'class_to_idx': image_datasets['train'].class_to_idx,
            'state_dict': model.state_dict(),}
    torch.save(checkpoint, path)

def main():
    
    # get arguments from command line
    input = get_args()
    data_dir = input.data_dir
    itrain_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    save_to = input.save_dir
    model_type = input.archC
    learning_rate = input.learning_rate
    epochs = input.epochs
    hidden_layers = input.hidden_layers
    output_size = input.output_size
    gpu = input.gpu
    drop = input.drop

    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ])
    }
    

    image_datasets = {x: datasets.ImageFolder(data_dir + '/' + x,transform = data_transforms[x]) for x in data_transforms}
    print(image_datasets['train'])


    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,shuffle=True) 
               for x in image_datasets}

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    model = model_dict[model_type]
    input_size = input_dict[model_type]
    
    for param in model.parameters():
        param.requires_grad = False

    model_clf = build_network(input_size, hidden_layers, drop, output_size)
    model.classifier = model_clf
    print(model)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr=0.001)

    if gpu==True:
        model.to('cuda')
    else:
        model.to('cpu')

    tr_model = train_model(model,dataloaders, criterion, optimizer, epochs, gpu)
  
    print("Model has been successfully trained") 
    
    save_model(tr_model, save_to, input_size, output_size, hidden_layers, drop, learning_rate, optimizer, image_datasets)
    print("Model has been successfully saved")

# Run the program
if __name__ == "__main__":
    main()




