import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json

from collections import OrderedDict


###################################
#
# Main funtion
#
#
###################################
def main():
    
    # get arguments from command line
    input = get_args()
    
    path_to_image = input.image_path
    checkpt = input.checkpoint
    num = input.top_k
    cat_names = input.category_names
    gpu = input.gpu
    
    # load category names file
    with open(cat_names, 'r') as f:
        cat_to_name = json.load(f)
       
    # load trained model
    model = load_checkpoint(checkpt)
    
    # Process images, predict classes, and display results
    #img = Image.open(path_to_image)
    #image = process_image(img)
    labels = predict(path_to_image, model, gpu, cat_to_name, num)
    print("Top {} classes are \n".format(num))
    print(labels)
    ans = labels.loc[labels['predictions'].idxmax()]

    print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\nThe catgeory of the flower is \"{}\" with prediction value \"{}\"\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n".format(ans.category, ans.predictions))


###################################
#
# Get command line arguments from 
# argparse
#
###################################
def get_args():
    """
        Get arguments from command line
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("image_path", type=str, help="path to image in which to predict class label")
    parser.add_argument("checkpoint", type=str, help="checkpoint in which trained model is contained")
    parser.add_argument("--top_k", type=int, default=5, help="number of classes to predict")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json",
                        help="file to convert label index to label names")
    parser.add_argument("--gpu", type=bool, default=False,
                        help="use GPU or CPU to train model: True = GPU, False = CPU")
    
    return parser.parse_args()

###################################
#
# build the layers of teh model to
# be trained.
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
# Load the model from prvoided checkpoint
# file.
#
###################################
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    
    model_load = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    for param in model_load.parameters():
        param.requires_grad=False
        
    model_load.class_to_idx = checkpoint['class_to_idx']
    
    model_clf = build_network(checkpoint['input_size'], checkpoint['hidden_layers'], checkpoint['drop'], checkpoint['output_size'])
    model_load.classifier = model_clf
    model_load.classifier.optimizer = checkpoint['optimizer']
    model_load.classifier.learning_rate = checkpoint['learning_rate']
    model_load.load_state_dict(checkpoint['state_dict'])
    return model_load

###################################
#
# Process the image to perform the
# necessary transformation like resize
# crop etc.
#
###################################
def process_image(path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img_loader = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor()])
    
    pil_image = Image.open(path)
    pil_image = img_loader(pil_image).float()
    
    np_image = np.array(pil_image)    
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std    
    np_image = np.transpose(np_image, (2, 0, 1))
            
    return np_image

###################################
#
# function to provide the topk classes
# for the given image 
# and their respective prediction. 
#
###################################
def predict(image_path, model, gpu, cat_to_name, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    pytorch_img = process_image(image_path)
    pytorch_tensor = torch.tensor(pytorch_img)
    
    # Got rutime error: Expected object of type torch.DoubleTensor but found type torch.FloatTensor
    # 
    pytorch_tensor = pytorch_tensor.float()
    
    #Got runtime error of match the convolution dimensions
    #https://discuss.pytorch.org/t/expected-stride-to-be-a-single-integer-value-or-a-list/17612
    pytorch_tensor = pytorch_tensor.unsqueeze(0)

    if gpu==True:
        pytorch_tensor = pytorch_tensor.to('cuda')
    
    model.eval()
    ps = model.forward(pytorch_tensor)
    preds = torch.exp(ps)
    
    
    top_p, top_class = preds.topk(topk)
    
    print(top_p)
    print(top_class)
    
    top_p = top_p.tolist()
    top_class = top_class.tolist()
    
    #print(top_class)
    #print(model.class_to_idx)
    
    #print(pd.Series(model.class_to_idx))
    #print(pd.Series(cat_to_name))
    
    #print(cat_to_name)
    cat_df = pd.DataFrame({'category': cat_to_name})
    #display(cat_df.head(n=15))
    cat_df['class'] = pd.Series(model.class_to_idx)
    cat_df = cat_df.set_index('class')
    #display(cat_df.head(n=10))
    
    # Limit the dataframe to top labels and add their predictions
    labels = cat_df.iloc[top_class[0]]
    labels['predictions'] = top_p[0]
    
    return labels



# Run the program
if __name__ == "__main__":
    main()
