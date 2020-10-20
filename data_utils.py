import torch
import numpy as np

from torchvision import datasets, transforms as T, models
from PIL import Image


def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    phases = ['train', 'valid', 'test']
    dirs = {'train': train_dir, 
            'valid': valid_dir, 
            'test' : test_dir}
    
    # transform & normalize
    normalize = T.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    
    data_transforms = {
        'train': T.Compose([T.RandomRotation(30),
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize]),
        'test': T.Compose([T.Resize(255),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize]),
        'valid': T.Compose([T.Resize(255),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize])
    }
    
    # Load the datasets with ImageFolder
    image_datasets = {x: datasets.ImageFolder(dirs[x], transform=data_transforms[x]) for x in phases}

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True) for x in phases}
    
    return image_datasets, dataloaders


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # scale
    image = Image.open(image)
    width, height = image.size
    aspect_ratio = width/height
    
    # resize
    if width <= height:
        image_resized = image.resize((256, int(256/aspect_ratio)))
    else:
        image_resized = image.resize((int(256*aspect_ratio), 256))
    
    # crop
    w, h = image_resized.size

    left = (w-224)/2
    upper = (h-224)/2
    right = left + 224
    lower = upper + 224
    
    image_cropped = image_resized.crop((left, upper, right, lower))
    
    # color
    np_image = np.array(image_cropped)/255
    
    # normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_normalized = (np_image - mean)/std
    
    # reorder dimensions
    processed_image = image_normalized.transpose((2, 0, 1))

    return torch.from_numpy(processed_image).type(torch.FloatTensor)