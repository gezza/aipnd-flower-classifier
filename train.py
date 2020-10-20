# imports
import json
import argparse
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict
from data_utils import load_data
from model_utils import define_model, train_model

# parse args from command line
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Parser command line arguments for Flower Image Classifier',
    )
    parser.add_argument('--data_dir', type=str , default='flowers', help='location of datasets')
    parser.add_argument('--save_dir', type=str , default='saved_models/checkpoint.pth', help='location to save checkpoint')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'vgg13'],  help='Pretrained model architecture. vgg16 or vgg13')
    parser.add_argument('--learning_rate', type=float , default=0.001, help='learning rate')
    parser.add_argument('--hidden_units', type=int , default=512, help='hidden units')
    parser.add_argument('--epochs', type=int , default=3, help='number of training epochs')
    parser.add_argument('--gpu', type=bool , default=False, help='use a GPU')

    return parser.parse_args()


def main():
    # load data
    image_datasets, dataloaders = load_data(args.data_dir)
    
    # label mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    categories_num = len(cat_to_name)

    # load pretrained model(s)
    model = define_model(args.arch, args.hidden_units, categories_num)
    
    # Define loss Function
    criterion = nn.NLLLoss()

    # Define optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Use GPU if available
    device = torch.device("cuda" if args.gpu else "cpu")
    model.to(device)
    
    # train model
    print("Start training model using: {}".format(device))
    train_model(model, image_datasets, dataloaders, criterion, optimizer, scheduler, args.epochs, device)
    print("Model training completed!")
    
    # save checkpoint
    if args.save_dir:
        model.class_to_idx = image_datasets['train'].class_to_idx
        checkpoint = {'arch': args.arch,
                      'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'classifier': model.classifier,
                      'epochs': args.epochs,
                      'class_to_idx': model.class_to_idx}
        torch.save(checkpoint, args.save_dir)
    
# Example command: python train.py --gpu true --arch vgg16 --learning_rate 0.003 --hidden_units 256 --epochs 5
if __name__ == "__main__":
    args = parse_arguments()
    main()
