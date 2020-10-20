import time
import torch
from torch import nn
from torchvision import models
from collections import OrderedDict

def define_model(arch, hidden_units, output_features):    
    # Load pretrained model
    model = getattr(models, arch)(pretrained=True)
    # print("Model with original classifier: {}".format(model))
    input_features = model.classifier[0].in_features
    
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Define classifier with ReLU & Dropout
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_features, 4096)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(0.2)),
        ('fc2', nn.Linear(4096, hidden_units)),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(0.2)),
        ('fc3', nn.Linear(hidden_units, output_features)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    # replace original classifier
    model.classifier = classifier

    # print("Model with new classifier: {}".format(model))

    return model


def train_model(model, datasets, dataloaders, criterion, optimizer, scheduler, epochs, device):
    start_time = time.time()
    print_frequency = 5
    steps = 0
    running_loss = 0

    for epoch in range(epochs):
        for images, labels in dataloaders['train']:
            steps += 1

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model(images)
            loss = criterion(logps, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if steps % print_frequency == 0:
                valid_loss = 0
                accuracy = 0

                model.eval()

                with torch.no_grad():
                    for images, labels in dataloaders['valid']:
                        images, labels = images.to(device), labels.to(device)

                        logps = model(images)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1) 
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print("Epoch {}/{}.. Step {}.. Train loss: {:.3f}.. Test loss: {:.3f}.. Test accuracy: {:.1f}%"
                      .format(epoch+1, 
                              epochs,
                              steps,
                              running_loss/print_frequency, 
                              valid_loss/len(dataloaders['valid']), 
                              accuracy/len(dataloaders['valid'])*100))
                running_loss = 0

                model.train()
                
    time_taken = time.time() - start_time
    print("Time taken to train: {:.1f} mins".format(time_taken/60))

    
def load_checkpoint(path):
    checkpoint = torch.load(path)

    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True) 
    elif checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained=True) 
    else:
        None
        
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.optimizer = checkpoint['optimizer']
    model.epochs = checkpoint['epochs']
   
    return model    
