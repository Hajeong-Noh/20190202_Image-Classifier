# Import packages for functions
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image

# Load flowers data
def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
  
    return trainloader, validloader,  train_data.class_to_idx

# Archtecture setup function
def setup(arch="vgg16", hidden_units=256, dropout=0.2, learning_rate=0.001, gpu=True):
    input_units = 0
    if arch=="vgg16":
        model = models.vgg16(pretrained=True)
        input_units = 25088
    elif arch=="densenet121":
        model = models.densenet121(pretrained=True)
        input_units = 1024
    else:
        print("{} is not a valid architecture. Available architectures are: vgg16 and densenet121".format(arch))
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = nn.Sequential(nn.Linear(input_units, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden_units, 102),
                                     nn.LogSoftmax(dim=1))
    
    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
    
    if torch.cuda.is_available() and gpu:
            model.cuda()
    
    return model, criterion, optimizer

def train_network(model, criterion, optimizer, trainloader, validloader, epochs=2,  print_every=10, gpu=True):
    steps = 0
    running_loss = 0

    print("--------------Training Start------------- ")
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1

            # Move input and label tensors to the default device
            if torch.cuda.is_available() and gpu:
                inputs, labels = inputs.cuda(), labels.cuda()   

            optimizer.zero_grad()  
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        if torch.cuda.is_available() and gpu:
                            inputs, labels = inputs.cuda(), labels.cuda()
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    print("--------------Training End------------- ")
    return epochs
    
def save_checkpoint(path, arch, hidden_units, dropout, learning_rate, gpu, model, optimizer, class_to_idx):
    checkpoint = {
            'arch': arch,
            'hidden_units': hidden_units,
            'dropout': dropout,
            'learning_rate': learning_rate,
            'gpu': gpu,
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'class_to_idx': class_to_idx}

    torch.save(checkpoint, 'checkpoint.pth')
    
def load_checkpoint(path='checkpoint.pth'):
    checkpoint = torch.load(path)
    
    model, criterion, optimizer = setup(checkpoint['arch'], checkpoint['hidden_units'], checkpoint['dropout'], checkpoint['learning_rate'], checkpoint['gpu'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    idx_to_class = { v : k for k, v in model.class_to_idx.items()}
                              
    return model, optimizer, idx_to_class

def map_label(category_names):
    # Label mapping
    import json
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    
    transform = transforms.Compose([transforms.Resize(255),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                                             [0.229, 0.224, 0.225])])
    
    img = transform(img)
    return img

def predict(image_path, model, topk, gpu, cat_to_name, idx_to_class):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    img = process_image(image_path)
    img = img.unsqueeze(0)
    
    with torch.no_grad():
        if torch.cuda.is_available() and gpu:
            output = model.forward(img.cuda())
        else:
            output = model.forward(img)
            
    prob = torch.exp(output)
    
    probs, indice = prob.topk(topk)
    probs = np.array(probs.view(5))
    indice = indice.view(5)
    classes = [idx_to_class[idx] for idx in indice.cpu().numpy()]
    class_names = [cat_to_name[cl] for cl in classes]

    print("----Predicted class names and probabilities----")
    for i in range(len(probs)):
        print(class_names[i], ": ", round(probs[i] * 100, 2), "%")