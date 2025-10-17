import argparse#passes paths/epochs/flags from the command line
import os,time#os is for the file paths, time is for logging the training time

import torch#pytorch
import torch.nn as nn#imoprts neural network building blocks
import torch.optim as optim#imports optimizers(AdamW)
from torchvision.models import resnet50, ResNet50_Weights#imports the resnet50 model and its weights

from dataloaders import build_loaders#helps seperate the data in the dataloader much more efficiently

#model builder(replaces head, optional freezing of backbone)
def build_model(num_classes: int, freeze_backbone: bool = False) -> nn.Module:#num_classes: number of output classes(amount of fish species), freeze_backbone: if true, the weights of the convolutional layers will not be used, good for small datasets
    weights = ResNet50_Weights.IMAGENET1K_V2 #loads a pretrained weight that is better for image proccessing
    model = resnet50(weights=weights) #gives me the resnet50 model with convolutional layers and a fc layer with space for 1000 classes

    if feeze_backbone:#stops the weights of the convolutional layers from being updated during training(except the final layer), better for small datasets
        for p in model.parameters():#model.parameters() represents all the weights in the model
            p.requires_grad = False#tells pytorch to not update the weights

    in_features = model.fc.in_features #gets the number of imput features for the final layer(2048)
    model.fc = nn.Linear(in_features, num_classes) #replaces the final layer that has 1000 outputs with a new layer that has the amount of outputs that equals the number of classes in the dataset
    return model



#trains one epoch
#its point is to encapsulate one full pass through the training dataset
def train_one_epoch(model, loader, criterion, optimizer, device):#model: the neural network model to be trained; loader: the dataloader that provides the training data in mini batches; criterion: the loss function used the measure the models performance; optimizer: the optimization algorithm used to update the model weights; device: the computing device(GPU,CPU) that will be used for training
    model.train()#enables training mode, which in turn enables dropout and batchnorm
    total_loss, total_correct, total_seen = 0.0, 0, 0#is used to calculate the avrage loss and accuracy;total_loss: is a python float, accumulates the average loss as a real number; total_correct: is an integer, accumulates the number of correct predictions; total_seen: is an integer, accumulates the total number of samples processed

    for images, targets in loader:#iterates over minin batches of data from the dataloader
        images, targets = images.to(device), targets.to(device)#transfers the batch to the compute device I set(GPU), helps speed up the training by moving batches in and out of the GPU

        optimizer.zero_grad(set_to_none=True) #clears the gradients from all the tensors(stores weights, inputs, outputs...), set_to_none=True is better for memory efficiency
        outputs = model(images) #sends a batch of image tensors through the model, it returns logits(a raw score for each class)
        loss = criterion(outputs, targets) #calculates the loss between the model outputs and the true labels
        loss.backward() #computes the gradients of the loss
        optimizer.step() #updates the model weights based on the computed gradients

        #updates the total loss and accuracy
        preds = outputs.argmax(dim=1) #gets the index of the highest logit across the whole class, represents the predicted class for each image
        total_correct += (preds == targets).sum().item() #compares the predicted classes with the true labels, sums the number of correct predictions, and adds it to total_correct

        #continues updating the total loss and accuracy
        bs = images.size(0) #gets the batch size(number of images in current batch)
        total_seen += bs #adds the batch size to total_seen
        total_loss += loss.item() * bs #add the loss for the current batch to total_loss

    #final calculations for the eopch
    avg_loss = total_loss / max(total_seen, 1) #calculates the average loss across all samples, max is used to avoid division by zero
    avg_acc = total_correct / max(total_seen, 1) #calculates the average accuracy across all samples
    return avg_loss, avg_acc



#evaluation loop; measures how well the model generalizes without changing the weights(no optmizer step)
@torch.no_grad()#disables gradient calculation, which reduces memory consumption
def evaluate(model, loader, criterion, device): #model: the cnn model to be evaluated; loader: the dataloader that provides the training data; criterion: the loss function used to measure the model's performance; device: the computing device(GPU,CPU) that will be used for evalutaion
    model.eval() #enables evaluation mode, disables dropout and batchnorm
    total_loss, total_correct, total_seen = 0.0, 0, 0 #used to calculate the average loss and accuracy

    #iterates over mini batches of data from the dataloader
    for images, targets in loader: #goes through the dataloader in mini batches
        images, targets = images.to(device),targets.to(device) #moves the batch to the compute device
        outputs = model(images) #sends a batch of image tensors through the model, returns logits(a raw score)
        loss=criterion(outputs, targets) #calculates the loss between the model outputs and the true labels

        #updates the total loss and accuracy
        preds = outputs.argmax(dim=1) #gets the index of the highest logit across the whole class, represents the predicted class for each image
        total_correct += (preds == targets).sum().item() #compares the predicted classes with the true labels, sums the number of correct predictions and adds it to toal_correct

        #continues updating the total loss and accuracy
        bs=image.size(0) #gets the batch size(num of images in batch)
        total_seen += bs #adds the batch size to total_seen
        total_loss += loss.item() * bs #adds the loss for the current batch to total_loss

    #final calculations for epoch
    avg_loss = total_loss/max(total_seen, 1)#calculates the average loss across all samples, max is used to not divide by zero
    avg_acc = total_correct/max(total_seen, 1) #calculates the avrerage accuracy across all samples
    return avg_loss, avg_acc


#saves the model checkpoint(freezes the current state of everything)
def save_checkpoint(path,model,optimizer,epoch,best_val_acc,class_names): #path: file path to save the checkpoint; model: the cnn model to be saved; optimizer: saves the current state of the optimizer; epoch: the current(last completed) epoch number; best_val_acc: the best validation accuracy achieved so far; class_names: list of names of the classes in the dataset(names of the fish)
    os.makedirs(os.path.dirname(path), exist_ok=True) #ensures the directory for the checkpoint file exists, creates it if it doesnt
    payload = {
        "epoch": epoch, #last completed epoch number
        "model_state": model.state_dict(), #saves model weights
        "optimizer_state": optimizer.state_dict(), #saves optimizer state
        "best_val_acc": best_val_acc, #best validation accuracy
        "class_names": class_names, #list of class names
    }
    torch.save(payload, path) #saves the payload dictionary to the specified path



#
def main():
    #---CLI---(command line interface)
    ap = argparse.ArgumentParser(description="Fine tune ResNet50 on an ImageFolder fish dataset")#creates an argument parser object that will handle command line arguments
    ap.add_argument("--date-root", type=str, default="data/fish", help="Folder with train/val/test")#path to the dataset folder, contains subfolders for training, validation and testing data
    ap.add_argument("--image-size", type=int, default=224, help="Model input size (ResNet50=224)")#size of input images, ResNet50 expects 224x224 images
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size") #number of samples processed before the weights are updates
    ap.add_argument("--epochs", type=int, default=15, help="Number of training epochs") #number of times the entire training dataset is passed through the model
    ap.add_argument("--lr", type=float, default=3e-4, help="Learning rate (AdamW)") #step size at each iteration while moving toward a minimum of the loss function
    ap.add_argument("--weight-decay", type=float, default=0.05, help="AdamW weight decay") #regularation term to prevent overfitting
    ap.add_argument("--freeze-backbone", action="store_true", help="Train only the final head(better for small datasets)") #if set, only the final layer will be trained
    ap.add_argument("--out-dir", type=str, default="checkpoints", help="Where to save models") #directory to save model checkpoints
    args= ap.parse_args() #parses the command line arguments and stores them in the args variable

    #---device and data---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #sets the computing device to GPU if available, otherwise uses CPU

    train_loader, val_loader, test_loader, class_names = build_loaders(
        root=args.data_root, img_size=args.img_size, batch_size=args.batch_size
    )#returns three dataloaders ready for training, and a list of class names

    #---model, loss, optimizer---
    model=build_model(num_classes=len(class_names), freeze_backbone=args.freeze_backbone).to(device) #builds the model and transfers it to the computing device
    criterion = nn.CrossEntropyLoss() #loss function for multi-class classification problems
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)#AdamW optimzer with specified learning rate and weight decay

    #---training loop---
    