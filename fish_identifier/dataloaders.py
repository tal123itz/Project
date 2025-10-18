#imports
import os #for file path operations
import torch #for tensor operations (checks if CUDA exists)
from torch.utils.data import DataLoader #for creating data loaders(batches dataset samples into minibatches)
from torchvision import datasets, transforms #for dataset handling and transformations (reads images from class folders/ defines augmentations)

#constants that ResNet50 models are typically trained on (R, G, B)
IMAGENET_MEAN = [0.485, 0.456, 0.406] #for normalizing images
IMAGENET_STD  = [0.229, 0.224, 0.225]

#helper function that builds a pipeline for validation
def _train_transform(img_size=224): #helper that builds a pipeline for training (default sizes match resnet50)
    """
    Data augmentation + normalization for training.
    """
    return transforms.Compose([ #chains a list of operations together
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)), #crops a random portion of the image and resizes it to img_size(combats overfitting)
        transforms.RandomHorizontalFlip(0.5), #flips image horizontally(useful for many fish images)
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05), #small random changes in brightness, contrast, saturation, hue (to simulate different lighting)
        transforms.ToTensor(), #converts image to tensor and scales pixel values to [0,1]
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD), #normalizes tensor to standard values to match imagenet stats
    ])

#evaluation transform pipeline
def _eval_transform(img_size=224): #helper that builds a pipeline for evaluation with no randomness
    """
    Deterministic preprocessing for validation and test.
    """
    return transforms.Compose([ #chains a list of operations together
        transforms.Resize(int(img_size * 1.14)), #resizes the shorter side a little larger than the target/img_size
        transforms.CenterCrop(img_size), #crops the center to the exact img_size size
        transforms.ToTensor(), #converts image to tensor and scales pixel values to [0,1]
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD), #normalizes tensor to standard values to match imagenet stats
    ])

#building the dataloaders
def build_loaders(root="data/fish", img_size=224, batch_size=32): #main function to build dataloaders
    """
    Returns:
      train_loader, val_loader, test_loader, class_names
    Expects directory structure:
      {root}/train/<class>/*.jpg
      {root}/val/<class>/*.jpg
      {root}/test/<class>/*.jpg
    """ #builds and returns dataloaders for training, validation, and testing

    #creates datasets for training, validation, and testing using ImageFolder
    train_ds = datasets.ImageFolder(os.path.join(root, "train"), transform=_train_transform(img_size))
    val_ds   = datasets.ImageFolder(os.path.join(root, "val"),   transform=_eval_transform(img_size))
    test_ds  = datasets.ImageFolder(os.path.join(root, "test"),  transform=_eval_transform(img_size))
    class_names = train_ds.classes #gets class names from training dataset

    #creates dataloaders for each dataset/loader settings
    num_workers = min(8, os.cpu_count() or 2) #number of subprocesses for data loading(avoids overscrubbing CPU)
    pin = torch.cuda.is_available() #if CUDA (GPU) is available, use it

    #data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, #shuffles data each epoch(improves SGD training)(SGD = Stochastic Gradient Descent)
                            num_workers=num_workers, pin_memory=pin, drop_last=True) #drop_last=True to ensure all batches are the same size(important for batch norm layers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, #the rest dont shuffle because evaluation doesn't need it
                            num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    
    