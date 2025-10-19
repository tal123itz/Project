import os #for file path
import torch #for tensor operations/picking a device
import torch.nn as nn #brings neural network modules
from torchvision.models import resnet50 #imports the resnet50 architecture
from dataloaders import build_loaders #imports the function that builds dataloaders

#configuration constants
DATA_ROOT  = "Dataset"   # folder containing train/val/test subfolders
IMG_SIZE   = 224 #image size expected by ResNet50
BATCH_SIZE = 32 #number of samples per batch

BEST_PATH  = "checkpoints/best.pt" #path to save the best model
LAST_PATH  = "checkpoints/last.pt" #path to save the last model

#evaluation function(no gradient calculation needed)
@torch.no_grad() #disables gradient calculation for efficiency
def evaluate(model, loader, device): #a utility that scores a model on a given dataset(loader) with a specified device(CPU/GPU)
    model.eval() #sets the model to evaluation mode(disables dropout, batchnorm updates, etc)
    criterion = nn.CrossEntropyLoss() #loss function for multi-class classification with logits

    total_loss, total_correct, total_seen = 0.0, 0, 0 #initializes accumulators for loss, correct predictions, and total samples
    for images, targets in loader: #iterates over batches from the dataloader
        images, targets = images.to(device), targets.to(device) #moves data(tensors) to the specified device(CPU/GPU)
        outputs = model(images) #forwards the images through the model to get predictions(logits)
        loss = criterion(outputs, targets) #computes the batch loss

        preds = outputs.argmax(dim=1) #counts the correct predictions of the batch
        total_correct += (preds == targets).sum().item() #counts the correct predictions of the batch

        bs = images.size(0) #gets the batch size
        total_seen += bs #updates the total number of samples seen
        total_loss += loss.item() * bs #accumulates the total sum of losses

    avg_loss = total_loss / max(total_seen, 1) #computes average loss over all samples
    avg_acc  = total_correct / max(total_seen, 1) #computes average accuracy over all samples
    return avg_loss, avg_acc #returns average loss and accuracy

#rebuilds the model head for inference
def build_model_for_inference(num_classes: int): #creates a ResNet50 model with the correct number of output classes
    model = resnet50(weights=None) #builds the architecture without pretrained weights
    in_features = model.fc.in_features #gets the number of penultimate(before final) layer features
    model.fc = nn.Linear(in_features, num_classes) #replaces the 1000 class final layer with a new layer for num_classes
    return model #returns the modified model

#main
def main():
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #chooses GPU if available, otherwise CPU

    # test loader + class names from dataloaders.py
    _, _, test_loader, class_names = build_loaders( 
        root="Dataset", img_size=IMG_SIZE, batch_size=BATCH_SIZE
    ) #builds only the test loader. useful for evaluation

    # choose checkpoint (prefer best, fallback to last)
    ckpt_path = BEST_PATH if os.path.exists(BEST_PATH) else LAST_PATH #prefers best model checkpoint
    if not os.path.exists(ckpt_path): ##if neither checkpoint exists, raises an error
        raise FileNotFoundError("No checkpoint found in 'checkpoints/'. Train the model first.")

    # load checkpoint; prefer class_names stored in it (ensures exact label order)
    ckpt = torch.load(ckpt_path, map_location=device) #loads the checkpoint
    class_names = ckpt.get("class_names", class_names) #uses class names from checkpoint if available

    # rebuild model with correct head size, load weights, move to device
    model = build_model_for_inference(len(class_names)) #creates a ResNet50 model with the correct number of output classes
    model.load_state_dict(ckpt["model_state"]) #loads the trained weights into the model
    model.to(device) #moves the model to the specified device(CPU/GPU)

    # evaluate on test set
    test_loss, test_acc = evaluate(model, test_loader, device) #runs evaluation to get metrics
    print(f"Test: loss {test_loss:.4f}, acc {test_acc:.4f}") #prints the test loss and accuracy
    print("Classes:", class_names) #print the class names

#entry point
if __name__ == "__main__":
    main()