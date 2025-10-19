from tkinter import *
from tkinter import Tk,Button,Label#button is for the buttons used, label is for the image
from PIL import Image, ImageTk #Pillow
from tkinter import filedialog#is needed to download files from computer

import torch #main pytorch library; required for tensors and model handling
import torch.nn as nn #neaural network module
from torchvision import transforms #for image transformations
from torchvision.models import resnet50 #renet50 architecture

#absolute path to the inference model
import os #for file paths
ROOT = os.path.dirname(os.path.abspath(__file__)) #gets the root directory of the project(UI.py location)
INFER_PATH = os.path.join(ROOT, "checkpoints", "model_fish_inference.pt") #builds the path to the inference model

IMG_SIZE = 224 #image size expected by ResNet50

#constants that ResNet50 models are typically trained on (R, G, B)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

#preprocessing pipeline for input images/validation
preprocess = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)), #resizes the shorter side a little larger than the target/img_size
    transforms.CenterCrop(IMG_SIZE), #crops the center to the exact img_size size
    transforms.ToTensor(), #converts image to tensor and scales pixel values to [0,1]
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD), #normalizes tensor to standard values to match imagenet stats
])

#loads the inference model/trained model 
def load_inference_model():
    """
    Load the compact checkpoint (weights + class_names),
    rebuild ResNet50's final layer to the correct size, and return (model, class_names, device).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #chooses GPU if available, otherwise CPU
    blob = torch.load(INFER_PATH, map_location=device)  #loads the saved dictionary with model weights and class names
    class_names = blob["class_names"] #preserves the exact label order used during training

    model = resnet50(weights=None)  #builds the architecture without pretrained weights
    model.fc = nn.Linear(model.fc.in_features, len(class_names)) #replaces the 1000 class head/final layer with a new layer for the correct number of classes
    model.load_state_dict(blob["model_state"]) #loads the trained weights into the model
    model.eval().to(device) #sets the model to evaluation mode and moves it to the specified device(CPU/GPU); turns off dropout, batchnorm updates
    return model, class_names, device

#small prediction helper
@torch.no_grad()#disables gradient calculation for efficiency
def predict_image(model, class_names, device, image_path, topk=3): 
    """
    Open an image, apply eval transforms, run a forward pass, and return top-k [(name, prob), ...].
    """
    img = Image.open(image_path).convert("RGB")  # ensure 3 channels
    x = preprocess(img).unsqueeze(0).to(device) #adds a batch dimension and moves to device
    probs = model(x).softmax(dim=1).squeeze(0)  #applies softmax to get probabilities and removes batch dimension to get a 1D tensor
    values, indices = probs.topk(topk) #gets the top-k probabilities and their corresponding class indices because we want the most likely predictions
    return [(class_names[i], float(v)) for v, i in zip(values, indices)] #builds a list of tuples with class names and their probabilities and returns it


# ---- load model once at startup ----
model, class_names, device = load_inference_model()



def loadImage():
    file_path = filedialog.askopenfilename(#opens the file selection dialog
        title="Select an Image",#name of the dialog window
        filetypes=(("Image files", "*.png;*.jpg;*.jpeg;*.gif"), ("All files", "*.*"))#file types that can be downloaded
        #change file types later
    )
    if file_path:#checks if a file was selected
        img=Image.open(file_path)#opens the image with pillow
        img=img.resize((250,250))#resizes the image to fit
        #change resize later to whatever fits best
        photo=ImageTk.PhotoImage(img)#converts the image to a photo that tkinter can use
        label.config(image=photo)#tells the label to use the image
        label.image=photo#keeps a reference of the image so that it doesnt get deleted(garbage collected)

        #get predictions
        results = predict_image(model, class_names, device, file_path, topk=3) #gets the top 3 predictions for the selected image
        lines = [f"{i+1}) {name}: {prob:.2f}" for i, (name, prob) in enumerate(results)] #formats the predictions nicly into lines
        result_label.config(text="\n".join(lines)) #shows the predictions in the result label


root = Tk()#creates a tkinter window
root.title("Fish Identifier")
#root.iconbitmap(...)#add icon later
root.geometry("400x400")#sets the size of the window


button = Button(root, text="Load Image", command=loadImage)#creates a button inside the root 
button.pack(pady=10)#adds the button into the window with vertical padding

label = Label(root)#creates a lebel inside the root that will hold the image
label.pack#adds the label(image) into the window

#text label for predictions
result_label = Label(root, text="", justify="left", font=("Segoe UI", 10)) #creates a label to display prediction results
result_label.pack(pady=8) #adds the result label into the window with vertical padding


root.mainloop()#makes the tkinter window show up