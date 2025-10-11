from tkinter import *
from tkinter import Tk,Button,Label#button is for the buttons used, label is for the image
from PIL import Image, ImageTk #Pillow
from tkinter import filedialog#is needed to download files from computer


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


root = Tk()#creates a tkinter window
root.title("Fish Identifier")
#root.iconbitmap(...)#add icon later
root.geometry("400x400")#sets the size of the window


button = Button(root, text="Load Image", command=loadImage)#creates a button inside the root 
button.pack(pady=10)#adds the button into the window with vertical padding

label = Label(root)#creates a lebel inside the root that will hold the image
label.pack#adds the label(image) into the window


root.mainloop()#makes the tkinter window show up