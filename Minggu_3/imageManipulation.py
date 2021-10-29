#############################
# Mochamad Nauval Dwisatya  #
# 191524023                 #
#############################

# NEEDED TO BE UPDATED
# - Load gif in canvas
# - Access pixel
# - Sampling
# - Color Quantization

# import needed package
import cv2, os, imageio, numpy as np, argparse
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from sklearn.cluster import MiniBatchKMeans

# create window
root = Tk()
root.title("Image Manipulation")
root.geometry("600x600+10+20")
filename = ''

# function to open image
def openImg():
    filetypes = (
        ('JPG', '*.jpg'),
        ('JPEG', '*.jpeg'),
        ('JPEG-2000', '*.jp2'),
        ('PNG', '*.png'),
        ('TIFF', '*.tiff'),
        ('SVG', '*.svg'),
        ('GIF', '*.gif'),
        ('BMP', '*.bmp'),
        ('All files', '*.*')
    )
    global filename
    filename = filedialog.askopenfilename(title='Open a file', filetypes=filetypes)
    print(filename)
    # read format files
    _, extension = os.path.splitext(filename)
    print(extension)

    if(filename != '' and extension != '.gif'):
        # check format files
        if(extension == '.svg'):
            img = svg2png()
        else:
            img = Image.open(filename)
        img.thumbnail((550, 550), Image.ANTIALIAS)
        imgfile = ImageTk.PhotoImage(image=img)
        showImg(imgfile)
    
    # open gif file
    if(extension == '.gif'):
        gif = imageio.mimread(filename)
        nums = len(gif)
        print('nums:', nums)

        imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in gif]
        #imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in gif]
        
        i = 0

        while True:
            cv2.imshow('gif', imgs[i])
            if(cv2.waitKey(100) == 27):
                break
            i = (i+1) % nums
            print(i)
        cv2.destroyAllWindows()

def showImg(img):
    global canvas

    # show image
    canvas.create_image((canvas.winfo_height()/2, canvas.winfo_width()/2), anchor=CENTER, image=img)
    canvas.image = img

# function for manipulate image
def toGrey():
    global filename
    _, extension = os.path.splitext(filename)
    
    if(filename != ''):
        # check format files
        if(extension == '.svg'):
            cvimg = cv2.imread('temp.png')
        else:
            cvimg = cv2.imread(filename)
        # change image to grey
        cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2GRAY)
        cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)

        b,g,r = cv2.split(cvimg)
        imgmerge = cv2.merge((r,g,b))
        imarray = Image.fromarray(imgmerge)
        # resize image with ratio
        imarray.thumbnail((550, 550), Image.ANTIALIAS)
        imgfile = ImageTk.PhotoImage(image=imarray)
        showImg(imgfile)

def resetImg():
    global filename
    _, extension = os.path.splitext(filename)
    
    if(filename != ''):
        # check format files
        if(extension == '.svg'):
            cvimg = cv2.imread('temp.png')
        else:
            cvimg = cv2.imread(filename)
        b,g,r = cv2.split(cvimg)
        imgmerge = cv2.merge((r,g,b))
        imarray = Image.fromarray(imgmerge)
        # resize image with ratio
        imarray.thumbnail((550, 550), Image.ANTIALIAS)
        imgfile = ImageTk.PhotoImage(image=imarray)
        showImg(imgfile)

def svg2png():
    drawing = svg2rlg(filename)
    renderPM.drawToFile(drawing, "temp.png", fmt="PNG")
    img = renderPM.drawToPIL(drawing)
    return img

def colorQuantization():
    global filename
    image = cv2.imread(filename)
    cv2.imwrite('C:/Users/Gopel/Pictures/pel-keren.jpg', quantineImage(image,5))
    image = cv2.resize(image, (550,550), interpolation = cv2.INTER_AREA)
    cv2.imshow('Quantizationed', quantineImage(image,5))
    cv2.waitKey(0)

def quantineImage(image, k):
    i = np.float32(image).reshape(-1,3)
    condition = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,20,1.0)
    _,label,center = cv2.kmeans(i, k , None, condition,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    final_img = center[label.flatten()]
    final_img = final_img.reshape(image.shape)
    return final_img

# create menubar
menubar = Menu(root)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label='Open', command=openImg)
filemenu.add_command(label='Show')

# separate list menu
filemenu.add_separator()

filemenu.add_command(label='Exit', command=root.quit)

# create edit menu
editmenu = Menu(menubar, tearoff=0)
editmenu.add_command(label='Reset', command=resetImg)
editmenu.add_command(label='2Grey', command=toGrey)
editmenu.add_command(label='Quantization', command=colorQuantization)

# add cascade
menubar.add_cascade(label='File', menu=filemenu)
menubar.add_cascade(label='Edit', menu=editmenu)

# add menubar to root
root.config(menu=menubar)

# create canvas for image place
canvas = Canvas(root, width=550, height=550, background='white')
canvas.pack()

# make window stay
root.mainloop()