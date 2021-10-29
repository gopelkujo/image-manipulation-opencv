#############################
# Mochamad Nauval Dwisatya  #
# 191524023                 #
#############################

# NEEDED TO BE UPDATED
# - Load gif in canvas
# - Access pixel

# import needed package
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import os
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
# experiment
import imageio

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
    # !!! experiment
    if(extension == '.gif'):
        gif = imageio.mimread(filename)
        nums = len(gif)
        print('nums:', nums)

        # imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in gif]
        imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in gif]
        
        i = 0

        while True:
            cv2.imshow('gif', imgs[i])
            if(cv2.waitKey(100) == 27):
                break
            i = (i+1)%nums
            print(i)
        cv2.destroyAllWindows()

def showImg(img):
    global canvas

    # show image
    canvas.create_image((canvas.winfo_height()/2, canvas.winfo_width()/2), anchor=CENTER, image=img)
    canvas.image = img

# function for manipulate image
def changeColor():
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

def resetColor():
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

# create menubar
menubar = Menu(root)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label='Open', command=openImg)
filemenu.add_command(label='Show')

# separate list menu
filemenu.add_separator()

filemenu.add_command(label='Exit', command=root.quit)

menubar.add_cascade(label='File', menu=filemenu)

# add menubar to root
root.config(menu=menubar)

# create canvas for image place
canvas = Canvas(root, width=550, height=550, background='white')
canvas.pack()

# create button to trigger image manipulation
imgBtn = Button(text='Manipulate!', padx=100, command=changeColor).pack(side=LEFT)

# create reset manipulate image button
resetBtn = Button(text='Reset!', padx=100, command=resetColor).pack(side=RIGHT)

# make window stay
root.mainloop()