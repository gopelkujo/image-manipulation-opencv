#############################
# Mochamad Nauval Dwisatya  #
# 191524023                 #
#############################

# NEEDED TO BE UPDATED
# - Refactor code
# - Load gif in canvas
# - UI (Make 2 canvas in 1 window for before after, label mode, button convert)

# import needed package
import cv2, os, imageio, numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM

# create window
root = Tk()
root.title("Image Manipulation")
root.geometry("450x450")
filename = ''

# function to open image
def openImg():
    global filename

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
        img.thumbnail((400, 400), Image.ANTIALIAS)
        imgfile = ImageTk.PhotoImage(image=img)
        showImg(imgfile)
    
    # open gif file
    if(extension == '.gif'):
        global canvas
        canvas.delete('all') # clean canvas
        gif = imageio.mimread(filename)
        nums = len(gif)

        imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in gif]
        #imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in gif]
        
        i = 0

        while True:
            cv2.imshow('gif', imgs[i])
            if(cv2.waitKey(100) == 27):
                break
            i = (i+1) % nums
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
        imarray.thumbnail((400, 400), Image.ANTIALIAS)
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
        imarray.thumbnail((400, 400), Image.ANTIALIAS)
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
    # cv2.imwrite('C:/Users/Gopel/Pictures/pel-keren.jpg', quantineImage(image,5))
    image = cv2.resize(image, (400,400), interpolation = cv2.INTER_AREA)

    # create new windows
    newWindow = Toplevel(root)
    newWindow.title('input')
    newWindow.geometry('200x30')
    entry1 = Entry(newWindow)
    entry1.pack(side=LEFT)

    def getInput():
        quant = int(entry1.get())
        newWindow.destroy()
        cv2.imshow('Quantizationed', quantineImage(image, quant))
        cv2.waitKey(0)
        # imarray = Image.fromarray(quantineImage(image, quant))
        # imarray.thumbnail((400, 400), Image.ANTIALIAS)
        # imnp = np.asarray(imarray)
        # photo = ImageTk.PhotoImage(image = Image.fromarray(imnp))
        # showImg(photo)

    button1 = Button(newWindow, text='Process', command=getInput)
    button1.pack(side=RIGHT)
        

def quantineImage(image, k):
    i = np.float32(image).reshape(-1,3)
    condition = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,20,1.0)
    _,label,center = cv2.kmeans(i, k , None, condition,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    final_img = center[label.flatten()]
    final_img = final_img.reshape(image.shape)
    return final_img

def samplingImage():
    global filename
    cvimg = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    
    # create new windows
    newWindow = Toplevel(root)
    newWindow.title('input')
    newWindow.geometry('200x30')
    entry1 = Entry(newWindow)
    entry1.pack(side=LEFT)

    def getInput():
        input = int(entry1.get())
        newWindow.destroy()
        cvblur = cv2.blur(cvimg, (input, input)) # example blur in range 50
        imarray = Image.fromarray(cvblur)
        imarray.thumbnail((400, 400), Image.ANTIALIAS)
        imnp = np.asarray(imarray)
        photo = ImageTk.PhotoImage(image = Image.fromarray(imnp))
        showImg(photo)

    button1 = Button(newWindow, text='Process', command=getInput)
    button1.pack(side=RIGHT)
    

def incIntensity():
    global filename
    cvimg = cv2.imread(filename)
    (rows,cols, _) = cvimg.shape

    def countRGB(value):
        for i in range(rows):
            for j in range(cols):
                (r,g,b) = cvimg[i,j]
                r = r + value
                if(r > 255): r = 255
                g = g + value
                if(g > 255): g = 255
                b = b + value
                if(b > 255): b = 255
                cvimg[i,j] = [r, g, b]

        b,g,r = cv2.split(cvimg)
        imgmerge = cv2.merge((r,g,b))
        imarray = Image.fromarray(imgmerge)
        # resize image with ratio
        imarray.thumbnail((400, 400), Image.ANTIALIAS)
        imgfile = ImageTk.PhotoImage(image=imarray)
        showImg(imgfile)

    # create new windows
    newWindow = Toplevel(root)
    newWindow.title('input')
    newWindow.geometry('200x30')
    entry1 = Entry(newWindow)
    entry1.pack(side=LEFT)

    def getInput():
        input = int(entry1.get())
        newWindow.destroy()
        countRGB(input)

    button1 = Button(newWindow, text='Process', command=getInput)
    button1.pack(side=RIGHT)
    

def decIntensity():
    global filename
    cvimg = cv2.imread(filename)
    (rows,cols, _) = cvimg.shape

    def countRGB(value):
        for i in range(rows):
            for j in range(cols):
                (r,g,b) = cvimg[i,j]
                r = r - value
                if(r < 0): r = 0
                g = g - value
                if(g < 0): g = 0
                b = b - value
                if(b < 0): b = 0
                cvimg[i,j] = [r, g, b]

        b,g,r = cv2.split(cvimg)
        imgmerge = cv2.merge((r,g,b))
        imarray = Image.fromarray(imgmerge)
        # resize image with ratio
        imarray.thumbnail((400, 400), Image.ANTIALIAS)
        imgfile = ImageTk.PhotoImage(image=imarray)
        showImg(imgfile)

    # create new windows
    newWindow = Toplevel(root)
    newWindow.title('input')
    newWindow.geometry('200x30')
    entry1 = Entry(newWindow)
    entry1.pack(side=LEFT)

    def getInput():
        input = int(entry1.get())
        newWindow.destroy()
        countRGB(input)

    button1 = Button(newWindow, text='Process', command=getInput)
    button1.pack(side=RIGHT)

def klise():
    global filename
    cvimg = cv2.imread(filename)
    (rows,cols, _) = cvimg.shape

    for i in range(rows):
        for j in range(cols):
            (r,g,b) = cvimg[i,j]
            r = 255 - r
            if(r > 255): r = 255
            if(r < 0): r = 0
            g = 255 - g
            if(g > 255): g = 255
            if(g < 0): g = 0
            b = 255 - b
            if(b > 255): b = 255
            if(b < 0): b = 0
            cvimg[i,j] = [r, g, b]

    b,g,r = cv2.split(cvimg)
    imgmerge = cv2.merge((r,g,b))
    imarray = Image.fromarray(imgmerge)
    # resize image with ratio
    imarray.thumbnail((400, 400), Image.ANTIALIAS)
    imgfile = ImageTk.PhotoImage(image=imarray)
    showImg(imgfile)

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
editmenu.add_command(label='Sampling', command=samplingImage)
editmenu.add_command(label='Increase Intensity', command=incIntensity)
editmenu.add_command(label='Decrease Intensity', command=decIntensity)
editmenu.add_command(label='Klise', command=klise)

# add cascade
menubar.add_cascade(label='File', menu=filemenu)
menubar.add_cascade(label='Edit', menu=editmenu)

# add menubar to root
root.config(menu=menubar)

# create canvas for image place
canvas = Canvas(root, width=400, height=400, background='white')
canvas.pack()

# make window stay
root.mainloop()