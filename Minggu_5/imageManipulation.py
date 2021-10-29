#############################
# Mochamad Nauval Dwisatya  #
# 191524023                 #
#############################

# NEEDED TO BE UPDATED
# - Refactor code
# - Load gif in canvas

# import needed package
import cv2, os, imageio, numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage import exposure

# set up window
root = Tk()
root.title("Image Manipulation")
root.geometry("650x450")
filename = ''
mode_status = ''

# function to open image
def openImg():
    global filename

    print('[INFO] Selecting files..')

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
    print('[INFO] File path is ' + str(filename))
    showImg()

def showImg():
    # global canvas_original
    global filename
    global canvas_original

    # read format files
    _, extension = os.path.splitext(filename)

    if(filename != '' and extension != '.gif'):
        # check format files
        if(extension == '.svg'):
            img = svg2png()
        else:
            img = Image.open(filename)
        img.thumbnail((250, 250), Image.ANTIALIAS)
        imgfile = ImageTk.PhotoImage(image=img)
        # showImg(imgfile)
        canvas_result.delete('all')
        # insert image to canvas_original
        canvas_original.create_image((canvas_original.winfo_height()/2, canvas_original.winfo_width()/2), anchor=CENTER, image=imgfile)
        canvas_original.image = imgfile

        
    # open gif file
    if(extension == '.gif'):
        canvas_original.delete('all') # clean canvas
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

def showResult(img):
    global canvas_result
    global img_result
    global mode_text
    global mode_status
    mode_text.set(mode_status)
    img_result = img
    img_img = ImageTk.getimage(img)
    img_img.save('temp_result.png')

    # show image
    canvas_result.create_image((canvas_result.winfo_height()/2, canvas_result.winfo_width()/2), anchor=CENTER, image=img)
    canvas_result.image = img

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
        imarray.thumbnail((250, 250), Image.ANTIALIAS)
        imgfile = ImageTk.PhotoImage(image=imarray)
        showImg(imgfile)

# function for manipulate image
def toGrey():
    global filename
    global mode_status
    mode_status = 'Gray Scale'
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
        imarray.thumbnail((250, 250), Image.ANTIALIAS)
        imgfile = ImageTk.PhotoImage(image=imarray)
        showResult(imgfile)

def svg2png():
    drawing = svg2rlg(filename)
    renderPM.drawToFile(drawing, "temp.png", fmt="PNG")
    img = renderPM.drawToPIL(drawing)
    return img

def colorQuantization():
    global filename
    global canvas_result
    global mode_status
    mode_status = 'Quantization'
    image = cv2.imread(filename)
    image = cv2.resize(image, (250,250), interpolation = cv2.INTER_AREA)

    # create new windows
    newWindow = Toplevel(root)
    newWindow.title('input')
    newWindow.geometry('200x30')
    entry1 = Entry(newWindow)
    entry1.pack(side=LEFT)

    def getInput():
        quant = int(entry1.get())
        newWindow.destroy()
        imarray = Image.fromarray(quantineImage(image, quant))
        imarray.thumbnail((250, 250), Image.ANTIALIAS)
        imnp = np.asarray(imarray)
        photo = ImageTk.PhotoImage(image = Image.fromarray(imnp))
        showResult(photo)

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
    global mode_status
    mode_status = 'Sampling'
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
        imarray.thumbnail((250, 250), Image.ANTIALIAS)
        imnp = np.asarray(imarray)
        photo = ImageTk.PhotoImage(image = Image.fromarray(imnp))
        showResult(photo)

    button1 = Button(newWindow, text='Process', command=getInput)
    button1.pack(side=RIGHT)

def incIntensity():
    global filename
    global mode_status
    mode_status = 'Increase Intensity'
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
        imarray.thumbnail((250, 250), Image.ANTIALIAS)
        imgfile = ImageTk.PhotoImage(image=imarray)
        showResult(imgfile)

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
    global mode_status
    mode_status = 'Decrease Intensity'
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
        imarray.thumbnail((250, 250), Image.ANTIALIAS)
        imgfile = ImageTk.PhotoImage(image=imarray)
        showResult(imgfile)

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
    global mode_status
    mode_status = 'Klise'
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
    # save image result as temp
    # imarray.save('temp_result.jpg')

    # resize image with ratio
    imarray.thumbnail((250, 250), Image.ANTIALIAS)
    imgfile = ImageTk.PhotoImage(image=imarray)
    showResult(imgfile)

# show histogram of the image
def showHistogram():
    img = cv2.imread('temp_result.png')
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    # canvas_result.pack_forget()
    plt.show()

def showHisEqual():
    global img_result
    img_img = ImageTk.getimage(img_result).convert('RGB')
    img_img = np.array(img_img)
    img_img = img_img[:, :, ::-1].copy()
    img_img = cv2.cvtColor(img_img, cv2.COLOR_BGR2GRAY)
    equalized_img = cv2.equalizeHist(img_img)
    equalized_img = cv2.cvtColor(equalized_img, cv2.COLOR_BGR2RGB)
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([equalized_img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    # canvas_result.pack_forget()
    plt.show()

def showHisSpec():
    global filename
    global img_result
    print("[INFO] loading source and reference images...")
    src = cv2.imread(filename)
    ref = ImageTk.getimage(img_result).convert('RGB')
    ref = np.array(ref)
    ref = ref[:, :, ::-1].copy()
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
    # determine if we are performing multichannel histogram matching
    # and then perform histogram matching itself
    print("[INFO] performing histogram matching...")
    multi = True if src.shape[-1] > 1 else False
    matched = exposure.match_histograms(src, ref, multichannel=multi)
    # show the output images
    # cv2.imshow("Source", src)
    # cv2.imshow("Reference", ref)
    # cv2.imshow("Matched", matched)
    # cv2.waitKey(0)
    
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([matched],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    # canvas_result.pack_forget()
    plt.show()

# create menubar
menubar = Menu(root)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label='Open', command=openImg)

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
editmenu.add_command(label='Histogram', command=showHistogram)
editmenu.add_command(label='Histogram Equalization', command=showHisEqual)
editmenu.add_command(label='Histogram Specification', command=showHisSpec)

# add cascade
menubar.add_cascade(label='File', menu=filemenu)
menubar.add_cascade(label='Edit', menu=editmenu)

# add menubar to root
root.config(menu=menubar)

# add label to root
mode_text = StringVar()
mode_text.set('Mode Status')
label = Label(root, textvariable=mode_text, fg = "black", font = "Times").pack()

# create canvas for original image
canvas_original = Canvas(root, width=250, height=250, background='white')
canvas_original.pack(side=LEFT)

# create canvas for result image
canvas_result = Canvas(root, width=250, height=250, background='white')
canvas_result.pack(side=RIGHT)

# make window stay
root.mainloop()