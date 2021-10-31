#############################
# Mochamad Nauval Dwisatya  #
# 191524023                 #
#############################

# NEEDED TO BE UPDATED
# - Refactor code
# - Load histogram in canvas

# import needed package
import cv2, tkinter, os, numpy as np
from PIL import ImageTk, Image
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from matplotlib import pyplot as plt
from skimage import exposure

# set up window
root = tkinter.Tk()
root.title('Image Manipulation')
root.geometry('650x450')

# set global variable value
filename = ''
mode_status = ''

# function to open image
def openImg():
    global filename, img_original, canvas_original, menubar

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

    filename = tkinter.filedialog.askopenfilename(title='Open a file', filetypes=filetypes)
    print('[INFO] File path is ' + str(filename))
    print('[INFO] Show the image to the canvas.')

    # read format files
    _, extension = os.path.splitext(filename)

    if(filename != ''):
        # check format files
        if(extension == '.svg'):
            img = svg2png()
        else:
            img = Image.open(filename)
        
        # adjust the image to fit the canvas
        img.thumbnail((250, 250), Image.ANTIALIAS)
        img_original = img
        imgfile = ImageTk.PhotoImage(image=img)
        canvas_result.delete('all')

        # insert image to canvas_original
        canvas_original.create_image((canvas_original.winfo_height()/2, canvas_original.winfo_width()/2), anchor=tkinter.CENTER, image=imgfile)
        canvas_original.image = imgfile
        menubar.entryconfig('Edit', state='normal')
    else:
        tkinter.messagebox.showerror(title='Error', message='Image not found!')

def showResult(img):
    global canvas_result, img_result, mode_text, mode_status

    mode_text.set(mode_status)
    img_result = img
    img_img = ImageTk.getimage(img_result)
    img_img.save('temp_result.png')

    # show image
    canvas_result.create_image((canvas_result.winfo_height()/2, canvas_result.winfo_width()/2), anchor=tkinter.CENTER, image=img)
    canvas_result.image = img

def resetImg():
    global filename, canvas_original, canvas_result, menubar
    canvas_original.delete('all')
    canvas_result.delete('all')
    filename = ''
    menubar.entryconfig('Edit', state='disable')

def imageToCv2():
    img_cv = img_original
    img_cv = img_cv.convert('RGB')
    img_cv = np.array(img_cv)
    img_cv = img_cv[:, :, ::-1].copy()
    return img_cv

def cv2ToImage():
    pass

# function for manipulate image
def toGrey():
    global mode_status

    print('[INFO] Processing image grayscale')
    mode_status = 'Gray Scale'
    
    cvimg = imageToCv2()

    # change image to grey
    cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2GRAY)
    cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)

    b,g,r = cv2.split(cvimg)
    imgmerge = cv2.merge((r,g,b))
    imarray = Image.fromarray(imgmerge)
    imarray.thumbnail((250, 250), Image.ANTIALIAS)
    imgfile = ImageTk.PhotoImage(image=imarray)
    showResult(imgfile)

def svg2png():
    drawing = svg2rlg(filename)
    renderPM.drawToFile(drawing, 'temp.png', fmt='PNG')
    img = renderPM.drawToPIL(drawing)
    return img

def colorQuantization():
    global canvas_result, mode_status
    
    print('[INFO] Processing image quantization')
    mode_status = 'Quantization'
    image = imageToCv2()

    # create new windows
    newWindow = tkinter.Toplevel(root)
    newWindow.title('input')
    newWindow.geometry('200x30')
    entry1 = tkinter.Entry(newWindow)
    entry1.pack(side=tkinter.LEFT)

    def getInput():
        quant = int(entry1.get())
        newWindow.destroy()
        imarray = Image.fromarray(quantineImage(image, quant))
        imarray.thumbnail((250, 250), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image = imarray)
        showResult(photo)

    button1 = tkinter.Button(newWindow, text='Process', command=getInput)
    button1.pack(side=tkinter.RIGHT)       

def quantineImage(image, k):
    i = np.float32(image).reshape(-1,3)
    condition = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,20,1.0)
    _,label,center = cv2.kmeans(i, k , None, condition,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    final_img = center[label.flatten()]
    final_img = final_img.reshape(image.shape)
    return final_img

def samplingImage():
    global mode_status
    
    print('[INFO] Processing image sampling')
    mode_status = 'Sampling'
    cvimg = cv2.cvtColor(imageToCv2(), cv2.COLOR_BGR2RGB)
    
    # create new windows
    newWindow = tkinter.Toplevel(root)
    newWindow.title('input')
    newWindow.geometry('200x30')
    entry1 = tkinter.Entry(newWindow)
    entry1.pack(side=tkinter.LEFT)

    def getInput():
        input = int(entry1.get())
        newWindow.destroy()
        cvblur = cv2.blur(cvimg, (input, input)) # example blur in range 50
        imarray = Image.fromarray(cvblur)
        imarray.thumbnail((250, 250), Image.ANTIALIAS)
        imnp = np.asarray(imarray)
        photo = ImageTk.PhotoImage(image = Image.fromarray(imnp))
        showResult(photo)

    button1 = tkinter.Button(newWindow, text='Process', command=getInput)
    button1.pack(side=tkinter.RIGHT)

def incIntensity():
    global mode_status
    
    print('[INFO] Increasing image intensity')
    mode_status = 'Increase Intensity'
    cvimg = imageToCv2()
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
    newWindow = tkinter.Toplevel(root)
    newWindow.title('input')
    newWindow.geometry('200x30')
    entry1 = tkinter.Entry(newWindow)
    entry1.pack(side=tkinter.LEFT)

    def getInput():
        input = int(entry1.get())
        newWindow.destroy()
        countRGB(input)

    button1 = tkinter.Button(newWindow, text='Process', command=getInput)
    button1.pack(side=tkinter.RIGHT)

def decIntensity():
    global mode_status

    print('[INFO] Decreasing image intensity')
    mode_status = 'Decrease Intensity'
    cvimg = imageToCv2()
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
        imarray.thumbnail((250, 250), Image.ANTIALIAS)
        imgfile = ImageTk.PhotoImage(image=imarray)
        showResult(imgfile)

    # create new windows
    newWindow = tkinter.Toplevel(root)
    newWindow.title('input')
    newWindow.geometry('200x30')
    entry1 = tkinter.Entry(newWindow)
    entry1.pack(side=tkinter.LEFT)

    def getInput():
        input = int(entry1.get())
        newWindow.destroy()
        countRGB(input)

    button1 = tkinter.Button(newWindow, text='Process', command=getInput)
    button1.pack(side=tkinter.RIGHT)

def klise():
    global mode_status

    print('[INFO] Processing klise filter')
    mode_status = 'Klise'
    cvimg = imageToCv2()
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
    imarray.thumbnail((250, 250), Image.ANTIALIAS)
    imgfile = ImageTk.PhotoImage(image=imarray)
    showResult(imgfile)

def lowPassFilter():
    print('[INFO] Processing low pass filter')
    cvimg = imageToCv2()

    #prepare the 5x5 shaped filter
    kernel = np.array([[1, 1, 1, 1, 1], 
                    [1, 1, 1, 1, 1], 
                    [1, 1, 1, 1, 1], 
                    [1, 1, 1, 1, 1], 
                    [1, 1, 1, 1, 1]])
    kernel = kernel/sum(kernel)

    #filter the source image
    cvimg = cv2.filter2D(cvimg,-1,kernel)

    b,g,r = cv2.split(cvimg)
    imgmerge = cv2.merge((r,g,b))
    imarray = Image.fromarray(imgmerge)
    imarray.thumbnail((250, 250), Image.ANTIALIAS)
    imgfile = ImageTk.PhotoImage(image=imarray)
    showResult(imgfile)

def highPassFilter():
    print('[INFO] Processing high pass filter')
    cvimg = imageToCv2()

    #edge detection filter
    # kernel = np.array([[0.0, -1.0, 0.0], 
    #                 [-1.0, 4.0, -1.0],
    #                 [0.0, -1.0, 0.0]])

    kernel = np.array([[0.0, -1.0, 0.0], 
                    [-1.0, 5.0, -1.0],
                    [0.0, -1.0, 0.0]])

    kernel = kernel/(np.sum(kernel) if np.sum(kernel)!=0 else 1)

    #filter the source image
    cvimg = cv2.filter2D(cvimg,-1,kernel)

    b,g,r = cv2.split(cvimg)
    imgmerge = cv2.merge((r,g,b))
    imarray = Image.fromarray(imgmerge)
    imarray.thumbnail((250, 250), Image.ANTIALIAS)
    imgfile = ImageTk.PhotoImage(image=imarray)
    showResult(imgfile)

def bandPassFilter():
    print('[INFO] Processing band pass filter')
    cvimg = imageToCv2()

    # Apply high pass filter
    kernel = np.array([[0.0, -1.0, 0.0], 
                    [-1.0, 5.0, -1.0],
                    [0.0, -1.0, 0.0]])

    kernel = kernel/(np.sum(kernel) if np.sum(kernel)!=0 else 1)
    cvimg = cv2.filter2D(cvimg,-1,kernel)

    # Apply low pass filter
    kernel = np.array([[1, 1, 1, 1, 1], 
                    [1, 1, 1, 1, 1], 
                    [1, 1, 1, 1, 1], 
                    [1, 1, 1, 1, 1], 
                    [1, 1, 1, 1, 1]])
    kernel = kernel/sum(kernel)
    cvimg = cv2.filter2D(cvimg,-1,kernel)
    b,g,r = cv2.split(cvimg)
    imgmerge = cv2.merge((r,g,b))
    imarray = Image.fromarray(imgmerge)
    imarray.thumbnail((250, 250), Image.ANTIALIAS)
    imgfile = ImageTk.PhotoImage(image=imarray)
    showResult(imgfile)

# show histogram of the image
def showHistogram():
    print('[INFO] Processing histogram')
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
    
    print('[INFO] Processing equalization histogram')
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
    global img_result

    print('[INFO] Processing specification histogram')
    src = imageToCv2()
    ref = ImageTk.getimage(img_result).convert('RGB')
    ref = np.array(ref)
    ref = ref[:, :, ::-1].copy()
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)

    # determine if we are performing multichannel histogram matching
    # and then perform histogram matching itself
    multi = True if src.shape[-1] > 1 else False
    matched = exposure.match_histograms(src, ref, multichannel=multi)
    
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([matched],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    # canvas_result.pack_forget()
    plt.show()

# create menubar
menubar = tkinter.Menu(root)
filemenu = tkinter.Menu(menubar, tearoff=0)
filemenu.add_command(label='Open', command=openImg)

# separate list menu
filemenu.add_separator()

filemenu.add_command(label='Exit', command=root.quit)

# create edit menu
editmenu = tkinter.Menu(menubar, tearoff=0)
editmenu.add_command(label='Reset', command=resetImg)
editmenu.add_command(label='2Grey', command=toGrey)
editmenu.add_command(label='Quantization', command=colorQuantization)
editmenu.add_command(label='Sampling', command=samplingImage)
editmenu.add_command(label='Increase Intensity', command=incIntensity)
editmenu.add_command(label='Decrease Intensity', command=decIntensity)
editmenu.add_command(label='Klise', command=klise)
editmenu.add_command(label='Low Pass Filter', command=lowPassFilter)
editmenu.add_command(label='High Pass Filter', command=highPassFilter)
editmenu.add_command(label='Band Pass Filter', command=bandPassFilter)
editmenu.add_separator()
editmenu.add_command(label='Histogram', command=showHistogram)
editmenu.add_command(label='Histogram Equalization', command=showHisEqual)
editmenu.add_command(label='Histogram Specification', command=showHisSpec)

# add cascade
menubar.add_cascade(label='File', menu=filemenu)
menubar.add_cascade(label='Edit', menu=editmenu)
menubar.entryconfig('Edit', state='disable')

# add menubar to root
root.config(menu=menubar)

# add label to root
mode_text = tkinter.StringVar()
mode_text.set('Mode Status')
label = tkinter.Label(root, textvariable=mode_text, fg = 'black', font = 'Times').pack()

# create canvas for original image
canvas_original = tkinter.Canvas(root, width=250, height=250, background='white')
canvas_original.pack(side=tkinter.LEFT)

# create canvas for result image
canvas_result = tkinter.Canvas(root, width=250, height=250, background='white')
canvas_result.pack(side=tkinter.RIGHT)

# make window stay
root.mainloop()