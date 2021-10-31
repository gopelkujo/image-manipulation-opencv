#############################
# Mochamad Nauval Dwisatya  #
# 191524023                 #
#############################

# import needed package
import cv2, tkinter, os, numpy as np
from PIL import ImageTk, Image
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage import exposure

# set global variable value
canvas_size_x = 280
canvas_size_y = 280
window_size_x = 750
window_size_y = 650
filename = ''
mode_status = ''
selected_histogram = 'histogram'

# set up window
root = tkinter.Tk()
root.title('Image Manipulation')
root.geometry(str(window_size_x) + 'x' + str(window_size_y))

# function to open image
def openImg():
    global filename, img_original, canvas_original, menubar
    filename_temp = filename
    print('[INFO] Selecting files...')

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
    
    # read format files
    _, extension = os.path.splitext(filename)

    if(filename == ''):
        filename = filename_temp
    
    # double check for canceling open file case
    if(filename != ''):
        print('[INFO] File path is ' + str(filename))
        print('[INFO] Show the image to the canvas.')
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
        print('[INFO] Image not found.')

def showResult(img):
    global canvas_result, img_result, mode_text, mode_status, btn_hist_ori, btn_hist_equ, btn_hist_spe
    print("[INFO] Show the result to the canvas.")

    mode_text.set(mode_status)
    img_result = img
    img_img = ImageTk.PhotoImage(image=img_result)
    save_img = ImageTk.getimage(img_img)
    save_img.save('temp_result.png')

    # show image
    canvas_result.create_image((canvas_result.winfo_height()/2, canvas_result.winfo_width()/2), anchor=tkinter.CENTER, image=img_img)
    canvas_result.image = img_img

    # auto update histogram
    if(selected_histogram == 'histogram'): showHistogram()
    if(selected_histogram == 'hisequ'): showHisEqual()
    if(selected_histogram == 'hisspe'): showHisSpec()

def resetImg():
    global filename, canvas_original, canvas_result, menubar, histogram_canvas, btn_hist_ori, btn_hist_equ, btn_hist_spe
    canvas_original.delete('all')
    canvas_result.delete('all')
    filename = ''
    # disable edit menu & histogram button
    menubar.entryconfig('Edit', state='disable')
    btn_hist_ori.config(state='disable')
    btn_hist_equ.config(state='disable')
    btn_hist_spe.config(state='disable')
    histogram_canvas.get_tk_widget().forget()

def imageToCv2(img):
    img_cv = img
    img_cv = img_cv.convert('RGB')
    img_cv = np.array(img_cv)
    img_cv = img_cv[:, :, ::-1].copy()
    return img_cv

def cv2ToImage(cvimg):
    b,g,r = cv2.split(cvimg)
    imgmerge = cv2.merge((r,g,b))
    imarray = Image.fromarray(imgmerge)
    imarray.thumbnail((canvas_size_x, canvas_size_y), Image.ANTIALIAS)
    # imgfile = ImageTk.PhotoImage(image=imarray)
    return imarray

# function for manipulate image
def toGrey():
    global mode_status

    print('[INFO] Processing image grayscale...')
    mode_status = 'Gray Scale'
    
    cvimg = imageToCv2(img_original)

    # change image to grey
    cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2GRAY)
    cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
    showResult(cv2ToImage(cvimg))

def svg2png():
    drawing = svg2rlg(filename)
    renderPM.drawToFile(drawing, 'temp.png', fmt='PNG')
    img = renderPM.drawToPIL(drawing)
    return img

def colorQuantization():
    global canvas_result, mode_status
    
    print('[INFO] Processing image quantization...')
    mode_status = 'Quantization'
    image = imageToCv2(img_original)

    # create new windows
    newWindow = tkinter.Toplevel(root)
    newWindow.title('input')
    newWindow.geometry('200x30')
    entry1 = tkinter.Entry(newWindow)
    entry1.pack(side=tkinter.LEFT)
    entry1.focus_force()

    def getInput():
        quant = int(entry1.get())
        newWindow.destroy()
        imarray = Image.fromarray(quantineImage(image, quant))
        imarray.thumbnail((250, 250), Image.ANTIALIAS)
        showResult(imarray)

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
    
    print('[INFO] Processing image sampling...')
    mode_status = 'Sampling'
    cvimg = cv2.cvtColor(imageToCv2(img_original), cv2.COLOR_BGR2RGB)
    
    # create new windows
    newWindow = tkinter.Toplevel(root)
    newWindow.title('input')
    newWindow.geometry('200x30')
    entry1 = tkinter.Entry(newWindow)
    entry1.pack(side=tkinter.LEFT)
    entry1.focus_force()

    def getInput():
        input = int(entry1.get())
        newWindow.destroy()
        cvblur = cv2.blur(cvimg, (input, input)) # example blur in range 50
        showResult(cv2ToImage(cvblur))

    button1 = tkinter.Button(newWindow, text='Process', command=getInput)
    button1.pack(side=tkinter.RIGHT)

def incIntensity():
    global mode_status
    
    print('[INFO] Increasing image intensity...')
    mode_status = 'Increase Intensity'
    cvimg = imageToCv2(img_original)
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

        showResult(cv2ToImage(cvimg))

    # create new windows
    newWindow = tkinter.Toplevel(root)
    newWindow.title('input')
    newWindow.geometry('200x30')
    entry1 = tkinter.Entry(newWindow)
    entry1.pack(side=tkinter.LEFT)
    entry1.focus_force()

    def getInput():
        input = int(entry1.get())
        newWindow.destroy()
        countRGB(input)

    button1 = tkinter.Button(newWindow, text='Process', command=getInput)
    button1.pack(side=tkinter.RIGHT)

def decIntensity():
    global mode_status

    print('[INFO] Decreasing image intensity...')
    mode_status = 'Decrease Intensity'
    cvimg = imageToCv2(img_original)
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

        showResult(cv2ToImage(cvimg))

    # create new windows
    newWindow = tkinter.Toplevel(root)
    newWindow.title('input')
    newWindow.geometry('200x30')
    entry1 = tkinter.Entry(newWindow)
    entry1.pack(side=tkinter.LEFT)
    entry1.focus_force()

    def getInput():
        input = int(entry1.get())
        newWindow.destroy()
        countRGB(input)

    button1 = tkinter.Button(newWindow, text='Process', command=getInput)
    button1.pack(side=tkinter.RIGHT)

def klise():
    global mode_status

    print('[INFO] Processing klise filter...')
    mode_status = 'Klise'
    cvimg = imageToCv2(img_original)
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

    showResult(cv2ToImage(cvimg))

def lowPassFilter():
    global mode_status

    print('[INFO] Processing low pass filter...')
    mode_status = 'Low Pass'
    cvimg = imageToCv2(img_original)

    #prepare the 5x5 shaped filter
    kernel = np.array([[1, 1, 1, 1, 1], 
                    [1, 1, 1, 1, 1], 
                    [1, 1, 1, 1, 1], 
                    [1, 1, 1, 1, 1], 
                    [1, 1, 1, 1, 1]])
    kernel = kernel/sum(kernel)

    #filter the source image
    cvimg = cv2.filter2D(cvimg,-1,kernel)
    showResult(cv2ToImage(cvimg))

def highPassFilter():
    global mode_status
    print('[INFO] Processing high pass filter...')
    cvimg = imageToCv2(img_original)
    mode_status = 'High Pass'

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
    showResult(cv2ToImage(cvimg))

def bandPassFilter():
    global mode_status
    print('[INFO] Processing band pass filter...')
    cvimg = imageToCv2(img_original)
    mode_status = 'Band Pass'

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
    showResult(cv2ToImage(cvimg))

# show histogram of the result image
def showHistogram():
    global histogram_canvas, selected_histogram, btn_hist_ori, btn_hist_equ, btn_hist_spe

    print('[INFO] Processing histogram...')
    img = imageToCv2(img_result)

    histogram_figure = plt.Figure(figsize=(4.5, 2.5), dpi=100)
    ax_histogram = histogram_figure.add_subplot(111)
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        ax_histogram.plot(histr, color=col)

    histogram_canvas.get_tk_widget().destroy()
    histogram_canvas = FigureCanvasTkAgg(histogram_figure, third_frame)
    histogram_canvas.draw()
    histogram_canvas.get_tk_widget().pack(pady=10)
    ax_histogram.set_title('Histogram')
    selected_histogram='histogram'
    btn_hist_ori.config(state='disable')
    btn_hist_equ.config(state='normal')
    btn_hist_spe.config(state='normal')

def showHisEqual():
    global histogram_canvas, selected_histogram, btn_hist_ori, btn_hist_equ, btn_hist_spe

    print('[INFO] Processing equalization histogram...')
    img_img = imageToCv2(img_result)
    img_img = cv2.cvtColor(img_img, cv2.COLOR_BGR2GRAY)
    equalized_img = cv2.equalizeHist(img_img)
    equalized_img = cv2.cvtColor(equalized_img, cv2.COLOR_BGR2RGB)

    histogram_figure = plt.Figure(figsize=(4.5, 2.5), dpi=100)
    ax_histogram = histogram_figure.add_subplot(111)
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([equalized_img],[i],None,[256],[0,256])
        ax_histogram.plot(histr,color = col)

    histogram_canvas.get_tk_widget().destroy()
    histogram_canvas = FigureCanvasTkAgg(histogram_figure, third_frame)
    histogram_canvas.draw()
    histogram_canvas.get_tk_widget().pack(pady=10)
    ax_histogram.set_title('Histogram Equalization')
    selected_histogram='hisequ'
    btn_hist_ori.config(state='normal')
    btn_hist_equ.config(state='disable')
    btn_hist_spe.config(state='normal')


def showHisSpec():
    global histogram_canvas, selected_histogram, btn_hist_ori, btn_hist_equ, btn_hist_spe

    print('[INFO] Processing specification histogram...')
    src = imageToCv2(img_original)
    ref = imageToCv2(img_result)
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)

    # determine if we are performing multichannel histogram matching
    # and then perform histogram matching itself
    multi = True if src.shape[-1] > 1 else False
    matched = exposure.match_histograms(src, ref, multichannel=multi)
    
    histogram_figure = plt.Figure(figsize=(4.5, 2.5), dpi=100)
    ax_histogram = histogram_figure.add_subplot(111)
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([matched],[i],None,[256],[0,256])
        ax_histogram.plot(histr,color = col)

    histogram_canvas.get_tk_widget().destroy()
    histogram_canvas = FigureCanvasTkAgg(histogram_figure, third_frame)
    histogram_canvas.draw()
    histogram_canvas.get_tk_widget().pack(pady=10)
    ax_histogram.set_title('Histogram Specification')
    selected_histogram='hisspe'
    btn_hist_ori.config(state='normal')
    btn_hist_equ.config(state='normal')
    btn_hist_spe.config(state='disable')

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
editmenu.add_separator()
editmenu.add_command(label='2Grey', command=toGrey)
editmenu.add_command(label='Quantization', command=colorQuantization)
editmenu.add_command(label='Sampling', command=samplingImage)
editmenu.add_command(label='Increase Intensity', command=incIntensity)
editmenu.add_command(label='Decrease Intensity', command=decIntensity)
editmenu.add_command(label='Klise', command=klise)
editmenu.add_command(label='Low Pass Filter', command=lowPassFilter)
editmenu.add_command(label='High Pass Filter', command=highPassFilter)
editmenu.add_command(label='Band Pass Filter', command=bandPassFilter)

# add cascade
menubar.add_cascade(label='File', menu=filemenu)
menubar.add_cascade(label='Edit', menu=editmenu)
menubar.entryconfig('Edit', state='disable')

# add menubar to root
root.config(menu=menubar)

# create frame
first_frame = tkinter.Frame(root, width=window_size_x, height=canvas_size_y+40)
first_frame.pack(anchor=tkinter.NW)

# create canvas for original image
canvas_original = tkinter.Canvas(first_frame, width=canvas_size_x, height=canvas_size_y, background='white')
canvas_original.pack(side=tkinter.LEFT, anchor=tkinter.NW, padx=25, pady=20)

# add label to root
mode_text = tkinter.StringVar()
mode_text.set('Mode Status')
label = tkinter.Label(first_frame, textvariable=mode_text, fg = 'black', font = 'Times').pack(side=tkinter.LEFT, anchor=tkinter.NW, pady=150)

# create canvas for result image
canvas_result = tkinter.Canvas(first_frame, width=canvas_size_x, height=canvas_size_y, background='white')
canvas_result.pack(side=tkinter.LEFT, anchor=tkinter.NW, padx=25, pady=20)

# create second frame
second_frame = tkinter.Frame(root)
second_frame.pack()

# create histogram button
btn_hist_ori = tkinter.Button(second_frame, text='Histogram', command=showHistogram, state='disable')
btn_hist_ori.pack(side=tkinter.LEFT, padx=5)
btn_hist_equ = tkinter.Button(second_frame, text='Histogram Equalization', command=showHisEqual, state='disable')
btn_hist_equ.pack(side=tkinter.LEFT, padx=5)
btn_hist_spe = tkinter.Button(second_frame, text='Histogram Specification', command=showHisSpec, state='disable')
btn_hist_spe.pack(side=tkinter.LEFT, padx=5)

third_frame = tkinter.Frame(root)
third_frame.pack()

# create histogram canvas
histogram_figure = plt.Figure(figsize=(4.5, 2.5), dpi=100)
ax_histogram = histogram_figure.add_subplot(111)
histogram_canvas = FigureCanvasTkAgg(histogram_figure, third_frame)
histogram_canvas.draw()
histogram_canvas.get_tk_widget().pack(pady=10)
ax_histogram.set_visible(False)
ax_histogram.set_title('Histogram')

# make window stay
root.mainloop()