import numpy as np
import cv2
from imutils.object_detection import non_max_suppression
import pytesseract
from matplotlib import pyplot as plt
from googletrans import Translator
from pytesseract.pytesseract import image_to_pdf_or_hocr
from PIL import Image, ImageEnhance
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

brfactor = 2 #brightens the image
contrfactor = 0.7 #increases contrast
shfactor = 0.5 #make it blur
kernel = np.ones((2,2),np.uint8)

def east_detect(image):
    t=[]
    # image = noise_removal(image)
    #layers of EAST required for text detection 
    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    #make a copy of the inputted image
    orig = image.copy()
    original = image.copy()

    #converts BGR to Y (B=Y, G=Y, R=Y)
    #converts single channel image (grayscale) to multichannel(RGB) by replicating
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    #number of rows and cols
    (H, W) = image.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height: Should be multiple of 32 
    (newW, newH) = (320, 320)

    rW = W / float(newW)
    rH = H / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))

    (H, W) = image.shape[:2]

    #deep neural network used
    net = cv2.dnn.readNet("./frozen_east_text_detection.pb")

    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)

    net.setInput(blob)

    (scores, geometry) = net.forward(layerNames)

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    results = []
    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            # Set minimum confidence as required
            if scoresData[x] < 0.5:
                continue
            # compute the offset factor as our resulting feature maps will
            #  x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        # draw the bounding box on the image
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # extract the region of interest
        r = orig[startY:endY, startX:endX]

        # configuration setting to convert image to string.
        configuration = ("-l eng --oem 1 --psm 8")
        # This will recognize the text from the image of bounding box
        text = pytesseract.image_to_string(r, config=configuration)

        # append bbox coordinate and associated text to the list of results
        results.append(((startX, startY, endX, endY), text))
    
    orig_image = orig.copy()
    trans_image= orig.copy()
    translator = Translator() 
    for ((start_X, start_Y, end_X, end_Y), text) in results:
        # Displaying text
        text = "".join([x if ord(x) < 128 else "" for x in text]).strip()
        t.append(translator.translate(text,src='en',dest='hi'))
        cv2.rectangle(orig_image, (start_X, start_Y), (end_X, end_Y), (0, 0, 255), 2)
        cv2.putText(orig_image,text , (start_X, start_Y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (240, 29, 29), 3) 
    for trans in t:
        print(trans.origin, ' -> ', trans.text)
        print(trans.origin, ' -> ', trans.pronunciation)
    for ((start_X, start_Y, end_X, end_Y), text) in results:
        text = "".join([x if ord(x) < 128 else "" for x in text]).strip()
        for trans in t:
            if text == trans.origin:
                cv2.rectangle(trans_image, (start_X, start_Y), (end_X, end_Y), (0, 0, 255), 2)
                if (trans.pronunciation) is not None  :
                    cv2.putText(trans_image,(trans.pronunciation).upper() , (start_X, start_Y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (240, 29, 29), 2) 
                else :
                    cv2.putText(trans_image,(trans.text).upper() , (start_X, start_Y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (240, 29, 29), 2)
    # plt.imshow(orig)
    fig = plt.figure()
    fig.add_subplot(1, 3, 1)
    plt.axis('off')
    plt.title("IMAGE FED TO EAST")
    plt.imshow(original)
    fig.add_subplot(1, 3, 2)
    plt.imshow(orig_image)
    plt.axis('off')
    plt.title("IMAGE WITH DETECTED TEXT")
    fig.add_subplot(1, 3, 3)
    plt.imshow(trans_image)
    plt.axis('off')
    plt.title("IMAGE WITH TRANSLATED TEXT")
    plt.show()

    return orig_image

def thresholding_truncation(image):
    ret,thr= cv2.threshold(image,127,255,cv2.THRESH_TRUNC)
    return thr

def thresholding_bin_inv(image):
    ret,thr= cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)
    return thr

def rgb2bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def rgb2gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def dilation(image):
    kernel = np.ones((2,2),np.uint8)
    return cv2.dilate(image,kernel,iterations=1)

def erosion(image):
    kernel = np.ones((2,2),np.uint8)
    return cv2.erode(image,kernel,iterations=1)

def isbright(image, dim, thresh):
    # Resize image to 10x10
    image = cv2.resize(image, (dim, dim))
    # Convert color space to LAB format and extract L channel
    L, A, B = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    # Normalize L channel by dividing all pixel values with maximum pixel value
    L = L/np.max(L)
    # Return True if mean is greater than thresh else False
    return np.mean(L) > thresh # false implying image is dark

def noise_removal(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)

def check_bright(image):
    a = isbright(image,20,0.4)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    if a==False:
        # image is dark so brighten it
        br = ImageEnhance.Brightness(image)
        bright_im = np.array(br.enhance(brfactor))
        return bright_im
    else:
        # image is already bright 
        contr = ImageEnhance.Contrast(image)
        hc_im = contr.enhance(contrfactor)
        blur = ImageEnhance.Sharpness(hc_im)
        bl_im = np.array(blur.enhance(shfactor))
        return bl_im

image = cv2.imread("path-to-input-image")
img = check_bright(image)

fig = plt.figure()
fig.add_subplot(1, 2, 1)
plt.axis('off')
plt.title("ORIGINAL IMAGE")
plt.imshow(image)
fig.add_subplot(1, 2, 2)
plt.imshow(img)
plt.axis('off')
plt.title("IMAGE AFTER PREPROCESSING")
plt.show()

img = east_detect(img)
cv2.imwrite("path-to-output-image", img)