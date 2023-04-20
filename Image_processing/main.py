import os
import cv2
import numpy as np

# Clone image function
def clone(img):
    height = img.shape[0]
    width = img.shape[1]
    img2=np.zeros((height,width),np.uint8)
    for i in range(0,height):
        for j in range(0,width):
            img2[i,j]=img[i,j]
    return img2
# Read image file

def function1(filename):
    #img=cv2.imread("5.jfif")
    img=cv2.imread(filename)
    # Get image dimensions
    height = img.shape[0]
    width = img.shape[1]

    # Reduce image size by half for faster processing
    if(width>1000):
        img = cv2.resize(img,(height//2,width//2))
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #print(str(height)+","+str(width))
    # Adaptive binarization, parameters can be adjusted as needed
    # Otsu's method thresholding, parameters can be adjusted as needed
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert image for easier processing
    binary=~binary
    # Find contours
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the largest contour (white) onto a blank black image
    mask = np.zeros_like(gray)

    # Iterate through contours, removing borders
    for i in range(len(contours)):
        rect = cv2.minAreaRect(contours[i])
        center, (width, height), angle = rect
        cv2.drawContours(mask, contours, i, 255, -1)
        # Remove lines
        if(width<10 and height>30):
            cv2.drawContours(mask, contours, i, 0, -1)
        if(height<10 and width>30):
            cv2.drawContours(mask, contours, i, 0, -1)
        # Remove large contours
        if(height>100 or width>100):
            cv2.drawContours(mask,contours,i,0,-1)

    # Morphological dilation operation, dilation size setting
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))

    # Perform dilation operation on the image
    mask1=clone(mask)
    mask1=cv2.medianBlur(mask1, 3)
    dilated = cv2.dilate(mask1, kernel)

    # Find contours
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Select character area to retain
    mask2 = np.zeros_like(gray)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > 6200:
            cv2.drawContours(mask2, contours, i, 255, -1)

    # Create character area image
    mask3=np.zeros_like(gray)
    h = mask3.shape[0]
    w = mask3.shape[1]
    # Retain characters in character area
    for i in range(0,h):
        for j in range(0,w):
            if(mask2[i,j]==255 and binary[i,j]==255):
                mask3[i,j]=255

    # When characters are larger, use result0 as a result
    result0=clone(mask3)
    #cv2.namedWindow("win")

    # Find contours
    contours, hierarchy = cv2.findContours(mask3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Remove large boundaries
    for i in range(len(contours)):
        rect = cv2.minAreaRect(contours[i])
        center, (width, height), angle = rect
        if(width>30 or height>30):
            cv2.drawContours(mask3, contours, i, 0, -1)

    # Morphological dilation, set dilation size to 15
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    # Perform dilation operation on the image
    mask1=clone(mask3)
    dilated = cv2.dilate(mask1, kernel)
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the index and area of large contours
    mask4 = np.zeros_like(gray)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > 5000:
            cv2.drawContours(mask4, contours, i, 255, -1)

    # Retain characters within character boundaries
    result=np.zeros_like(gray)
    for i in range(0,h):
        for j in range(0,w):
            if(mask4[i,j]==255 and binary[i,j]==255):
                result[i,j]=255


    # Remove boundaries
    contours, hierarchy = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        rect = cv2.minAreaRect(contours[i])
        center, (width, height), angle = rect
        if(width>30 or height>30):
            cv2.drawContours(result, contours, i, 0, -1)

    BLACK = [0,0,0]

    #print(result.shape)
    # When the text is larger, execute if(0), and when the text is smaller, execute if(1)
    res=result
    if(0):
        # Expand the boundary
        constant = cv2.copyMakeBorder(result,200,200,200,200,cv2.BORDER_CONSTANT,value=BLACK)
        # Invert to black text on white background
        constant=~constant
        #cv2.imshow("constant",constant)

        h = constant.shape[0]
        w = constant.shape[1]

        # Set new image size to 1000, 1000
        x=int(w/2-500)
        y=int(h/2-500)
        w=1000
        h=1000
        crop = constant[y:h + y, x:w + x]
        #cv2.imshow("crop",crop)
        # Resize the image to 2000, 2000
        img = cv2.resize(img, (2000 , 2000))
        #cv2.imwrite("result.jpg", crop)
        res=crop
    else:
        # Read in the previously processed image result0
        constant = cv2.copyMakeBorder(result0, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=BLACK)
        #cv2.imshow("constant", constant)
        constant = ~constant

        h = constant.shape[0]
        w = constant.shape[1]
        # rect=(int(w/2-500),int(h/2-500),1000,1000)
        x = int(w / 2 - 500)
        y = int(h / 2 - 500)
        w = 1000
        h = 1000
        crop = constant[y:h + y, x:w + x]
        #cv2.imshow("crop", crop)
        img = cv2.resize(img, (2000, 2000))
        #cv2.imwrite("result.jpg",crop)
        res=crop
    return res
idx=0
for root, dirs, files in os.walk('./FELO leaflet examples'):
    for file in files:
        path = os.path.join(root, file)
        print(path)
        fliesion = os.path.splitext(path)[-1]
        print(fliesion)
        res=0
        if(fliesion=='.png' or fliesion=='.jfif'):
            res=function1(path)
        idx=idx+1
        cv2.imwrite("result/"+str(idx)+'.png',res)




