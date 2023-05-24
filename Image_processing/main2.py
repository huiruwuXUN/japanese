import os
import cv2
import numpy as np
import math

#Processing function
#This function can remove some light color interference
def test1(name):
    # Read an image
    img=cv2.imread(name)

    if img is None:
        print(f"Error: Cannot open or read image file '{name}'. Please check file path and integrity.")
        return None

    #cv2.imshow("img",img)
    # Get the image dimensions
    height = img.shape[0]
    width = img.shape[1]
    # Arrays to count the pixels
    r_arr=[0]*256
    g_arr=[0]*256
    b_arr=[0]*256
    # Most pixels
    r_max_size=0
    g_max_size=0
    b_max_size=0
    # Pixel mean values
    r_mean=0
    g_mean=0
    b_mean=0
    # Background threshold
    qerr1=50
    # Bright color threshold
    qerr2=200
    # Font color threshold
    qerr3=200
    # Pixel color statistics
    for i in range(0,height):
        for j in range(0,width):
            (b,g,r)=img[i,j]
            b=int(b)
            g=int(g)
            r=int(r)
            r_arr[r]=r_arr[r]+1
            g_arr[g]=g_arr[g]+1
            b_arr[b]=b_arr[b]+1
    # Red channel mean extraction
    for i in range(0,256):
        a=r_arr[i]
        if(a>r_max_size):
            r_max_size=a
            r_mean=i
    # Green channel mean extraction
    for i in range(0,256):
        a=g_arr[i]
        if(a>g_max_size):
            g_max_arr=a
            g_mean=i
    # Blue channel mean extraction
    for i in range(0,256):
        a=b_arr[i]
        if(a>b_max_size):
            b_max_size=a
            b_mean=i
    # Color mean variance threshold
    for i in range(0,height):
        for j in range(0,width):
            (b,g,r)=img[i,j]
            b=int(b)
            g=int(g)
            r=int(r)
            # Calculate distance
            a0=b-b_mean
            b0=g-g_mean
            c0=r-r_mean
            # Calculate variance
            err=math.sqrt(a0*a0+b0*b0+c0*c0)
            # Set to white if less than threshold
            if(err<qerr1):
                img[i,j]=(255,255,255)
    # Color brightness variance threshold
    for i in range(0,height):
        for j in range(0,width):
            (b,g,r)=img[i,j]
            b=int(b)
            g=int(g)
            r=int(r)
            # Calculate distance
            a0=b-255
            b0=g-255
            c0=r-255
            # Calculate variance
            err=float(math.sqrt(float(a0*a0+b0*b0+c0*c0)))
            if(err<qerr2):
                img[i,j]=(255,255,255)
    # Font color threshold
    min_size=1000000
    min_color=(-1,-1,-1)
    for i in range(0,height):
        for j in range(0,width):
            (b, g, r) = img[i, j]
            b=int(b)
            g=int(g)
            r=int(r)
            # Calculate distance
            a=math.sqrt(float(b*b+g*g+r*r))
            if(a<min_size and r_arr[r]>100 and b_arr[b]>100 and g_arr[g]>100):
                min_size=a
                min_color=(b,g,r)

    #print(min_color)
    # According to the background color threshold
    for i in range(0,height):
        for j in range(0,width):
            (b, g, r) = img[i, j]
            b=int(b)
            g=int(g)
            r=int(r)
            (mb,mg,mr)=min_color
            a0=b-mb
            b0=g-mg
            c0=r-mr
            err=math.sqrt(float(a0*a0+b0*b0+c0*c0))
            # Calculate absolute value
            err1=abs(a0)
            err2=abs(b0)
            err3=abs(c0)
            #if(err3!=202):
            #    print(err3)
            '''
            if(err<qerr3 ):
                img[i,j]=(b,g,r)
            else:
                img[i,j]=(255,255,255)
            '''
            # Threshold
            if(err<qerr3 and err1<120 and err2<120 and err3<120):
                img[i,j]=(b,g,r)
            else:
                img[i,j]=(255,255,255)

    return img
for i in range(1, 47):
    name = "imgs/" + str(i) + ".png"
    print("Processing image " + str(i))
    img = test1(name)
    if img is not None:
        cv2.imwrite("result2/" + str(i) + ".png", img)


# Processing function
def test2(name):
    # Read an image
    img=cv2.imread(name)

    if img is None:
        print(f"Error: Cannot open or read image file '{name}'. Please check file path and integrity.")
        return None

    #cv2.imshow("img",img)
    # Get image dimensions
    height = img.shape[0]
    width = img.shape[1]
    # Arrays to count pixel values
    r_arr=[0]*256
    g_arr=[0]*256
    b_arr=[0]*256
    # Maximum pixel count
    r_max_size=0
    g_max_size=0
    b_max_size=0
    # Pixel mean values
    r_mean=0
    g_mean=0
    b_mean=0
    # Background threshold
    qerr1=50
    # Brightness threshold
    qerr2=200
    # Text threshold
    qerr3=200
    # Count pixel color values
    for i in range(0,height):
        for j in range(0,width):
            (b,g,r)=img[i,j]
            b=int(b)
            g=int(g)
            r=int(r)
            r_arr[r]=r_arr[r]+1
            g_arr[g]=g_arr[g]+1
            b_arr[b]=b_arr[b]+1
    # Extract red channel mean value
    for i in range(0,256):
        a=r_arr[i]
        if(a>r_max_size):
            r_max_size=a
            r_mean=i
    # Extract green channel mean value
    for i in range(0,256):
        a=g_arr[i]
        if(a>g_max_size):
            g_max_arr=a
            g_mean=i
    # Extract blue channel mean value
    for i in range(0,256):
        a=b_arr[i]
        if(a>b_max_size):
            b_max_size=a
            b_mean=i
    # Color mean variance threshold
    for i in range(0,height):
        for j in range(0,width):
            (b,g,r)=img[i,j]
            b=int(b)
            g=int(g)
            r=int(r)
            # Calculate distance
            a0=b-b_mean
            b0=g-g_mean
            c0=r-r_mean
            # Calculate variance
            err=math.sqrt(a0*a0+b0*b0+c0*c0)
            # Set to white if below threshold
            if(err<qerr1):
                img[i,j]=(255,255,255)
    # Color brightness variance threshold
    for i in range(0,height):
        for j in range(0,width):
            (b,g,r)=img[i,j]
            b=int(b)
            g=int(g)
            r=int(r)
            # Calculate distance
            a0=b-255
            b0=g-255
            c0=r-255
            # Calculate variance
            err=float(math.sqrt(float(a0*a0+b0*b0+c0*c0)))
            if(err<qerr2):
                img[i,j]=(255,255,255)
    return img
for i in range(1, 47):
    name = "imgs/" + str(i) + ".png"
    print("Processing image " + str(i))
    img = test2(name)
    if img is not None:
        cv2.imwrite("result3/" + str(i) + ".png", img)
