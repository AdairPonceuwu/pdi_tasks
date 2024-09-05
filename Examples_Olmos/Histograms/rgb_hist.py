import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image")
args = vars(ap.parse_args())

image_BGR = cv2.imread(args["image"])
image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
r,g,b = cv2.split(image_RGB)

#image: input image. Its possible to pass a list of images
#[0]: channel where the histogram is calculated. El primer valor es la imagen, segundo el canal de color (0 rojo, 1 verde, 2 azul)
#Mask: this parameter is optional
#[256]: histogram with 256 values
#[0,256]: the range of possiblevpixel values 

histr = cv2.calcHist((r),[0],None,[255],[0,256])
histg = cv2.calcHist((g),[0],None,[255],[0,256])
histb = cv2.calcHist((b),[0],None,[255],[0,256])

fig = plt.figure(figsize=[14, 14])

ax1=fig.add_subplot(2,1,1)
ax2=fig.add_subplot(2,3,4)
ax3=fig.add_subplot(2,3,5)
ax4=fig.add_subplot(2,3,6)

ax1.imshow(image_RGB)
ax1.set_title("Original image")

ax2.plot(histr, color="r")
ax2.set_title("Histogram RED")

ax3.plot(histg, color="g")
ax3.set_title("Histogram GREEN")

ax4.plot(histb, color="b")
ax4.set_title("Histogram BLUE")

plt.show()