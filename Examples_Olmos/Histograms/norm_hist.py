import matplotlib.pyplot as plt
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image")
args = vars(ap.parse_args())

image_BGR = cv2.imread(args["image"])
image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
image = cv2.cvtColor(image_RGB, cv2.COLOR_BGR2GRAY)

#image: input image. Its possible to pass a list of images
#[0]: channel where the histogram is calculated. El primer valor es la imagen, segundo el canal de color (0 rojo, 1 verde, 2 azul)
#Mask: this parameter is optional
#[256]: histogram with 256 values
#[0,256]: the range of possiblevpixel values 

hist = cv2.calcHist((image),[0],None,[256],[0,256])
hist /= hist.sum()

fig = plt.figure(figsize=[14, 5])

ax1=fig.add_subplot(1,2,1)
ax2=fig.add_subplot(1,2,2)

ax1.imshow(image, cmap="gray")
ax1.set_title("Original image")

ax2.plot(hist)
ax2.set_title("Histogram")

plt.show()