import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image")
ap.add_argument("-k", "--ksize", required=True, help="Size of the kernel")
ap.add_argument("-b", "--blurtype", required=True, help="Type of blur")
args = vars(ap.parse_args())


img = cv2.imread(args["image"])
if img is None:
    print("Error: Could not open or find the image.")
    exit()

kernel_size = int(args["ksize"])

blur_type = int(args["blurtype"])

def aplicar_kernel_blur(image):
    blur = cv2.blur(image, (kernel_size,kernel_size))
    return blur

def aplicar_kernel_gauss(image):
    blur = cv2.GaussianBlur(image, (kernel_size,kernel_size),0)
    return blur

def aplicar_kernel_median(image):
    blur = cv2.medianBlur(image, kernel_size)
    return blur

def aplicar_kernel_median(image):
    blur = cv2.medianBlur(image, kernel_size)
    return blur

def aplicar_kernel_bilateral(image):
    blur = cv2.bilateralFilter(image, kernel_size, 21, 21)
    return blur

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

r, g, b = cv2.split(rgb)

if blur_type == 0: 
    titulo = "Blur(Average)" 
    r_blur = aplicar_kernel_blur(r)
    g_blur = aplicar_kernel_blur(g)
    b_blur = aplicar_kernel_blur(b)

    r_blur = np.clip(r_blur, 0, 255).astype(np.uint8)
    g_blur = np.clip(g_blur, 0, 255).astype(np.uint8)
    b_blur = np.clip(b_blur, 0, 255).astype(np.uint8)

    rgb_blur = cv2.merge([r_blur, g_blur, b_blur])
    
elif blur_type == 1:
    titulo = "Gauss"
    r_blur = aplicar_kernel_gauss(r)
    g_blur = aplicar_kernel_gauss(g)
    b_blur = aplicar_kernel_gauss(b)

    r_blur = np.clip(r_blur, 0, 255).astype(np.uint8)
    g_blur = np.clip(g_blur, 0, 255).astype(np.uint8)
    b_blur = np.clip(b_blur, 0, 255).astype(np.uint8)

    rgb_blur = cv2.merge([r_blur, g_blur, b_blur])
    
elif blur_type == 2:
    titulo = "Median"
    r_blur = aplicar_kernel_median(r)
    g_blur = aplicar_kernel_median(g)
    b_blur = aplicar_kernel_median(b)

    r_blur = np.clip(r_blur, 0, 255).astype(np.uint8)
    g_blur = np.clip(g_blur, 0, 255).astype(np.uint8)
    b_blur = np.clip(b_blur, 0, 255).astype(np.uint8)

    rgb_blur = cv2.merge([r_blur, g_blur, b_blur])

elif blur_type == 3:
    titulo = "Bilateral"
    r_blur = aplicar_kernel_bilateral(r)
    g_blur = aplicar_kernel_bilateral(g)
    b_blur = aplicar_kernel_bilateral(b)

    r_blur = np.clip(r_blur, 0, 255).astype(np.uint8)
    g_blur = np.clip(g_blur, 0, 255).astype(np.uint8)
    b_blur = np.clip(b_blur, 0, 255).astype(np.uint8)

    rgb_blur = cv2.merge([r_blur, g_blur, b_blur])
    
plt.subplot(1, 2, 1)
plt.imshow(rgb)
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(rgb_blur)
plt.title(titulo)

plt.show()
