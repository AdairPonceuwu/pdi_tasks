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


# Corrección gamma
def correccion_gamma(channel, gamma=1.0):
    #Definimos el valor de gamma
    gamma = 1.0/gamma
    
    #Mapeamos gamma
    mapeo = np.array([(255 * (i / 255.0) ** gamma) for i in np.arange(0, 256)]).astype("uint8")
    
    #Mapea el canal con los valores gamma
    channel_gamma = mapeo[channel]
    
    #Regresamos el canal con la correcion aplicada
    return channel_gamma

r = correccion_gamma(r, 0.07)

g = correccion_gamma(g, 0.07)

b = correccion_gamma(b, 0.07)

def filtro_logaritmico(h_channel):
    # Convertimos el canal con valores flotantes
    h_float = h_channel.astype(np.float32) + 1.0  # Se añade 1.0 para evitar log(0)
    
    # Calculamos el logaritmo
    h_log = np.log(h_float)
    
    # Normalizamos el resultado logarítmico a un rango de 0 a 180
    h_aclarado = (h_log / np.log(h_float.max())) * 180
    
    # Convertimos el resultado de nuevo a uint8 para su uso en imágenes
    return h_aclarado.astype(np.uint8)

r = filtro_logaritmico(r)

g = filtro_logaritmico(g)

b = filtro_logaritmico(b)

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
    

canny = cv2.Canny(rgb_blur, 30, 130)

gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

#cv2.imshow("Canny", canny)

cv2.imshow("Edege detection", np.hstack([gray, canny]))

cv2.waitKey(0)