import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image")
ap.add_argument("-ql", "--qlow", required=True, help="Value for q_low")
ap.add_argument("-qh", "--qhigh", required=True, help="Value for q_high")
args = vars(ap.parse_args())

image_BGR = cv2.imread(args["image"])
image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
image = cv2.cvtColor(image_RGB, cv2.COLOR_BGR2GRAY)

# Calcular el histograma de la imagen
hist = cv2.calcHist([image], [0], None, [256], [0, 256])

# Calcular el histograma acumulativo
Hi = np.cumsum(hist)

# Obtenemos el número de pixeles
MN = Hi[-1]

# Parámetros para el recorte de contraste
q_low, q_high = float(args["qlow"]), float(args["qhigh"])

# Inicializamos los valores primos
a_prim_low = 0
a_prim_high = 0

# Encontrar el valor de a_prim_low
for i, value in enumerate(Hi):
    if value >= np.round(MN * q_low):
        a_prim_low = i
        break

# Encontrar el valor de a_prim_high
for i, value in enumerate(Hi):
    if value >= np.round(MN * (1 - q_high)):
        a_prim_high = i
        break

print("Numero de pixeles:", MN)
print("Valor acumulativo para a_prim_low:", np.round(MN * q_low))
print("Valor acumulativo para a_prim_high:", np.round(MN * (1 - q_high)))
print("a_prim_low:", a_prim_low)
print("a_prim_high:", a_prim_high)

# Creamos una matriz de zeros similar  ala imagen original
img = np.zeros_like(image)

#Minimos y máximos
a_min = 0
a_max = 255

for i, row in enumerate(image): 
    for j, value in enumerate(row):  
        if value <= a_prim_low:
            img[i, j] = np.uint8(a_min)
        elif a_prim_low < value < a_prim_high:
            img[i, j] = np.uint8(a_min + (value - a_prim_low) * ((a_max - a_min) / (a_prim_high - a_prim_low)))
        else:  # value >= a_prim_high
            img[i, j] = np.uint8(a_max)

# Dibujar las imagenes y graficar los histogramas
fig = plt.figure(figsize=[14, 14])

ax1=fig.add_subplot(2, 2, 1)
ax2=fig.add_subplot(2, 2, 2)
ax3=fig.add_subplot(2, 2, 3)
ax4=fig.add_subplot(2, 2, 4)

#Imagen Original
ax1.imshow(image, cmap='gray')
ax1.set_title('Imagen')
ax1.axis('off')

# Histograma de la imagen normal
hist = cv2.calcHist([image], [0], None, [256], [0, 256])

ax2.plot(hist, color='black')
ax2.set_title('Histograma de Imagen')

# Imagen ajustada
ax3.imshow(img, cmap='gray')
ax3.set_title('Imagen Ajustada')
ax3.axis('off')

# Histograma de la imagen ajustada
hist_ajustada = cv2.calcHist([img], [0], None, [256], [0, 256])

ax4.plot(hist_ajustada, color='black')
ax4.set_title('Histograma de Imagen Ajustada')

plt.show()