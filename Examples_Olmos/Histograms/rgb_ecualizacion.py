#Tranformar RGB A HSV
#Histograma e Histograma acumulativo sobre H
#Transformacion de Mapeo sobre H
#Corregir los valores sobre H
#Nuevo valor H->HSV a RGB

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image")
args = vars(ap.parse_args())

image_BGR = cv2.imread(args["image"])
image_HSV = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2HSV)

(h, s, v) = cv2.split(image_HSV)

# Calcular el histograma del canal H (matiz)
hist = cv2.calcHist([h], [0], None, [180], [0, 180])

# Calcular el histograma acumulativo
Hi = np.cumsum(hist)

# Obtener el número total de píxeles
MN = Hi[-1]

# Ocultar todos los elementos de Hi que sean iguales a 0
ecua_hi = np.ma.masked_equal(Hi, 0)

# Ecuación para ecualizar el histograma
ecua_hi = ((ecua_hi - ecua_hi.min()) * 180) / ((MN))

# Remplazar todos los valores enmascarados con un valor de 0
ecua_hi_mask = np.ma.filled(ecua_hi, 0).astype('uint8')

# Remapeo del canal H usando el histograma ecualizado
h_eq = ecua_hi_mask[h]

# Unir los canales H ecualizado, y los canales S y V originales
image_HSV_eq = cv2.merge([h_eq, s, v])

# Convertir la imagen HSV de vuelta a RGB
img_eq = cv2.cvtColor(image_HSV_eq, cv2.COLOR_HSV2BGR)


# Dibujar las imagenes y graficar los histogramas
fig = plt.figure(figsize=[10, 10])

ax1=fig.add_subplot(2, 3, 1)
ax2=fig.add_subplot(2, 3, 2)
ax3=fig.add_subplot(2, 3, 3)

ax4=fig.add_subplot(2, 3, 4)
ax5=fig.add_subplot(2, 3, 5)
ax6=fig.add_subplot(2, 3, 6)
# Mostrar imagen original
ax1.imshow(cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB))
ax1.set_title('Imagen Original')
ax1.axis('off')

# Histograma de la imagen original
colors = ('b', 'g', 'r')
for i, color in enumerate(colors):
    hist = cv2.calcHist([image_BGR[:, :, i]], [0], None, [256], [0, 256])
    hist /= hist.sum()
    Hi = np.cumsum(hist)
    if i == 0:
        ax2.plot(hist, color=color)
        ax2.set_title('Histograma Imagen Original (RGB)')
        ax3.plot(Hi, color=color)
        ax3.set_title('Histograma Acumulativo Imagen Original (RGB)')
    else:
        ax2.plot(hist, color=color)
        ax3.plot(Hi, color=color)

# Mostrar imagen ajustada
ax4.imshow(cv2.cvtColor(img_eq, cv2.COLOR_BGR2RGB))
ax4.set_title('Imagen Ajustada')
ax4.axis('off')

# Histograma de la imagen ajustada
for i, color in enumerate(colors):
    hist_ajustada = cv2.calcHist([img_eq[:, :, i]], [0], None, [256], [0, 256])
    hist_ajustada /= hist_ajustada.sum()
    Hi_aj = np.cumsum(hist_ajustada)
    if i == 0:
        ax5.plot(hist_ajustada, color=color)
        ax5.set_title('Histograma Imagen Ajustada (RGB)')
        ax6.plot(Hi_aj, color=color)
        ax6.set_title('Histograma Acumulativo Imagen Ajustada (RGB)')
    else:
        ax5.plot(hist_ajustada, color=color)
        ax6.plot(Hi_aj, color=color)

plt.tight_layout()
plt.show()