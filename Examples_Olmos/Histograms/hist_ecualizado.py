import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image")
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

#Ocultar todos los elementos de Hi que sean iguales a 0.
ecua_hi=np.ma.masked_equal(Hi, 0)

#Ecuacion para ecualizar el histograma
ecua_hi = ((ecua_hi - ecua_hi.min()) * 255) / ((MN))

#Remplazar todos los valores enmascarados anteriormente con un valor de 0
ecua_hi_mask = np.ma.filled(ecua_hi,0).astype('uint8')

#Remapeo de la imagen 
image_eq = ecua_hi_mask[image]

# Dibujar las imagenes y graficar los histogramas
fig = plt.figure(figsize=[10, 10])

ax1=fig.add_subplot(2, 3, 1)
ax2=fig.add_subplot(2, 3, 2)
ax3=fig.add_subplot(2, 3, 3)

ax4=fig.add_subplot(2, 3, 4)
ax5=fig.add_subplot(2, 3, 5)
ax6=fig.add_subplot(2, 3, 6)

#Imagen Original
image_Gray = cv2.merge([image, image, image])
ax1.imshow(image_Gray, cmap='gray')
ax1.set_title('Imagen')
ax1.axis('off')

# Histograma de la imagen normal
hist /= hist.sum()
Hi = np.cumsum(hist)
ax2.plot(hist, color='black')
ax2.set_title('Histograma de Imagen')

ax3.plot(Hi, color='black')
ax3.set_title('Histograma Acumulativo Imagen')

# Imagen ajustada
ax4.imshow(image_eq, cmap='gray')
ax4.set_title('Imagen Ajustada')
ax4.axis('off')

# Histograma de la imagen ajustada
hist_ajustada = cv2.calcHist([image_eq], [0], None, [256], [0, 256])
hist_ajustada /= hist_ajustada.sum()
Hi_aj = np.cumsum(hist_ajustada)

ax5.plot(hist_ajustada, color='black')
ax5.set_title('Histograma de Imagen Ajustada')

ax6.plot(Hi_aj, color='black')
ax6.set_title('Histograma Acumulativo Imagen Ajustada')

plt.tight_layout()
plt.show()