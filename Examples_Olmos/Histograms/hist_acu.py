import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np

# Argumentos del script
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image")
args = vars(ap.parse_args())

# Cargar la imagen y convertirla a escala de grises
image_BGR = cv2.imread(args["image"])
image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
image = cv2.cvtColor(image_RGB, cv2.COLOR_BGR2GRAY)

# Calcular el histograma
hist = cv2.calcHist([image], [0], None, [256], [0, 256])

# Calcular el histograma acumulativo
hist_cumulative = hist.cumsum()

# Normalizar el histograma acumulativo
hist_cumulative /= hist_cumulative[-1]

# Generar una figura para mostrar los resultados con matplotlib
fig = plt.figure(figsize=(14, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

# Mostrar la imagen original
ax1.imshow(image, cmap='gray')
ax1.set_title("Original Image")

# Mostrar el histograma acumulativo normalizado
ax2.plot(hist_cumulative, color='black')
ax2.set_title("Normalized Cumulative Histogram")

# Mostrar la figura
plt.show()