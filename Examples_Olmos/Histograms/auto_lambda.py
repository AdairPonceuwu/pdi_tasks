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

# Parámetros para el recorte de contraste
q_low, q_high = float(args["qlow"]), float(args["qhigh"])

# Calcular el histograma de la imagen
hist = cv2.calcHist([image], [0], None, [256], [0, 256])

# Calcular el histograma acumulativo
Hi = np.cumsum(hist)

# Obtenemos el número de pixeles
MN = Hi[-1]

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


#Minimos y máximos
a_min = 0
a_max = 255


x = np.linspace(0, 255, 256)

y2 = np.array((a_min + (x - a_prim_low) * ((a_max - a_min) / (a_prim_high - a_prim_low))).astype(np.uint8))

# Define gamma function using the lookup table
ajuste = lambda m: y2[m]

# Apply the gamma transformation to the image
image_ajustada = np.array(np.vectorize(ajuste)(image), dtype='uint8')

# Buscar los índices de la imagen donde los valores son <= a_low
indices_low = np.where(image <= a_prim_low)
image_ajustada[indices_low] = a_min

# Buscar los índices de la imagen donde los valores son >= a_high
indices_high = np.where(image >= a_prim_high)
image_ajustada[indices_high] = a_max

print(np.shape(image))
print(np.shape(image_ajustada))

# se genera una figura para mostrar los resultados con matplotlib
fig=plt.figure(figsize=(14,10))
# se maqueta el diseño del grafico
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,3,5)
# se dibuja la imagen original
ax1.imshow(image, cmap="gray")
ax1.set_title('Original image')
# se dibuja la imagen co el operador
ax2.imshow(image_ajustada, cmap="gray")
ax2.set_title('Gamma Correction')
#Se dibujan las graficas de las funciones


plt.show()