import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Main program
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(parser.parse_args())

image_BGR = cv2.imread(args["image"])
image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)

# Se calcula el valor máximo existente en los píxeles de la imagen
max_val = image_RGB.max()

# Constante C para la transformación logarítmica
div = np.log(1.0 + max_val)
C = 255.0 / div
print(C)
# Aplicar la transformación logarítmica

log_correction = lambda img: np.array(C * np.log(1.0 + img), dtype='uint8')
image_RGB_logaritmo = log_correction(image_RGB)

# Se genera una figura para mostrar los resultados con matplotlib
fig = plt.figure(figsize=(14, 10))

# Se marqueta el diseño del gráfico
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 3, 5)

# Se dibuja la imagen original
ax1.imshow(image_RGB)
ax1.set_title('Original Image')

# Se dibuja la imagen con la transformación logarítmica
ax2.imshow(image_RGB_logaritmo)
ax2.set_title('Logarithmic Transformation')

# Se dibujan las gráficas de las funciones
x = np.linspace(0, 255, 255)
# y1 = x
# y2 = C * np.log(1.0 + x)

identity_function = lambda x: x
log_function = lambda x : C * np.log(1.0 + x)

y1=identity_function(x)
y2=log_function(x)

ax3.plot(x, y1, color="r", linewidth=1, label="Id. Func")
msg = "Logarithmic Function"
ax3.plot(x, y2, color="b", linewidth=1, label=msg)
ax3.legend()
ax3.set_title('Logarithmic Function Plot')

plt.show()