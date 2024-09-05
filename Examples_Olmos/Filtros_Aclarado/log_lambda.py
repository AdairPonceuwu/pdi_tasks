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

# Filtro Logaritmico de aclarado
# Create a lookup table for logarithmic transformation
x = np.linspace(0, 255, 256)  # Corrected to include 256 values
r = 255.0
print(r)
C = 255 / (np.log(1 + r))
print(C)

# Logarithmic transformation lookup table
y2 = np.array(C * np.log(1 + x), dtype="uint8")

# Define gamma function using the lookup table
log = lambda m: y2[m]

# Apply the logarithmic transformation to the image
image_RGB_logaritmo = np.array(np.vectorize(log)(image_RGB), dtype='uint8')

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

ax3.plot(x, x, color="r", linewidth=1, label="Id. Func")
msg = "Logarithmic Function"
ax3.plot(x, y2, color="b", linewidth=1, label=msg)
ax3.legend()
ax3.set_title('Logarithmic Function Plot')

plt.show()