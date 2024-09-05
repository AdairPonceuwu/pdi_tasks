import argparse
import matplotlib.pyplot as plt
import numpy as np
import cv2

parser = argparse.ArgumentParser()

# Argumentos para decidir la imagen a procesar
parser.add_argument('-i', '--image', required=True, help='Ruta de la imagen')
parser.add_argument("-g", "--gamma", required=True, help= "Gamma value")
args = vars(parser.parse_args())

# Leer imagen y convertirla a escala de grises
image_BGR = cv2.imread(args['image'])
image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
gamma = float(args["gamma"])

x = np.linspace(0, 255, 256)  # Corrected to include 256 values

# Logarithmic transformation lookup table
y2 = np.array(255 * (x / 255) ** gamma, dtype="uint8")

# Define gamma function using the lookup table
gamma = lambda m: y2[m]

# Apply the gamma transformation to the image
image_RGB_gamma = np.array(np.vectorize(gamma)(image_RGB), dtype='uint8')


# se genera una figura para mostrar los resultados con matplotlib
fig=plt.figure(figsize=(14,10))
# se maqueta el diseño del grafico
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,3,5)
# se dibuja la imagen original
ax1.imshow(image_RGB)
ax1.set_title('Original image')
# se dibuja la imagen co el operador
ax2.imshow(image_RGB_gamma)
ax2.set_title('Gamma Correction')
#Se dibujan las graficas de las funciones


ax3.plot(x,x,color="r",linewidth=1,label = "Id.Func.")
msg="Func. Negativa"
ax3.plot(x,y2,color="b",linewidth=1,label=msg)
ax3.legend()
ax3.set_title("Funcion gamma")
plt.show()

