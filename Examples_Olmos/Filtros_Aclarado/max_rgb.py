import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Argumentos para decidir la imagen a procesar
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', required=True, help='Ruta de la imagen')
args = vars(parser.parse_args())

# Leer imagen y convertirla a escala de grises
image_RGB = cv2.imread(args['image'])
image_gray = cv2.cvtColor(image_RGB, cv2.COLOR_BGR2GRAY)

# se calcula el valor maximo existente en los pixeles en los pixeles de la imagen
# para obtener el negativo de la imagen se resta el valor maximo menos el valor de cada pixel
max = image_RGB.max()
image_RGB_negative = max - image_RGB

# se genera una figura para mostrar los resultados con matplotlib
flig=plt.figure(figsize=(14,10))
# se maqueta el diseño del grafico
ax1 = flig.add_subplot(2,2,1)
ax2 = flig.add_subplot(2,2,2)
ax3 = flig.add_subplot(2,2,3)
# se dibuja la imagen original
ax1.imshow(image_RGB)
ax1.set_title('Imagen Original')
# se dibuja la imagen co el operador
ax2.imshow(image_RGB_negative)
ax2.set_title('Negativo')
#Se dibujan las graficas de las funciones
x=np.linspace(0,255,255)
y1=x
y2=np.array(255-x)
ax3.plot(x,y1,color="r",linewidth=1,label = "Id.Func.")
msg="Func. Negativa"
ax3.plot(x,y2,color="b",linewidth=1,label=msg)
ax3.legend()
ax3.set_title("Funcion negativa")
plt.show()