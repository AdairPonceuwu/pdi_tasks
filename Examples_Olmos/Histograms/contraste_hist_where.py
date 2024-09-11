import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def contraste_restringido(image, q_low=0.1, q_high=0.9):
    # Calcular los percentiles
    a_prim_low = np.percentile(image, q_low * 100)
    a_prim_high = np.percentile(image, q_high * 100)
    
    print(a_prim_low)
    print(a_prim_high)
    # Calcular los nuevos valores de contraste restringido
    a_min, a_max = 0, 255
    
    # Crear una copia de la imagen para el ajuste
    # image_adjusted = np.zeros_like(image)

    # # Mapear los índices de la imagen donde los valores son <= a_low
    # indices_low = np.where(image <= a_low)
    # image_adjusted[indices_low] = a_min

    # # Mapear los índices de la imagen donde los valores son >= a_high
    # indices_high = np.where(image >= a_high)
    # image_adjusted[indices_high] = a_max

    # # Mapear los índices de la imagen donde los valores están entre a_low y a_high
    # indices_mid = np.where((image > a_low) & (image < a_high))
    # image_adjusted[indices_mid] = (a_min + (image[indices_mid] - a_low) * ((a_max - a_min) / (a_high - a_low))).astype(np.uint8)
    

    # return image_adjusted
    
    # Crea una copia de la imagen para ajustarla
    img = np.zeros_like(image)

    # Itera sobre cada píxel de la imagen usando enumerate
    for i, row in enumerate(image):  # Itera sobre cada fila
        for j, value in enumerate(row):  # Itera sobre cada columna
            if value <= a_prim_low:
                img[i, j] = np.uint8(a_min)
            elif a_prim_low < value < a_prim_high:
                img[i, j] = np.uint8(a_min + (value - a_prim_low) * ((a_max - a_min) / (a_prim_high - a_prim_low)))
            else:  # value >= a_prim_high
                img[i, j] = np.uint8(a_max)
    
    return img

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image")
args = vars(ap.parse_args())

image_BGR = cv2.imread(args["image"])
image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
image = cv2.cvtColor(image_RGB, cv2.COLOR_BGR2GRAY)
# Aplicar el ajuste de contraste restringido
imagen_ajustada = contraste_restringido(image, q_low=0.1, q_high=0.9)

# Calcular el histograma de la imagen original
hist_original = cv2.calcHist([image], [0], None, [256], [0, 256])

# Calcular el histograma de la imagen ajustada
hist_ajustada = cv2.calcHist([imagen_ajustada], [0], None, [256], [0, 256])

# Mostrar la imagen original y ajustada junto con sus histogramas
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Imagen original
axs[0, 0].imshow(image, cmap='gray')
axs[0, 0].set_title('Imagen Original')
axs[0, 0].axis('off')

# Histograma de la imagen original
axs[0, 1].plot(hist_original, color='black')
axs[0, 1].set_title('Histograma de Imagen Original')
axs[0, 1].set_xlim([0, 256])

# Imagen ajustada
axs[1, 0].imshow(imagen_ajustada, cmap='gray')
axs[1, 0].set_title('Imagen Ajustada')
axs[1, 0].axis('off')

# Histograma de la imagen ajustada
axs[1, 1].plot(hist_ajustada, color='black')
axs[1, 1].set_title('Histograma de Imagen Ajustada')
axs[1, 1].set_xlim([0, 256])

plt.tight_layout()
plt.show()
