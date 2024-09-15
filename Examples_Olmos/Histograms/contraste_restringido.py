import cv2
import matplotlib.pyplot as plt
import numpy as np


def contraste_res(image, q_low, q_high):

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

    print("Numero de pixeles:", MN)
    print("Valor acumulativo para a_prim_low:", np.round(MN * q_low))
    print("Valor acumulativo para a_prim_high:", np.round(MN * (1 - q_high)))
    print("a_prim_low:", a_prim_low)
    print("a_prim_high:", a_prim_high)

    # Creamos una matriz de zeros similar a la imagen original
    # img = np.zeros_like(image)

    #Minimos y máximos
    a_min = 0
    a_max = 255

    x = np.linspace(0, 255, 256)

    y2 = np.array((a_min + (x - a_prim_low) * ((a_max - a_min) / (a_prim_high - a_prim_low))).astype(np.uint8))

    # Define gamma function using the lookup table
    ajuste = lambda m: y2[m]

    # Apply the gamma transformation to the image
    image_ajustada = np.array(np.vectorize(ajuste)(image), dtype='uint8')

    # Ajustar los valores extremos en un solo paso
    image_ajustada[image <= a_prim_low] = a_min
    image_ajustada[image >= a_prim_high] = a_max

    # Dibujar las imagenes y graficar los histogramas
    fig = plt.figure(figsize=[20, 20])

    ax1=fig.add_subplot(2, 3, 1)
    ax2=fig.add_subplot(2, 3, 2)
    ax3=fig.add_subplot(2, 3, 3)

    ax4=fig.add_subplot(2, 3, 4)
    ax5=fig.add_subplot(2, 3, 5)
    ax6=fig.add_subplot(2, 3, 6)

    #Imagen Original
    ax1.imshow(image, cmap='gray')
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
    ax4.imshow(image_ajustada, cmap='gray')
    ax4.set_title('Imagen Ajustada')
    ax4.axis('off')

    # Histograma de la imagen ajustada
    hist_ajustada = cv2.calcHist([image_ajustada], [0], None, [256], [0, 256])
    hist_ajustada /= hist_ajustada.sum()
    Hi_aj = np.cumsum(hist_ajustada)

    ax5.plot(hist_ajustada, color='black')
    ax5.set_title('Histograma de Imagen Ajustada')

    ax6.plot(Hi_aj, color='black')
    ax6.set_title('Histograma Acumulativo Imagen Ajustada')

    plt.show()