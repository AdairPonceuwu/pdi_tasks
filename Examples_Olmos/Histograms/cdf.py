import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Función que implementa el algoritmo usando operaciones vectorizadas
def f_hs(h_A, L_R):
    K = len(h_A)
    P_A = np.cumsum(h_A) / np.sum(h_A)  # CDF de h_A

    # Convertimos los puntos de control a arrays de numpy
    a_vals = np.array([pt[0] for pt in L_R])
    q_vals = np.array([pt[1] for pt in L_R])

    # Inicializamos el array de mapeo
    f_hs = np.zeros(K, dtype=np.uint8)

    # Para b <= q0, asignamos a' = 0
    f_hs[P_A <= q_vals[0]] = 0

    # Para b >= qN, asignamos a' = K - 1
    f_hs[P_A >= q_vals[-1]] = K - 1

    # Para los demás valores, interpolamos linealmente
    mask = (P_A > q_vals[0]) & (P_A < q_vals[-1])
    P_A_masked = P_A[mask]

    # Encuentra en qué intervalo cae cada valor de P_A
    indices = np.searchsorted(q_vals, P_A_masked) - 1
    indices = np.clip(indices, 0, len(q_vals) - 2)  # Limitar índices a rangos válidos

    # Interpolación lineal entre a_n y a_{n+1}
    a_n = a_vals[indices]
    a_n1 = a_vals[indices + 1]
    q_n = q_vals[indices]
    q_n1 = q_vals[indices + 1]

    f_hs[mask] = a_n + ((P_A_masked - q_n) * (a_n1 - a_n)) / (q_n1 - q_n)

    return f_hs

# Argumentos de entrada
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image")
args = vars(ap.parse_args())

# Cargar la imagen en escala de grises
image_BGR = cv2.imread(args["image"])
image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
image = cv2.cvtColor(image_RGB, cv2.COLOR_BGR2GRAY)

image = cv2.merge([image,image,image])

# Calcular el histograma de la imagen original
hist_original = cv2.calcHist([image], [0], None, [256], [0, 256])

# Definir la función de distribución de referencia L_R
L_R = [(0, 0.0), (64, 0.25), (128, 0.5), (192, 0.75), (255, 1.0)]

# Obtener la función de mapeo f_hs
f_hs_mapping = f_hs(hist_original, L_R)

# Aplicar el mapeo a la imagen original
mapped_image = f_hs_mapping[image]

# Mostrar la imagen original y la nueva imagen mapeada
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("CDF")
plt.imshow(mapped_image, cmap='gray')

plt.show()

# Guardar la imagen resultante
cv2.imwrite("mapped_image.png", cv2.cvtColor(mapped_image, cv2.COLOR_GRAY2BGR))
