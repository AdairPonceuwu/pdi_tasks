import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import signal

# Manejo de la señal para cerrar el programa de manera segura
def signal_handler(sig, frame):
    print('Saliendo...')
    cap.release()
    plt.close('all')
    exit(0)

# Configurar el manejador de señales para CTRL+C
signal.signal(signal.SIGINT, signal_handler)

#####Video#####

cap = cv.VideoCapture(0)

# Crear una ventana de Matplotlib con 2 subgráficos (original y enmascarado)
plt.ion()  # Activar el modo interactivo
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Leer el primer fotograma para inicializar las imágenes en los subgráficos
ret, frame = cap.read()
frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

# Inicializar los subgráficos con la imagen original y la enmascarada
im1 = ax1.imshow(frame_rgb)
ax1.set_title('Original')
im2 = ax2.imshow(frame_rgb)
ax2.set_title('Con Máscara')

# Bandera para indicar si se debe cerrar
should_close = False

def on_key(event):
    global should_close
    if event.key == 'e':
        should_close = True

fig.canvas.mpl_connect('key_press_event', on_key)

try:
    while not should_close:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo capturar el video.")
            break

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        lower_h = 35
        lower_s = 50
        lower_v = 50
        upper_h = 85
        upper_s = 255
        upper_v = 255
        
        
        # Definir los rangos inferior y superior
        lower_vals = np.array([lower_h, lower_s, lower_v])
        upper_vals = np.array([upper_h, upper_s, upper_v])

        # Crear una máscara basada en los valores de H, S, V
        mask = cv.inRange(hsv, lower_vals, upper_vals)
        # Aplicar la máscara al fotograma
        masked_frame = cv.bitwise_and(frame, frame, mask=mask)

        # Convertir ambos fotogramas de BGR a RGB
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        masked_frame_rgb = cv.cvtColor(masked_frame, cv.COLOR_BGR2HSV)

        # Actualizar las imágenes en los subgráficos
        im1.set_data(frame_rgb)
        im2.set_data(masked_frame_rgb)

        # Refrescar la figura para mostrar los nuevos fotogramas
        plt.draw()
        plt.pause(0.1)  # Pequeña pausa para actualizar la imagen

finally:
    # Liberar los recursos al salir
    cap.release()
    plt.close('all')
