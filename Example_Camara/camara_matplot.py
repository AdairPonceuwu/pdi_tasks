import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import signal

# Manejo de la señal para cerrar el programa de manera segura
def signal_handler(sig, frame):
    print('Saliendo...')
    cap.release()
    plt.close('all')
    exit(0)

# Configurar el manejador de señales para CTRL+C
signal.signal(signal.SIGINT, signal_handler)

# Capturar el video desde la cámara (0 para la cámara predeterminada)
cap = cv.VideoCapture(0)

# Crear una ventana de Matplotlib con 2 subgráficos (original y enmascarado)
plt.ion()  # Activar el modo interactivo
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Leer el primer fotograma para inicializar las imágenes en los subgráficos
ret, frame = cap.read()
frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

# Inicializar los subgráficos con la imagen original y la enmascarada
im1 = ax1.imshow(frame_rgb)
ax1.set_title('Original')
im2 = ax2.imshow(frame_rgb)
ax2.set_title('Con Máscara')

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo capturar el video.")
            break

        
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HLS)

        
        lower = np.array([35, 50, 50])
        upper = np.array([85, 255, 255])

        # Crear una máscara basada en los valores de H, S, V
        mask = cv.inRange(hsv, lower, upper)
        # Aplicar la máscara al fotograma
        masked_frame = cv.bitwise_and(frame, frame, mask=mask)

        # Convertir ambos fotogramas de BGR a RGB
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        masked_frame_rgb = cv.cvtColor(masked_frame, cv.COLOR_BGR2RGB)
        
        # Actualizar las imágenes en los subgráficos
        im1.set_data(frame_rgb)
        im2.set_data(masked_frame_rgb)
        
        # Refrescar la figura para mostrar los nuevos fotogramas
        plt.draw()
        plt.pause(0.1)  # Pequeña pausa para actualizar la imagen

        # Si se presiona la tecla 'e', se sale del bucle
        if cv.waitKey(1) & 0xFF == ord('e'):
            break

finally:
    # Liberar los recursos al salir
    cap.release()
    plt.close('all')
    
