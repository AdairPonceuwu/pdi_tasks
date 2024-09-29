import cv2
import numpy as np

# Función de nada para usar con los sliders
def nada(x):
    pass

# Detección de líneas blancas y amarillas con rangos de HSV ajustables
def procesar_cuadro(cuadro, low_h, low_s, low_v, high_h, high_s, high_v):
    # Convertir el cuadro al espacio de color HSV
    hsv = cv2.cvtColor(cuadro, cv2.COLOR_BGR2HSV)

    # Crear la máscara con los valores actuales de los sliders
    mascara = cv2.inRange(hsv, (low_h, low_s, low_v), (high_h, high_s, high_v))

    return mascara

# Procesamiento de video con control interactivo
def procesar_video_interactivo(ruta_video):
    # Leer el video
    cap = cv2.VideoCapture(ruta_video)

    if not cap.isOpened():
        print("Error al abrir el archivo de video.")
        return

    # Crear una ventana para mostrar el video
    cv2.namedWindow('Líneas del carril', cv2.WINDOW_NORMAL)
    
    # Crear una ventana para los sliders
    cv2.namedWindow('Ajustes HSV')

    # Crear los sliders para ajustar los valores HSV
    cv2.createTrackbar('Low H', 'Ajustes HSV', 0, 180, nada)
    cv2.createTrackbar('High H', 'Ajustes HSV', 180, 180, nada)
    cv2.createTrackbar('Low S', 'Ajustes HSV', 0, 255, nada)
    cv2.createTrackbar('High S', 'Ajustes HSV', 60, 255, nada)  # Ajuste inicial para saturación baja (para líneas blancas)
    cv2.createTrackbar('Low V', 'Ajustes HSV', 200, 255, nada)
    cv2.createTrackbar('High V', 'Ajustes HSV', 255, 255, nada)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Leer los valores actuales de los sliders
        low_h = cv2.getTrackbarPos('Low H', 'Ajustes HSV')
        high_h = cv2.getTrackbarPos('High H', 'Ajustes HSV')
        low_s = cv2.getTrackbarPos('Low S', 'Ajustes HSV')
        high_s = cv2.getTrackbarPos('High S', 'Ajustes HSV')
        low_v = cv2.getTrackbarPos('Low V', 'Ajustes HSV')
        high_v = cv2.getTrackbarPos('High V', 'Ajustes HSV')

        # Procesar el cuadro con los valores actuales de HSV
        procesado = procesar_cuadro(frame, low_h, low_s, low_v, high_h, high_s, high_v)

        # Mostrar el video procesado en la ventana
        cv2.imshow('Líneas del carril', procesado)

        # Esperar 1 milisegundo entre cada cuadro
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar el objeto de captura y cerrar las ventanas
    cap.release()
    cv2.destroyAllWindows()

# Llamar a la función para procesar el video con control interactivo
ruta_video = 'D:\\proyectos_opencv\\Primer_Parcial\\lineas.mp4'
procesar_video_interactivo(ruta_video)
