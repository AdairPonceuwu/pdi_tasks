import cv2
import numpy as np

# Filtro gaussiano para suavizado
def filtro_gaussiano(img, kernel_size=5):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

# Autoajuste restringido de niveles
def autoajuste_restringido(img, low_perc=0.1, high_perc=99.9):
    low_val = np.percentile(img, low_perc)
    high_val = np.percentile(img, high_perc)
    img_rescaled = np.clip((img - low_val) * 255.0 / (high_val - low_val), 0, 255).astype(np.uint8)
    return img_rescaled

# Ecualización de histograma
def ecualizacion_histograma(img):
    return cv2.equalizeHist(img)

# Normalización del histograma
def normalizacion_histograma(img):
    return cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

# Corrección gamma
def correccion_gamma(img, gamma=1.0):
    invGamma = 1.0 / gamma
    tabla = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, tabla)

# Corrección logarítmica
def correccion_logaritmica(img):
    c = 255 / np.log(1 + np.max(img))
    img_log = c * (np.log(img + 1))
    img_log = np.array(img_log, dtype=np.uint8)
    return img_log

# Detección de líneas blancas y amarillas
def procesar_cuadro(cuadro, usar_gamma=True, gamma_valor=1.0):
    alto, ancho = cuadro.shape[:2]

    # Filtro Gaussiano para suavizar la imagen y reducir el ruido
    cuadro_suavizado = filtro_gaussiano(cuadro)

    # Convertir a espacio de color LAB para mejorar la detección de luminosidad
    lab = cv2.cvtColor(cuadro_suavizado, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)

    # Aplicar ecualización de histograma en el canal L
    l_ecualizado = ecualizacion_histograma(l)

    # Aplicar corrección gamma o logarítmica
    if usar_gamma:
        l_corregido = correccion_gamma(l_ecualizado, gamma=gamma_valor)
    else:
        l_corregido = correccion_logaritmica(l_ecualizado)

    # Reconstruir la imagen LAB con el canal L corregido
    lab_ajustado = cv2.merge([l_corregido, a, b])

    # Convertir de nuevo a BGR
    cuadro_ajustado = cv2.cvtColor(lab_ajustado, cv2.COLOR_Lab2BGR)

    # Convertir al espacio de color HSV para la detección de líneas blancas y amarillas
    hsv = cv2.cvtColor(cuadro_ajustado, cv2.COLOR_BGR2HSV)

    # Filtrar áreas brillantes y baja saturación para detectar líneas blancas
    mascara_blanca = cv2.inRange(hsv, (0, 0, 200), (180, 60, 255))

    # Detectar la línea amarilla en el espacio HSV
    mascara_amarilla = cv2.inRange(hsv, (15, 100, 100), (35, 255, 255))

    # Combinar ambas máscaras (blanca y amarilla)
    mascara_combined = cv2.bitwise_or(mascara_blanca, mascara_amarilla)

    # Crear una máscara negra para la mitad superior
    mascara_combined[:alto // 2, :] = 0  # Poner a cero (negro) la mitad superior

    return mascara_combined

# Procesamiento de video
def procesar_video(ruta_video, usar_gamma=True, gamma_valor=1.0):
    cap = cv2.VideoCapture(ruta_video)

    if not cap.isOpened():
        print("Error al abrir el archivo de video.")
        return

    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    salida = cv2.VideoWriter('video_procesado_mejorado.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (ancho, alto), isColor=False)

    cv2.namedWindow('Líneas del carril', cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        procesado = procesar_cuadro(frame, usar_gamma=usar_gamma, gamma_valor=gamma_valor)

        cv2.imshow('Líneas del carril', procesado)
        salida.write(procesado)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    salida.release()
    cv2.destroyAllWindows()

# Llamar a la función para procesar el video
ruta_video = 'D:\\proyectos_opencv\\Primer_Parcial\\lineas.mp4'
procesar_video(ruta_video, usar_gamma=True, gamma_valor=0.035)
