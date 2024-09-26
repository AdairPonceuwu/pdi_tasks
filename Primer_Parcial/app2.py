import cv2
import numpy as np

# Autoajuste restringido de niveles
def autoajuste_restringido(img, low_perc=1, high_perc=99):
    low_val = np.percentile(img, low_perc)
    high_val = np.percentile(img, high_perc)
    img_rescaled = np.clip((img - low_val) * 255.0 / (high_val - low_val), 0, 255).astype(np.uint8)
    return img_rescaled

# Ecualización de histograma acumulativo
def ecualizacion_histograma_acumulado(img):
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

# Detección de líneas blancas y amarillas con correcciones gamma/logarítmica y ecualización de H, S y V
def procesar_cuadro(cuadro, usar_gamma=True, gamma_valor=1.0):
    # Obtener el tamaño del cuadro
    alto, ancho = cuadro.shape[:2]

    # Convertir todo el cuadro al espacio de color HSV
    hsv = cv2.cvtColor(cuadro, cv2.COLOR_BGR2HSV)

    # Separar los canales H, S, V
    h, s, v = cv2.split(hsv)

    # Ecualizar
    s_ecualizado = ecualizacion_histograma_acumulado(s)

    v_ecualizado = ecualizacion_histograma_acumulado(v)
    
    # Normalización     
    v_normalizado = normalizacion_histograma(v_ecualizado)
    
    # Aplicar autoajuste restringido
        
    v_ajustado = autoajuste_restringido(v_normalizado)
    
    # Aplicar corrección gamma o logarítmica en el canal V ajustado
    if usar_gamma:
        v_corregido = correccion_gamma(v_ajustado, gamma=gamma_valor)

    
    # Reconstruir la imagen HSV con el canal H intacto, el canal S ecualizado, y el canal V corregido
    hsv_ajustado = cv2.merge([h, s_ecualizado, v_corregido])

    # Convertir de vuelta a espacio BGR
    cuadro_ajustado = cv2.cvtColor(hsv_ajustado, cv2.COLOR_HSV2BGR)

    # Convertir de nuevo a HSV para la detección de líneas
    hsv_ajustado = cv2.cvtColor(cuadro_ajustado, cv2.COLOR_BGR2HSV)
    
    # Filtrar áreas brillantes y baja saturación para detectar líneas blancas
    mascara_blanca = cv2.inRange(hsv_ajustado, (0, 0, 20), (30, 40, 255))

    # Detectar la línea amarilla en el espacio HSV
    mascara_amarilla = cv2.inRange(hsv_ajustado, (10, 20, 20), (35, 255, 255))

    # Combinar ambas máscaras (blanca y amarilla)
    mascara_combined = cv2.bitwise_or(mascara_blanca, mascara_amarilla)

    # Crear una máscara negra para la mitad superior
    mascara_combined[:alto // 2, :] = 0  # Poner a cero (negro) la mitad superior

    return mascara_combined

# Procesamiento de video
def procesar_video(ruta_video, usar_gamma=True, gamma_valor=1.0):
    # Leer el video
    cap = cv2.VideoCapture(ruta_video)

    if not cap.isOpened():
        print("Error al abrir el archivo de video.")
        return

    # Obtener propiedades del video
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Guardar el video procesado
    salida = cv2.VideoWriter('video_procesado_HSV.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (ancho, alto), isColor=False)

    # Crear una ventana para mostrar el video
    cv2.namedWindow('Líneas del carril', cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Procesar cada cuadro con ecualización en S y V
        procesado = procesar_cuadro(frame, usar_gamma=usar_gamma, gamma_valor=gamma_valor)

        # Mostrar el video procesado en la ventana 'Líneas del carril'
        cv2.imshow('Líneas del carril', procesado)

        # Guardar el cuadro procesado en el archivo de salida
        salida.write(procesado)
        
        # Esperar 1 milisegundo entre cada cuadro
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar el objeto de captura y de salida, y cerrar las ventanas
    cap.release()
    salida.release()
    cv2.destroyAllWindows()

# Llamar a la función para procesar el video con ecualización en S y V
ruta_video = 'D:\\proyectos_opencv\\Primer_Parcial\\lineas.mp4'
procesar_video(ruta_video, usar_gamma=True, gamma_valor=0.035)
