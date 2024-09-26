import cv2
import numpy as np

# Autoajuste restringido de niveles
def autoajuste_restringido(img, low_perc=0.1, high_perc=99.9):
    low_val = np.percentile(img, low_perc)
    high_val = np.percentile(img, high_perc)
    img_rescaled = np.clip((img - low_val) * 255.0 / (high_val - low_val), 0, 255).astype(np.uint8)
    return img_rescaled

# Ecualización de histograma
def ecualizacion_histograma(img):
    return cv2.equalizeHist(img)

# Corrección gamma
def correccion_gamma(img, gamma=1.0):
    invGamma = 1.0 / gamma
    tabla = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, tabla)

# Filtro logarítmico de aclarado
def correccion_logaritmica(img):
    # Convertir a flotante
    img_float = img.astype(np.float32)
    # Aplicar la transformación logarítmica
    c = 255 / np.log(1 + np.max(img_float))
    log_img = c * np.log(1 + img_float)
    # Convertir de vuelta a uint8
    log_img = np.uint8(log_img)
    return log_img

# Operador negativo
def operador_negativo(img):
    return 255 - img

# Detección de líneas blancas y amarillas con corrección gamma/logarítmica en LAB
def procesar_cuadro(cuadro, usar_gamma=True, gamma_valor=1.0):
    # Obtener el tamaño del cuadro
    alto, ancho = cuadro.shape[:2]

    # Convertir todo el cuadro al espacio de color LAB
    lab = cv2.cvtColor(cuadro, cv2.COLOR_BGR2LAB)

    # Separar los canales L, A, B
    l, a, b = cv2.split(lab)

    # --- Operaciones sobre el canal L (luminosidad) ---
    # Aplicar autoajuste restringido
    l_res = autoajuste_restringido(l)

    # Aplicar ecualización de histograma en el canal L ajustado
    l_ecualizado = ecualizacion_histograma(l_res)

    # Aplicar corrección gamma o logarítmica en el canal L ecualizado
    if usar_gamma:
        l_corregido = correccion_gamma(l_ecualizado, gamma=gamma_valor)
    else:
        l_corregido = correccion_logaritmica(l_ecualizado)

    # --- Opcional: Aplicar operador negativo al canal A o B si es necesario ---
    # a_negativo = operador_negativo(a)  # Si quieres probarlo

    # Reconstruir la imagen LAB con el canal L ajustado
    lab_ajustado = cv2.merge([l_corregido, a, b])

    # Convertir de vuelta al espacio BGR para su posterior procesamiento
    cuadro_ajustado = cv2.cvtColor(lab_ajustado, cv2.COLOR_LAB2BGR)

    # Convertir el cuadro ajustado a HSV para la detección de colores
    hsv_ajustado = cv2.cvtColor(cuadro_ajustado, cv2.COLOR_BGR2HSV)

    # Filtrar áreas brillantes y baja saturación para detectar líneas blancas
    mascara_blanca = cv2.inRange(hsv_ajustado, (0, 0, 200), (180, 60, 255))

    # Detectar la línea amarilla en el espacio HSV
    mascara_amarilla = cv2.inRange(hsv_ajustado, (15, 100, 100), (35, 255, 255))

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
    salida = cv2.VideoWriter('video_procesado_lab_opt.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (ancho, alto), isColor=False)

    # Crear una ventana para mostrar el video
    cv2.namedWindow('Líneas del carril', cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Procesar cada cuadro con las operaciones permitidas en LAB
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

# Llamar a la función para procesar el video
ruta_video = 'D:\\proyectos_opencv\\Primer_Parcial\\lineas.mp4'
procesar_video(ruta_video, usar_gamma=True, gamma_valor=0.08)
