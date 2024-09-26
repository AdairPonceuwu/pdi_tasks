import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=False, help="Path to the input video")
args = vars(ap.parse_args())

# CDF(histograma acumulado)
def cdf(hist):
    return hist.cumsum()

# Autocontraste restringido
def autocontraste_restringido(img, q=0.005):
    # Calcular el histograma de la imagen (256 niveles de intensidad)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()

    # Calcular percentiles
    low_perc = np.percentile(img, q * 100)
    high_perc = np.percentile(img, (1 - q) * 100)

    # Rescaling con los valores calculados
    img_rescaled = np.clip((img - low_perc) * 255 / (high_perc - low_perc), 0, 255).astype(np.uint8)
    
    return img_rescaled

# Corrección gamma
def correccion_gamma(img, gamma=1.0):
    inv_gamma = 1.0 / gamma
    tabla = np.array([(255 * (i / 255.0) ** inv_gamma) for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, tabla)

# Normalización de histograma usando OpenCV
def normalizacion_histograma_god(img, channel_max):
    return cv2.normalize(img, None, 0, channel_max, cv2.NORM_MINMAX)

# Procesar cuadro
def procesar_cuadro(cuadro, gamma_valor=1.0):
    # Convertir todo el cuadro al espacio de color HSV
    hsv = cv2.cvtColor(cuadro, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Procesar canales
    s_res = autocontraste_restringido(s)
    v_res = autocontraste_restringido(v)

    s_normalizado = normalizacion_histograma_god(s_res, 255.0)
    v_normalizado = normalizacion_histograma_god(v_res, 255.0)

    # Aplicar corrección gamma
    v_corregido = correccion_gamma(v_normalizado, gamma=gamma_valor)

    # Reconstruir la imagen HSV
    hsv_ajustado = cv2.merge([h, s_normalizado, v_corregido])

    # Filtrar áreas brillantes y baja saturación para detectar líneas blancas
    mascara_blanca = cv2.inRange(hsv_ajustado, (0, 0, 255), (100, 60, 255))

    # Detectar la línea amarilla en el espacio HSV
    mascara_amarilla = cv2.inRange(hsv_ajustado, (15, 100, 100), (35, 255, 255))

    # Combinar ambas máscaras
    mascara_combined = cv2.bitwise_or(mascara_blanca, mascara_amarilla)

    # Crear una máscara negra para la mitad superior
    alto = cuadro.shape[0]
    mascara_combined[:alto // 2, :] = 0  # Poner a cero (negro) la mitad superior

    return mascara_combined

# Procesar video
def procesar_video(ruta_video, gamma_valor=1.0):
    cap = cv2.VideoCapture(ruta_video if ruta_video else 0)

    if not cap.isOpened():
        print("Error al abrir el archivo de video.")
        return

    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    salida = cv2.VideoWriter('video_procesado.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (ancho, alto), isColor=False)
    cv2.namedWindow('Líneas del carril', cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        procesado = procesar_cuadro(frame, gamma_valor=gamma_valor)
        cv2.imshow('Líneas del carril', procesado)
        salida.write(procesado)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    salida.release()
    cv2.destroyAllWindows()

ruta_video = args["video"]
procesar_video(ruta_video, gamma_valor=1.5)
