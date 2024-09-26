import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="Path to the input video")
args = vars(ap.parse_args())

# CDF(histograma acumulado)
def cdf(hist):
    cdf = hist.cumsum()  
    return cdf

# CDF normalizado
def cdf_normalizado(hist, cdf_hist):
    cdf_normalizado = cdf_hist * float(hist.max()) / cdf_hist.max() 
    return cdf_normalizado
    
# Autocontraste_restringido
def autocontraste_restringido(channel, q=0.05):
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    
    hist_acumulado = cdf(hist)
    
    # Calcular la CDF (histograma acumulado)
    cdf_norma = cdf_normalizado(hist, hist_acumulado)
    
    # Calcular low_percentile y high_percentile basados en q
    low_perc = q * 100  
    high_perc = (1 - q) * 100 
    
    # Encontrar los valores en la CDF correspondientes a los percentiles deseados
    a_prim_low = np.searchsorted(cdf_norma, low_perc / 100.0)  
    a_prim_high = np.searchsorted(cdf_norma, high_perc / 100.0)  
    
    # Minimos y máximos
    a_min = 0
    a_max = 255
    
    # Verificar si a_prim_high es igual a a_prim_low para evitar división por cero
    if a_prim_high == a_prim_low:
        # Si ambos son iguales, devolvemos el canal
        channel_rescaled = channel.copy()
    else:
        # Autocontraste de los valores
        channel_rescaled = np.clip((channel - a_prim_low) * a_max / (a_prim_high - a_prim_low), a_min, a_max).astype(np.uint8)
    
    return channel_rescaled

# Ecualización de histograma
def ecualizacion_histograma(channel):
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    
    # Calcular la CDF (histograma acumulado)
    hist_acumulado = cdf(hist)

    #Ocultar todos los elementos de Hi que sean iguales a 0.
    ecua_hi = np.ma.masked_equal(hist_acumulado, 0)

    #Ecuacion para ecualizar el histograma
    ecua_hi = ((ecua_hi - ecua_hi.min()) * 255) / hist_acumulado[-1]

    #Remplazar todos los valores enmascarados anteriormente con un valor de 0
    ecua_hi_mask = np.ma.filled(ecua_hi,0).astype('uint8')

    #Remapeo del canal
    channel_eq = ecua_hi_mask[channel]
    
    return channel_eq

# Normalizacion con base a los valores de los canales
def normalizacion_histograma(channel, channel_max):
    channel = channel.astype(np.float32)
    
    min_val = np.min(channel)
    max_val = np.max(channel)
    
    if max_val - min_val != 0: 
        normalized_channel = (channel - min_val) * (channel_max / (max_val - min_val))
    else:
        normalized_channel = channel.copy() 

    normalized_channel = normalized_channel.astype(np.uint8)
    
    return normalized_channel

# Corrección gamma
def correccion_gamma(channel, gamma=1.0):
    gamma = 1.0/gamma
    
    mapeo = np.array([(255 * (i / 255.0) ** gamma) for i in np.arange(0, 256)]).astype("uint8")
    
    channel_gamma = mapeo[channel]
    
    return channel_gamma

def filtro_logaritmico(h_channel):
    h_float = h_channel.astype(np.float32) + 1.0  # Evitar log(0)
    h_log = np.log(h_float)
    h_aclarado = (h_log / np.log(h_float.max())) * 180
    return h_aclarado.astype(np.uint8)


# Detección de líneas blancas y amarillas con correcciones gamma/logarítmica
def procesar_cuadro(cuadro, gamma_valor=1.0):
    # Convertir todo el cuadro al espacio de color HSV
    hsv = cv2.cvtColor(cuadro, cv2.COLOR_BGR2HSV)

    # Separar los canales H, S, V
    h, s, v = cv2.split(hsv)

    # Aplicar ecualización de histograma primero
    s_ecualizado = ecualizacion_histograma(s)
    v_ecualizado = ecualizacion_histograma(v)

    # Luego aplicar el autocontraste restringido
    s_res = autocontraste_restringido(s_ecualizado)
    v_res = autocontraste_restringido(v_ecualizado)

    # Normalización post-procesamiento para garantizar que los valores estén en el rango adecuado
    s_normalizado = normalizacion_histograma(s_res, 255.0)
    v_normalizado = normalizacion_histograma(v_res, 255.0)
    h_normalizado = normalizacion_histograma(h, 180.0)
        
    #Aplicar corrección gamma o logarítmica en el canal V
    h_corregido = filtro_logaritmico(h_normalizado)
    v_corregido = correccion_gamma(v_normalizado, gamma=gamma_valor)
    
    # Reconstruir la imagen HSV con el canal V ajustado
    hsv_ajustado = cv2.merge([h_corregido, s_normalizado, v_corregido])
    
    # Filtrar áreas brillantes y baja saturación para detectar líneas blancas
    mascara_blanca = cv2.inRange(hsv_ajustado, (0, 0, 200), (180, 60, 255))

    # Detectar la línea amarilla en el espacio HSV
    mascara_amarilla = cv2.inRange(hsv_ajustado, (5, 200, 200), (120, 255, 255))

    # Combinar ambas máscaras (blanca y amarilla)
    mascara_combined = cv2.bitwise_or(mascara_blanca, mascara_amarilla)
    
    # Obtener el tamaño del cuadro
    alto, _ = cuadro.shape[:2]

    # Crear una máscara negra para la mitad superior
    mascara_combined[:alto // 2, :] = 0  # Poner a cero (negro) la mitad superior

    return mascara_combined

# Procesamiento de video
def procesar_video(ruta_video, gamma_valor=1.0):
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
    salida = cv2.VideoWriter('video_procesado_corregido.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (ancho, alto), isColor=False)

    # Crear una ventana para mostrar el video
    cv2.namedWindow('Líneas del carril', cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Procesar cada cuadro con corrección gamma o logarítmica
        procesado = procesar_cuadro(frame, gamma_valor=gamma_valor)

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
# Puedes probar con corrección gamma o logarítmica cambiando los parámetros
ruta_video = args["video"]
procesar_video(ruta_video, gamma_valor=0.07) 
