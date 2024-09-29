import cv2
import numpy as np
import argparse

#Solicitamos la ubicacion del video
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="Path to the input video")
args = vars(ap.parse_args())

# Funcion para calcular el CDF(histograma acumulado)
def cdf(hist):
    cdf = hist.cumsum()  
    return cdf

# Funcion para calcular CDF normalizado
def cdf_normalizado(hist, cdf_hist):
    # Normalizamos el CDF acumulado con base a su histograma
    cdf_normalizado = cdf_hist * float(hist.max()) / cdf_hist.max() 
    return cdf_normalizado
    
# Funcion para calcular el autocontraste_restringido
def autocontraste_restringido(channel, q=0.05):
    #Calcular el histograma
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    
    #Calcular el histograma acumulado
    hist_acumulado = cdf(hist)
    
    # Calcular CDF normalizado
    cdf_norma = cdf_normalizado(hist, hist_acumulado)
    
    # Calcular low_percentile y high_percentile basados en q
    low_perc = q * 100  
    high_perc = (1 - q) * 100 
    
    # Encontrar los valores en el CDF correspondientes a los percentiles deseados
    a_prim_low = np.searchsorted(cdf_norma, low_perc / 100.0)  
    a_prim_high = np.searchsorted(cdf_norma, high_perc / 100.0)  
    
    # Minimos y máximos
    a_min = 0
    a_max = 255
    
    # Verificar si a_prim_high es igual a a_prim_low para evitar división por cero
    if a_prim_high == a_prim_low:
        # Si ambos son iguales, devolvemos el canal sin cambios
        channel_rescaled = channel.copy()
    else:
        # Autocontraste de los valores
        channel_rescaled = np.clip((channel - a_prim_low) * a_max / (a_prim_high - a_prim_low), a_min, a_max).astype(np.uint8)
    #Regresamos el canal ajustado
    return channel_rescaled

# Ecualización de histograma
def ecualizacion_histograma(channel):
    #Calculamos el histograma
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    
    #Calcular el histograma acumulado
    hist_acumulado = cdf(hist)

    #Ocultar todos los elementos del CDF que sean iguales a 0.
    ecua_hi = np.ma.masked_equal(hist_acumulado, 0)

    #Ecuacion para ecualizar el histograma
    ecua_hi = ((ecua_hi - ecua_hi.min()) * 255) / hist_acumulado[-1]

    #Remplazar todos los valores enmascarados anteriormente con un valor de 0
    ecua_hi_mask = np.ma.filled(ecua_hi,0).astype('uint8')

    #Remapeo del canal
    channel_eq = ecua_hi_mask[channel]
    
    #Regresamos el canal ecualizado
    return channel_eq

# Normalizacion con base a los valores de los canales
def normalizacion_histograma(channel, channel_max):
    channel = channel.astype(np.float32)
    
    #Encontramos el valor maximo y minimo del histograma
    min_val = np.min(channel)
    max_val = np.max(channel)
    
    #Verificamos que los valores no sean los mismos
    if max_val - min_val != 0: 
        #Normalizamos el canal
        normalized_channel = (channel - min_val) * (channel_max / (max_val - min_val))
    else:
        #Regresamos el canal sin modificacion
        normalized_channel = channel.copy() 

    #Lo convertimos al tipo de dato correspondiente
    normalized_channel = normalized_channel.astype(np.uint8)
    
    #Regresamos el canal normalizado
    return normalized_channel

# Corrección gamma
def correccion_gamma(channel, gamma=1.0):
    #Definimos el valor de gamma
    gamma = 1.0/gamma
    
    #Mapeamos gamma
    mapeo = np.array([(255 * (i / 255.0) ** gamma) for i in np.arange(0, 256)]).astype("uint8")
    
    #Mapea el canal con los valores gamma
    channel_gamma = mapeo[channel]
    
    #Regresamos el canal con la correcion aplicada
    return channel_gamma

def filtro_logaritmico(h_channel):
    # Convertimos el canal con valores flotantes
    h_float = h_channel.astype(np.float32) + 1.0  # Se añade 1.0 para evitar log(0)
    
    # Calculamos el logaritmo
    h_log = np.log(h_float)
    
    # Normalizamos el resultado logarítmico a un rango de 0 a 180
    h_aclarado = (h_log / np.log(h_float.max())) * 180
    
    # Convertimos el resultado de nuevo a uint8 para su uso en imágenes
    return h_aclarado.astype(np.uint8)

def procesar_frame(cuadro, gamma_valor=1.0):
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

    # Normalización para garantizar que los valores estén en el rango adecuado
    s_normalizado = normalizacion_histograma(s_res, 255.0)
    v_normalizado = normalizacion_histograma(v_res, 255.0)
    #h_normalizado = normalizacion_histograma(h, 180.0)
        
    #Aplicar corrección gamma y logarítmica 
    h_corregido = filtro_logaritmico(h)
    v_corregido = correccion_gamma(v_normalizado, gamma=gamma_valor)
    
    # Reconstruir la imagen HSV con el canal V ajustado
    hsv_ajustado = cv2.merge([h_corregido, s_normalizado, v_corregido])
    
    # Filtrar áreas brillantes y baja saturación para detectar líneas blancas
    mascara_blanca = cv2.inRange(hsv_ajustado, (0, 0, 200), (180, 60, 255))

    # Detectar la línea amarilla en el espacio HSV
    mascara_amarilla = cv2.inRange(hsv_ajustado, (5, 200, 200), (120, 255, 255))

    # Combinar ambas máscaras (blanca y amarilla)
    mascara_combined = cv2.bitwise_or(mascara_blanca, mascara_amarilla)
    
    # Obtener el tamaño del frame
    alto, ancho = cuadro.shape[:2]

    # Crear una máscara negra que cubra mas de la mitad superior
    mascara_combined[:int(alto * 0.6), :] = 0  

    # Eliminar un 20% del lado izquierdo
    mascara_combined[:, :int(ancho * 0.2)] = 0  

    #Regresamos el frame procesado
    return mascara_combined


# Procesamiento de video
def procesar_video(ruta_video, gamma_valor=1.0):
    
    # Leer el video
    cap = cv2.VideoCapture(ruta_video)

    #Mandamos un mensaje en caso de no poder abrir el archivo
    if not cap.isOpened():
        print("Error al abrir el archivo de video.")
        return

    # Obtener propiedades del video
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Guardar el video ya procesado
    salida = cv2.VideoWriter('lineas_procesadas.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (ancho, alto), isColor=False)
    
    # Crear una ventana para mostrar el video
    cv2.namedWindow('Líneas del carril', cv2.WINDOW_NORMAL)

    #Comenzamos con la captura de cada frame y su procesado
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Procesar cada frame
        procesado = procesar_frame(frame, gamma_valor=gamma_valor)

        # Mostramos solo el video procesado
        cv2.imshow('Líneas del carril', procesado)

        # Guardamos el frame para al final tener el video procesado
        salida.write(procesado)
        
        # Esperamos y definimos una tecla de salida
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Finalizamos el proceso
    cap.release()
    salida.release()
    cv2.destroyAllWindows()

# Llamar a la función para procesar el video
ruta_video = args["video"]
procesar_video(ruta_video, gamma_valor=0.07) 
