import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

# Solicitar la ubicación del video
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="Path to the input video")
args = vars(ap.parse_args())

# Inicializar histogramas globales acumulados para canales H, S y V
# Histogramas iniciales
histograma_h_inicial = np.zeros((180,), dtype=np.float32) 
histograma_s_inicial = np.zeros((256,), dtype=np.float32)
histograma_v_inicial = np.zeros((256,), dtype=np.float32)
# Histogramas procesados
histograma_s_ecualizado = np.zeros((256,), dtype=np.float32)
histograma_v_ecualizado = np.zeros((256,), dtype=np.float32)
histograma_s_autoajustado = np.zeros((256,), dtype=np.float32)
histograma_v_autoajustado = np.zeros((256,), dtype=np.float32)
histograma_v_normalizado = np.zeros((256,), dtype=np.float32)
# Histogramas finales
histograma_h_final = np.zeros((180,), dtype=np.float32)
histograma_s_final = np.zeros((256,), dtype=np.float32)
histograma_v_final = np.zeros((256,), dtype=np.float32)

# Función para calcular el CDF (histograma acumulado)
def cdf(hist):
    cdf = hist.cumsum()  
    return cdf

# Función para calcular CDF normalizado
def cdf_normalizado(hist, cdf_hist):
    # Normalizamos el CDF acumulado con base a su histograma
    cdf_normalizado = cdf_hist * float(hist.max()) / cdf_hist.max() 
    return cdf_normalizado
    
# Función para calcular el autocontraste restringido
def autocontraste_restringido(channel, q=0.05):
    # Calcular el histograma
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    
    # Calcular el histograma acumulado
    hist_acumulado = cdf(hist)
    
    # Calcular CDF normalizado
    cdf_norma = cdf_normalizado(hist, hist_acumulado)
    
    # Calcular low_percentile y high_percentile basados en q
    low_perc = q * 100  
    high_perc = (1 - q) * 100 
    
    # Encontrar los valores en el CDF correspondientes a los percentiles deseados
    a_prim_low = np.searchsorted(cdf_norma, low_perc / 100.0)  
    a_prim_high = np.searchsorted(cdf_norma, high_perc / 100.0)  
    
    # Mínimos y máximos
    a_min = 0
    a_max = 255
    
    # Verificar si a_prim_high es igual a a_prim_low para evitar división por cero
    if a_prim_high == a_prim_low:
        # Si ambos son iguales, devolvemos el canal sin cambios
        channel_rescaled = channel.copy()
    else:
        # Autocontraste de los valores
        channel_rescaled = np.clip((channel - a_prim_low) * a_max / (a_prim_high - a_prim_low), a_min, a_max).astype(np.uint8)
    # Regresamos el canal ajustado
    return channel_rescaled

# Ecualización de histograma
def ecualizacion_histograma(channel):
    # Calculamos el histograma
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    
    # Calcular el histograma acumulado
    hist_acumulado = cdf(hist)

    # Ocultar todos los elementos del CDF que sean iguales a 0.
    ecua_hi = np.ma.masked_equal(hist_acumulado, 0)

    # Ecuación para ecualizar el histograma
    ecua_hi = ((ecua_hi - ecua_hi.min()) * 255) / hist_acumulado[-1]

    # Remplazar todos los valores enmascarados anteriormente con un valor de 0
    ecua_hi_mask = np.ma.filled(ecua_hi,0).astype('uint8')

    # Remapeo del canal
    channel_eq = ecua_hi_mask[channel]
    
    # Regresamos el canal ecualizado
    return channel_eq

# Normalización con base a los valores de los canales
def normalizacion_histograma(channel, channel_max):
    channel = channel.astype(np.float32)
    
    # Encontramos el valor máximo y mínimo del histograma
    min_val = np.min(channel)
    max_val = np.max(channel)
    
    # Verificamos que los valores no sean los mismos
    if max_val - min_val != 0: 
        # Normalizamos el canal
        normalized_channel = (channel - min_val) * (channel_max / (max_val - min_val))
    else:
        # Regresamos el canal sin modificación
        normalized_channel = channel.copy() 

    # Lo convertimos al tipo de dato correspondiente
    normalized_channel = normalized_channel.astype(np.uint8)
    
    # Regresamos el canal normalizado
    return normalized_channel

# Corrección gamma
def correccion_gamma(channel, gamma=1.0):
    # Definimos el valor de gamma
    gamma = 1.0/gamma
    
    # Mapeamos gamma
    mapeo = np.array([(255 * (i / 255.0) ** gamma) for i in np.arange(0, 256)]).astype("uint8")
    
    # Mapeamos el canal con los valores gamma
    channel_gamma = mapeo[channel]
    
    # Regresamos el canal con la corrección aplicada
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

def procesar_frame(frame, gamma_valor=1.0):
    global histograma_h_inicial, histograma_s_inicial, histograma_v_inicial
    global histograma_s_ecualizado, histograma_v_ecualizado
    global histograma_s_autoajustado, histograma_v_autoajustado
    global histograma_v_normalizado
    global histograma_h_final, histograma_s_final, histograma_v_final
    
    # Convertir todo el cuadro al espacio de color HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Separar los canales H, S, V
    h, s, v = cv2.split(hsv)

    # Calcular histogramas iniciales y acumular
    hist_h = cv2.calcHist([h], [0], None, [180], [0, 180])
    hist_s = cv2.calcHist([s], [0], None, [256], [0, 256])
    hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])
    histograma_h_inicial += hist_h.flatten()
    histograma_s_inicial += hist_s.flatten()
    histograma_v_inicial += hist_v.flatten()
    
    # Aplicar ecualización de histograma primero
    s_ecualizado = ecualizacion_histograma(s)
    v_ecualizado = ecualizacion_histograma(v)
    
    # Calcular histogramas ecualizado y acumular
    hist_s_ecualizado = cv2.calcHist([s_ecualizado], [0], None, [256], [0, 256])
    hist_v_ecualizado = cv2.calcHist([v_ecualizado], [0], None, [256], [0, 256])
    histograma_s_ecualizado += hist_s_ecualizado.flatten()
    histograma_v_ecualizado += hist_v_ecualizado.flatten()

    # Luego aplicar el autocontraste restringido
    s_res = autocontraste_restringido(s_ecualizado)
    v_res = autocontraste_restringido(v_ecualizado)
    
    # Calcular histogramas autoajustado y acumular
    hist_s_res = cv2.calcHist([s_res], [0], None, [256], [0, 256])
    hist_v_res = cv2.calcHist([v_res], [0], None, [256], [0, 256])
    histograma_s_autoajustado += hist_s_res.flatten()
    histograma_v_autoajustado += hist_v_res.flatten()
    
    # Normalización para garantizar que los valores estén en el rango adecuado
    s_normalizado = normalizacion_histograma(s_res, 255.0)
    v_normalizado = normalizacion_histograma(v_res, 255.0)
    
    hist_v_normalizado = cv2.calcHist([v_normalizado], [0], None, [256], [0, 256])
    histograma_v_normalizado += hist_v_normalizado.flatten()
        
    #Aplicar corrección gamma y logarítmica 
    h_corregido = filtro_logaritmico(h)
    v_corregido = correccion_gamma(v_normalizado, gamma=gamma_valor)
    
    # Después del procesamiento, recalculamos los histogramas finales
    hist_h_final = cv2.calcHist([h_corregido], [0], None, [180], [0, 180])
    hist_s_final = cv2.calcHist([s_normalizado], [0], None, [256], [0, 256])
    hist_v_final = cv2.calcHist([v_corregido], [0], None, [256], [0, 256])
    histograma_h_final += hist_h_final.flatten()
    histograma_s_final += hist_s_final.flatten()
    histograma_v_final += hist_v_final.flatten()
    
    # Reconstruir la imagen HSV con los canales procesados
    hsv_ajustado = cv2.merge([h_corregido, s_normalizado, v_corregido])
    
    # Detectar líneas blancas en el espacio HSV
    mascara_blanca = cv2.inRange(hsv_ajustado, (0, 0, 200), (180, 60, 255))

    # Detectar la línea amarilla en el espacio HSV
    mascara_amarilla = cv2.inRange(hsv_ajustado, (5, 200, 200), (120, 255, 255))

    # Combinar ambas máscaras (blanca y amarilla)
    mascara_combined = cv2.bitwise_or(mascara_blanca, mascara_amarilla)
    
    # Obtener el tamaño del frame
    alto, ancho = frame.shape[:2]

    # Crear una máscara negra que cubra mas de la mitad superior
    mascara_combined[:int(alto * 0.6), :] = 0  

   # Establecer un 20% del lado  en negro
    mascara_combined[:, :int(ancho * 0.2)] = 0  

    # Regresamos el frame procesado
    return mascara_combined

# Procesamiento de video
def procesar_video(ruta_video, gamma_valor=1.0):
    cap = cv2.VideoCapture(ruta_video)

    # Mandamos un mensaje en caso de no poder abrir el archivo
    if not cap.isOpened():
        print("Error al abrir el archivo de video.")
        return

    # Obtener propiedades del video
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Guardar el video procesado
    #salida = cv2.VideoWriter('lineas_procesadas.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (ancho, alto), isColor=False)
    
    # Crear una ventana para mostrar el video
    cv2.namedWindow('Líneas del carril', cv2.WINDOW_NORMAL)

    paused = False  # Variable para controlar el estado de pausa
    
    #Comenzamos con la captura de cada frame y su procesado
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if not paused:
            # Procesar cada frame
            procesado = procesar_frame(frame, gamma_valor=gamma_valor)
            #salida.write(procesado)
            cv2.imshow('Líneas del carril', procesado)

        key = cv2.waitKey(1) & 0xFF

        # Pausar
        if key == ord('p'):
            paused = True
            # Mostrar histogramas 
            mostrar_histogramas_globales()

        # Reanudar
        elif key == ord('r'):
            paused = False

        # Salir
        elif key == ord('q'):
            break

    # Finalizamos el proceso
    cap.release()
    #salida.release()
    cv2.destroyAllWindows()

# Mostrar histogramas globales acumulados
def mostrar_histogramas_globales():
    plt.figure(figsize=(12, 6))
    
    # Histograma global inicial para canal h
    plt.subplot(3, 4, 1)
    plt.plot(histograma_h_inicial, color='red')
    plt.title('Histograma h Inicial')
    plt.xlabel('Intensidad de píxel')
    plt.ylabel('Frecuencia')
    
    # Histograma global inicial para canal s
    plt.subplot(3, 4, 2)
    plt.plot(histograma_s_inicial, color='blue')
    plt.title('Histograma s Inicial')
    plt.xlabel('Intensidad de píxel')
    plt.ylabel('Frecuencia')
    
    # Histograma global inicial para canal v
    plt.subplot(3, 4, 7)
    plt.plot(histograma_v_inicial, color='green')
    plt.title('Histograma v Inicial')
    plt.xlabel('Intensidad de píxel')
    plt.ylabel('Frecuencia')
    
    # Histograma ecualizado s
    plt.subplot(3, 4, 6)
    plt.plot(histograma_s_ecualizado, color='blue')
    plt.title('Histograma ecualizado s')
    plt.xlabel('Intensidad de píxel')
    plt.ylabel('Frecuencia')
    
    # Histograma ecualizado v
    plt.subplot(3, 4, 11)
    plt.plot(histograma_v_ecualizado, color='green')
    plt.title('Histograma ecualizado v')
    plt.xlabel('Intensidad de píxel')
    plt.ylabel('Frecuencia')
    
    # Histograma autoajustado para canal s
    plt.subplot(3, 4, 10)
    plt.plot(histograma_s_autoajustado, color='blue')
    plt.title('Histograma autoajustado s')
    plt.xlabel('Intensidad de píxel')
    plt.ylabel('Frecuencia')
    
    # Histograma autoajustado para canal v
    plt.subplot(3, 4, 4)
    plt.plot(histograma_v_autoajustado, color='green')
    plt.title('Histograma autoajustado v')
    plt.xlabel('Intensidad de píxel')
    plt.ylabel('Frecuencia')

    # Histograma global final para canal h (con filtro logaritmico)
    plt.subplot(3, 4, 5)
    plt.plot(histograma_h_final, color='red')
    plt.title('Histograma h filtro logaritmico')
    plt.xlabel('Intensidad de píxel')
    plt.ylabel('Frecuencia')

    # Histograma global final para canal s (normalizado)
    plt.subplot(3, 4, 3)
    plt.plot(histograma_s_final, color='blue')
    plt.title('Histograma s normalizado')
    plt.xlabel('Intensidad de píxel')
    plt.ylabel('Frecuencia')
    
    # Histograma global para canal v normalizado
    plt.subplot(3, 4, 8)
    plt.plot(histograma_v_normalizado, color='green')
    plt.title('Histograma v normalizado')
    plt.xlabel('Intensidad de píxel')
    plt.ylabel('Frecuencia')
    
    # Histograma global final para canal v (con corrección gamma)
    plt.subplot(3, 4, 12)
    plt.plot(histograma_v_final, color='green')
    plt.title('Histograma v corregido con gamma')
    plt.xlabel('Intensidad de píxel')
    plt.ylabel('Frecuencia')
    
    plt.tight_layout()
    plt.show()

# Llamar a la función para procesar el video
ruta_video = args["video"]
procesar_video(ruta_video, gamma_valor=0.07) 

#Cuando finalice, mostramos los histogramas
mostrar_histogramas_globales()