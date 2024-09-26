import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="Path to the input video")
args = vars(ap.parse_args())

# Inicializar histogramas globales acumulados para canales H, S y V
histograma_h_inicial = np.zeros((180,), dtype=np.float32) 
histograma_s_inicial = np.zeros((256,), dtype=np.float32)
histograma_v_inicial = np.zeros((256,), dtype=np.float32)
histograma_s_ecualizado = np.zeros((256,), dtype=np.float32)
histograma_v_ecualizado = np.zeros((256,), dtype=np.float32)
histograma_s_autoajustado = np.zeros((256,), dtype=np.float32)
histograma_v_autoajustado = np.zeros((256,), dtype=np.float32)
histograma_v_correjido = np.zeros((256,), dtype=np.float32)
histograma_h_final = np.zeros((180,), dtype=np.float32)
histograma_s_final = np.zeros((256,), dtype=np.float32)
histograma_v_final = np.zeros((256,), dtype=np.float32)

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
        channel_rescaled[channel <= a_prim_low] = a_min
        channel_rescaled[channel >= a_prim_high] = a_max
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

# Corrección gamma
def correccion_gamma(channel, gamma=1.0):
    gamma = 1.0/gamma
    
    mapeo = np.array([(255 * (i / 255.0) ** gamma) for i in np.arange(0, 256)]).astype("uint8")
    
    channel_gamma = mapeo[channel]
    
    return channel_gamma

#Normalizacion con base a los valores de los canales
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

# Detección de líneas blancas y amarillas con correcciones gamma/logarítmica
def procesar_cuadro(cuadro, gamma_valor=1.0):
    global histograma_h_inicial, histograma_s_inicial, histograma_v_inicial
    global histograma_s_ecualizado, histograma_v_ecualizado
    global histograma_s_autoajustado, histograma_v_autoajustado
    global histograma_v_correjido
    global histograma_h_final, histograma_s_final, histograma_v_final
    # Convertir todo el cuadro al espacio de color HSV
    hsv = cv2.cvtColor(cuadro, cv2.COLOR_BGR2HSV)

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
    
    hist_s_ecualizado = cv2.calcHist([s_ecualizado], [0], None, [256], [0, 256])
    hist_v_ecualizado = cv2.calcHist([v_ecualizado], [0], None, [256], [0, 256])
    histograma_s_ecualizado += hist_s_ecualizado.flatten()
    histograma_v_ecualizado += hist_v_ecualizado.flatten()

    # Luego aplicar el autocontraste restringido
    s_res = autocontraste_restringido(s_ecualizado)
    v_res = autocontraste_restringido(v_ecualizado)
    
    hist_s_res = cv2.calcHist([s_res], [0], None, [256], [0, 256])
    hist_v_res = cv2.calcHist([v_res], [0], None, [256], [0, 256])
    
    histograma_s_autoajustado += hist_s_res.flatten()
    histograma_v_autoajustado += hist_v_res.flatten()
    
    # Normalización post-procesamiento para garantizar que los valores estén en el rango adecuado
    s_normalizado = normalizacion_histograma(s_res, 255.0)
    v_normalizado = normalizacion_histograma(v_res, 255.0)
        
    #Aplicar corrección gamma o logarítmica en el canal V
    v_corregido = correccion_gamma(v_normalizado, gamma=gamma_valor)
    
   
    # Después del procesamiento, recalculamos los histogramas
    hist_h_final = cv2.calcHist([h], [0], None, [180], [0, 180])
    hist_s_final = cv2.calcHist([s_normalizado], [0], None, [256], [0, 256])
    hist_v_final = cv2.calcHist([v_corregido], [0], None, [256], [0, 256])
    histograma_h_final += hist_h_final.flatten()
    histograma_s_final += hist_s_final.flatten()
    histograma_v_final += hist_v_final.flatten()
    
    # Reconstruir la imagen HSV con el canal V ajustado
    hsv_ajustado = cv2.merge([h, s_normalizado, v_corregido])
    
    # Filtrar áreas brillantes y baja saturación para detectar líneas blancas
    mascara_blanca = cv2.inRange(hsv_ajustado, (0, 0, 200), (180, 60, 255))

    # Detectar la línea amarilla en el espacio HSV
    mascara_amarilla = cv2.inRange(hsv_ajustado, (15, 100, 100), (35, 255, 255))


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
    
# Al final del procesamiento de todo el video, mostramos los histogramas globales
def mostrar_histogramas_globales():
    plt.figure(figsize=(12, 6))
    
    # Histograma global inicial para canal H
    plt.subplot(3, 4, 4)
    plt.plot(histograma_h_inicial, color='red')
    plt.title('Histograma Global H Inicial')
    plt.xlabel('Intensidad de píxel')
    plt.ylabel('Frecuencia')
    
    # Histograma global inicial para canal S
    plt.subplot(3, 4, 1)
    plt.plot(histograma_s_inicial, color='blue')
    plt.title('Histograma Global S Inicial')
    plt.xlabel('Intensidad de píxel')
    plt.ylabel('Frecuencia')
    
    # Histograma global inicial para canal V
    plt.subplot(3, 4, 6)
    plt.plot(histograma_v_inicial, color='green')
    plt.title('Histograma Global V Inicial')
    plt.xlabel('Intensidad de píxel')
    plt.ylabel('Frecuencia')
    
    # Histograma ecualizado s
    plt.subplot(3, 4, 5)
    plt.plot(histograma_s_ecualizado, color='blue')
    plt.title('Histograma ecualizado s')
    plt.xlabel('Intensidad de píxel')
    plt.ylabel('Frecuencia')
    
    # Histograma ecualizado v
    plt.subplot(3, 4, 10)
    plt.plot(histograma_v_ecualizado, color='green')
    plt.title('Histograma ecualizado v')
    plt.xlabel('Intensidad de píxel')
    plt.ylabel('Frecuencia')
    
    # Histograma autoajustado para canal s
    plt.subplot(3, 4, 9)
    plt.plot(histograma_s_autoajustado, color='blue')
    plt.title('Histograma autoajustado s')
    plt.xlabel('Intensidad de píxel')
    plt.ylabel('Frecuencia')
    
    # Histograma autoajustado para canal v
    plt.subplot(3, 4, 3)
    plt.plot(histograma_v_autoajustado, color='green')
    plt.title('Histograma autoajustado v')
    plt.xlabel('Intensidad de píxel')
    plt.ylabel('Frecuencia')

    # Histograma global final para canal H
    plt.subplot(3, 4, 8)
    plt.plot(histograma_h_final, color='red')
    plt.title('Histograma Global H Final')
    plt.xlabel('Intensidad de píxel')
    plt.ylabel('Frecuencia')

    # Histograma global final para canal S
    plt.subplot(3, 4, 2)
    plt.plot(histograma_s_final, color='blue')
    plt.title('Histograma s normalizado')
    plt.xlabel('Intensidad de píxel')
    plt.ylabel('Frecuencia')
    
    # Histograma global final para canal V
    plt.subplot(3, 4, 7)
    plt.plot(histograma_v_final, color='green')
    plt.title('Histograma v corregido con gamma')
    plt.xlabel('Intensidad de píxel')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.show()

# Llamar a la función para procesar el video
# Puedes probar con corrección gamma o logarítmica cambiando los parámetros
ruta_video = args["video"]
procesar_video(ruta_video, gamma_valor=0.1) 

mostrar_histogramas_globales()