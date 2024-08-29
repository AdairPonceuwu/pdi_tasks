import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

cap = cv.VideoCapture(0)

plt.ion()  # ventana interactiva
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
plt.subplots_adjust(bottom=0.35)  

# Leer el primer fotograma para inicializar las imágenes en los subgráficos
#Convertimos el frame a rgb para plt
ret, frame = cap.read()
frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

# Inicializar los subgráficos con la imagen original y la enmascarada
im1 = ax1.imshow(frame_rgb)
ax1.set_title('ORIGINAL')
im2 = ax2.imshow(frame_rgb)
ax2.set_title('MASK')

# Valores iniciales para los sliders
lower_h = 90
lower_s = 128
lower_v = 128
upper_h = 90
upper_s = 128
upper_v = 128

# Crear sliders para ajustar los valores de HSV
ax_lower_h = plt.axes([0.2, 0.25, 0.65, 0.03])
ax_lower_s = plt.axes([0.2, 0.2, 0.65, 0.03])
ax_lower_v = plt.axes([0.2, 0.15, 0.65, 0.03])
ax_upper_h = plt.axes([0.2, 0.1, 0.65, 0.03])
ax_upper_s = plt.axes([0.2, 0.05, 0.65, 0.03])
ax_upper_v = plt.axes([0.2, 0.0, 0.65, 0.03])

l_h = Slider(ax_lower_h, 'Lower H', 0.0, 179.0, valinit=lower_h)
l_s = Slider(ax_lower_s, 'Lower S', 0.0, 255.0, valinit=lower_s)
l_v = Slider(ax_lower_v, 'Lower V', 0.0, 255.0, valinit=lower_v)
u_h = Slider(ax_upper_h, 'Upper H', 0.0, 179.0, valinit=upper_h)
u_s = Slider(ax_upper_s, 'Upper S', 0.0, 255.0, valinit=upper_s)
u_v = Slider(ax_upper_v, 'Upper V', 0.0, 255.0, valinit=upper_v)

# Función para actualizar los valores de la máscara cuando se cambian los sliders
def update(val):
    global lower_h, lower_s, lower_v, upper_h, upper_s, upper_v
    lower_h = l_h.val
    lower_s = l_s.val
    lower_v = l_v.val
    upper_h = u_h.val
    upper_s = u_s.val
    upper_v = u_v.val

# Llamar a la función de actualización cuando se cambien los sliders
l_h.on_changed(update)
l_s.on_changed(update)
l_v.on_changed(update)
u_h.on_changed(update)
u_s.on_changed(update)
u_v.on_changed(update)

# Función para restablecer los sliders a sus valores iniciales
def resetSlider(event):
    l_h.reset()
    l_s.reset()
    l_v.reset()
    u_h.reset()
    u_s.reset()
    u_v.reset()

# Crear botón de restablecimiento
resetax = plt.axes([0.8, 0.9, 0.1, 0.04])
button = Button(resetax, 'Reset', color='gold', 
                hovercolor='skyblue')

# Llamar a la función resetSlider cuando se haga clic en el botón de restablecimiento
button.on_clicked(resetSlider)

# Bandera para indicar si se debe cerrar
should_close = False

def on_key(event):
    global should_close
    if event.key == 'e':
        should_close = True

fig.canvas.mpl_connect('key_press_event', on_key)

try:
    while not should_close:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo capturar el video.")
            break

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        #Creamos dos arreglos para crear la maskara hsv
        lower = np.array([lower_h, lower_s, lower_v], dtype=np.uint8)
        upper = np.array([upper_h, upper_s, upper_v], dtype=np.uint8)
        
        # Crear una máscara basada en los valores de H, S, V
        mask = cv.inRange(hsv, lower, upper)
        
        res = cv.bitwise_and(frame, frame, mask=mask)
        
        # Convertir de HSV a RGB para mostrar en plt
        frame_rgb = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
        
        #Actualizamos los dos recuadros 
        im1.set_data(frame_rgb)
        im2.set_data(res)

        #Moestramos los nuevos fotogramas
        plt.draw()
        plt.pause(0.2) 

finally:
    # Liberar los recursos al salir
    cap.release()
    plt.close('all')
