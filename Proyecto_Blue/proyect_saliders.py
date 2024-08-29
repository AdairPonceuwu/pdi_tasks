import cv2 as cv
import numpy as np

def nothing(x):
    pass

# Crear una ventana para los sliders
cv.namedWindow('Trackbars')

# Crear sliders para los valores de H, S y V
cv.createTrackbar('H Lower', 'Trackbars', 0, 179, nothing)
cv.createTrackbar('H Upper', 'Trackbars', 179, 179, nothing)
cv.createTrackbar('S Lower', 'Trackbars', 0, 255, nothing)
cv.createTrackbar('S Upper', 'Trackbars', 255, 255, nothing)
cv.createTrackbar('V Lower', 'Trackbars', 0, 255, nothing)
cv.createTrackbar('V Upper', 'Trackbars', 255, 255, nothing)

cap = cv.VideoCapture(0)

while True:

    # Capturar cada frame
    _, frame = cap.read()

    # Convertir de BGR a HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Obtener los valores de H, S, V desde los sliders
    h_lower = cv.getTrackbarPos('H Lower', 'Trackbars')
    h_upper = cv.getTrackbarPos('H Upper', 'Trackbars')
    s_lower = cv.getTrackbarPos('S Lower', 'Trackbars')
    s_upper = cv.getTrackbarPos('S Upper', 'Trackbars')
    v_lower = cv.getTrackbarPos('V Lower', 'Trackbars')
    v_upper = cv.getTrackbarPos('V Upper', 'Trackbars')

    # Definir los rangos inferior y superior
    lower_bound = np.array([h_lower, s_lower, v_lower])
    upper_bound = np.array([h_upper, s_upper, v_upper])

    # Crear una máscara basada en los valores de H, S, V
    mask = cv.inRange(hsv, lower_bound, upper_bound)

    # Realizar la operación AND bitwise entre la máscara y la imagen original
    res = cv.bitwise_and(frame, frame, mask=mask)

    # Mostrar los resultados
    cv.imshow('Frame', frame)
    cv.imshow('Mask', mask)
    cv.imshow('Result', res)

    # Salir si se presiona la tecla ESC
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

# Liberar los recursos y cerrar las ventanas
cap.release()
cv.destroyAllWindows()
