import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

def filtros(image):
    
    #Suavizado
    blur = cv2.medianBlur(image, 3, 0)
    
    #Separamos los canales
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    #Autoajuste
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(7, 7))
    v_clahe = clahe.apply(v)

    hsv_clahe = cv2.merge((h, s, v_clahe))
    enhanced = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)
    
    # Aplicar corrección gamma
    invGamma = 1.0 / 0.80
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(enhanced, table)
       
    return gamma_corrected

# Funcion para plotear las caras encontradas
def plot_all_faces(faces_list):
    num_faces = len(faces_list)
    cols = 5  #
    rows = (num_faces // cols) + (num_faces % cols > 0)
    
    plt.figure(figsize=(15, rows * 3))  
    for i, face in enumerate(faces_list):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    plt.tight_layout()
    plt.show()

#Argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required=True, help="Ruta al clasificador de caras en cascada")
ap.add_argument("-v", "--video", required=True, help="Ruta al archivo de video")
args = vars(ap.parse_args())

# cargar el detector
face_cascade = cv2.CascadeClassifier(args["face"])

# Abre el video
cap = cv2.VideoCapture(args["video"])

if not cap.isOpened():
    print("Error al abrir el archivo de video.")
    exit()

frame_count = 0  
face_image_count = 0  
all_faces = [] 

scale_percent = 50  

while True:
    
    ret, frame = cap.read()

    
    if not ret:
        break

    frame_count += 1  

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta las caras en el frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Procesa cada cara detectada
    for (x, y, w, h) in faces:
        if frame_count % 24 == 0:
            #Enfocamos la region de la cara
            face_region = frame[y:y+h, x:x+w]

            #Aplicamos los filtros al rostro
            filtered_face = filtros(face_region)

            #Guardamos el rostro encontrado
            all_faces.append(filtered_face)

            #Aumentamos el contador de imagenes
            face_image_count += 1

            # Mostramos el enfoque de las caras en el video
            frame[y:y+h, x:x+w] = filtered_face

    # Dibujamos un rectangulo en los rostros detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 152, 70), 2)

    # Redimensiona el frame antes de mostrarlo
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow('Deteccion de rostros', resized_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera los recursos
cap.release()
cv2.destroyAllWindows()

# Mostrar todas las caras detectadas al final
if all_faces:
    plot_all_faces(all_faces)
else:
    print("No se detectaron rostros.")
