import cv2 as cv
import numpy as np
import argparse


parser = argparse.ArgumentParser()


parser.add_argument("index_camera", help="No encontrada", type=int)
args = parser.parse_args()

# We create a VideoCapture object to read from the camera (pass 0):
capture = cv.VideoCapture(args.index_camera)


if capture.isOpened() is False:
    print("Error opening the camera")
    exit()

frame_width = capture.get(cv.CAP_PROP_FRAME_WIDTH)
frame_height = capture.get(cv.CAP_PROP_FRAME_WIDTH)
fps = capture.get(cv.CAP_PROP_FPS)

print(frame_width)
print(frame_height)
print("Velocidad de FPS {}".format(fps))

low_color_h = []
low_color_s = []
low_color_v = []
high_color_h = []
high_color_s = []
high_color_v = []

while capture.isOpened():

    # Capture frame by frame from the camera
    ret, frame = capture.read()

    if ret is True:

        frame = cv.resize(frame, (0, 0), fx=0.4, fy=0.4)

        # Convert this frame to HSV
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Min and Max values are computed
        h, s, v = cv.split(hsv_frame)

        mean_h = np.mean(h)
        mean_s = np.mean(s)
        mean_v = np.mean(v)

        n = h.size

        std_h = np.sqrt(np.sum((h - mean_h) ** 2) / n)
        std_s = np.sqrt(np.sum((s - mean_s) ** 2) / n)
        std_v = np.sqrt(np.sum((v - mean_v) ** 2) / n)


        low_color_h.append(mean_h - std_h)
        low_color_s.append(mean_s - std_s)
        low_color_v.append(mean_v - std_v)
        high_color_h.append(mean_h + std_h)
        high_color_s.append(mean_s + std_s)
        high_color_v.append(mean_v + std_v)

        # Display the hsv frame
        cv.imshow("Reading HSV values...", hsv_frame)

        # Press q on keyboard to exit the program
        if cv.waitKey(20) & 0xFF == ord('q'):
            break

    else:
        break

# Destruir ventanas y mostrar valores
cv.destroyAllWindows()
low_values = np.array([np.min(low_color_h), np.min(low_color_s), np.min(low_color_v)])
high_values =np.array([np.max(high_color_h), np.max(high_color_s), np.max(high_color_v)])

low_values_h = np.array((np.min(low_color_h)))
high_values_h = np.array((np.max(high_color_h)))



# Imprimir Valores
print(low_values)
print(high_values)




while capture.isOpened():

    ret, frame = capture.read()

    if ret is True:

        frame = cv.resize(frame, (0,0), fx = 0.4, fy = 0.4)


        cv.imshow('input frame from the camera', frame)

        # Convert the HSV images
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv_frame)

        #Threshold the HSV images
        mask = cv.inRange(hsv_frame, low_values, high_values)
        res = cv.bitwise_and(hsv_frame, hsv_frame, mask = mask)

        #Threshold the H Channel
        mask_h = cv.inRange(hsv_frame, low_values_h, high_values_h)
        res_h = cv.bitwise_and(h, h, mask = mask_h)

        # Display frames:
        cv.imshow('Mask input camera', mask)
        cv.imshow('Segmented HSV', res)
        cv.imshow('Mask H channel', mask_h)

        if cv.waitKey(20) & 0xFF == ord('q'):
            break

    else:
        break


cv.destroyAllWindows()