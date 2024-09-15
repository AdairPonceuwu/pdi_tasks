import argparse
import cv2

from gray_ecu import gray
from hsv_ecu import hsv
from contraste_restringido import contraste_res

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image")
ap.add_argument("-ql", "--qlow", required=False, help="Value for q_low")
ap.add_argument("-qh", "--qhigh", required=False, help="Value for q_high")
args = vars(ap.parse_args())

image = cv2.imread(args["image"], cv2.IMREAD_UNCHANGED)

gray_flag = True

if len(image.shape) == 2:
    img = cv2.merge([image,image,image])
else:
    gray_flag = False
    img_BGR = image
    image_HSV = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HSV)

if (args["qlow"] and args["qhigh"]) == None:
    if gray_flag:
        gray(img)
    else:
        hsv(img_BGR, image_HSV)
elif gray_flag:
    # Parámetros para el recorte de contraste
    q_low, q_high = float(args["qlow"]), float(args["qhigh"])
    contraste_res(img, q_low, q_high)



