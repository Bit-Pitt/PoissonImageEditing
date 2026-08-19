
'''
    Script per creare una maschera binaria per l'immagine "source_resized.png".
'''

import cv2
import numpy as np

# Carica immagine
image = cv2.imread("source_resized.png")

# Mask binaria inizialmente tutta a 0
mask = np.zeros(image.shape[:2], dtype=np.uint8)

# Raggio del pennello
radius = 10

drawing = False


def mouse_callback(event, x, y, flags, param):
    global drawing, mask

    # Inizio a disegnare
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(mask, (x, y), radius, 1, -1)

    # Sto muovendo il mouse mentre tengo premuto
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(mask, (x, y), radius, 1, -1)

    # Smetto di disegnare
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


# Crea finestra
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)

while True:

    # Copia dell'immagine per visualizzare la zona selezionata
    display = image.copy()

    # Evidenzia la mask in rosso
    display[mask == 1] = (0, 0, 255)

    cv2.imshow("Image", display)

    key = cv2.waitKey(1) & 0xFF

    # ESC per uscire
    if key == 27:
        break

    # S per salvare
    elif key == ord("s"):
        cv2.imwrite("mask.png", mask * 255)
        print("Mask salvata!")

cv2.destroyAllWindows()