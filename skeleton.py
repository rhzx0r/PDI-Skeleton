import numpy as np
import cv2
import matplotlib.pyplot as plt

def skeletonize_manual(image):
    skeleton = np.zeros(image.shape, np.uint8)
    eroded = np.zeros(image.shape, np.uint8)
    temp = np.zeros(image.shape, np.uint8)

    while True:
        cv2.erode(image, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), eroded)
        cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), temp)
        cv2.subtract(image, temp, temp)
        cv2.bitwise_or(skeleton, temp, skeleton)
        image[:,:] = eroded[:,:]
        if cv2.countNonZero(image) == 0:
            break

    return skeleton

# Cargar una imagen en blanco y negro
image = cv2.imread('img/person.jpg', 0)

# Binarizar la imagen (convertir a blanco y negro)
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Aplicar la función skeletonize_manual
skeleton_manual = skeletonize_manual(binary_image)

# Mostrar la imagen original y el resultado del skeletonize manual
plt.subplot(1, 2, 1)
plt.imshow(binary_image, cmap='gray')
plt.title('Imagen Original')
plt.subplot(1, 2, 2)
plt.imshow(skeleton_manual, cmap='gray')
plt.title('Skeleton Manual')
plt.show()
