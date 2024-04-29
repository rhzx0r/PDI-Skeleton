import numpy as np
import cv2
import matplotlib.pyplot as plt

def skeletonize_manual(image):
    # Inicializar matrices para almacenar resultados intermedios
    skeleton = np.zeros(image.shape, np.uint8)  # Skeleton resultante
    eroded = np.zeros(image.shape, np.uint8)    # Imagen erosionada
    temp = np.zeros(image.shape, np.uint8)      # Almacenamiento temporal

    # Iterar hasta que la imagen se vuelva completamente negra
    while True:
        # Erosión de la imagen original
        cv2.erode(image, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), eroded)
        # Dilatación de la imagen erosionada
        cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), temp)
        # Calcular la diferencia entre la imagen original y la imagen dilatada
        cv2.subtract(image, temp, temp)
        # Unir la diferencia con el esqueleto actual
        cv2.bitwise_or(skeleton, temp, skeleton)
        # Actualizar la imagen original con la imagen erosionada para la siguiente iteración
        image[:,:] = eroded[:,:]
        # Verificar si la imagen original se ha vuelto completamente negra
        if cv2.countNonZero(image) == 0:
            break

    # Devolver el resultado final del esqueleto
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
