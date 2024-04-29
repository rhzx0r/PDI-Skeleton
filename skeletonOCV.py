import cv2

# Cargar una imagen en blanco y negro
image = cv2.imread('img/person.jpg', 0)

# Binarizar la imagen (convertir a blanco y negro)
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Aplicar la función de skeletonize proporcionada por OpenCV
skeleton_opencv = cv2.ximgproc.thinning(binary_image, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

# Mostrar la imagen original y el resultado del skeletonize con OpenCV
cv2.imshow('Original', binary_image)
cv2.imshow('Skeleton OpenCV', skeleton_opencv)
cv2.waitKey(0)
cv2.destroyAllWindows()
