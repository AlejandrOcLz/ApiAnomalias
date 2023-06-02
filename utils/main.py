import cv2

def main():
    # Cargar imagen en escala de grises
    image = cv2.imread('img/4.jpg', cv2.IMREAD_GRAYSCALE)
    image = cv2.bitwise_not(image)
    # Crear detector SimpleBlobDetector con parámetros personalizados
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 1;
    params.maxThreshold = 2000;

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 10
    params.maxArea = 100

    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.001
    params.maxCircularity = 100000

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.9

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.6

    detector = cv2.SimpleBlobDetector_create(params)

    # Detectar keypoints en la imagen
    keypoints = detector.detect(image)

    # Dibujar keypoints y etiquetas en una copia de la imagen
    image_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

    for i, keypoint in enumerate(keypoints):
        x, y = keypoint.pt
        label = f'kp{i}'
        cv2.putText(image_keypoints, label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    # Mostrar imagen con keypoints y etiquetas
    cv2.imshow('Keypoints con etiquetas', image_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Pedir al usuario el label de la imagen a recortar
    label_to_crop = input('Introduce el label del keypoint a recortar (ejemplo: kp0, kp1, etc.): ')
    print("cortado "+label_to_crop)

    # Encontrar el keypoint correspondiente al label ingresado
    index_to_crop = int(label_to_crop[2:])  # Extrae el número del label (asumiendo el formato 'kp0', 'kp1', etc.)
    keypoint_to_crop = keypoints[index_to_crop]

    # Definir el tamaño del área a recortar alrededor del keypoint
    crop_size = 30  # Ajusta este valor según el tamaño deseado

    # Calcular las coordenadas del área a recortar
    x, y = keypoint_to_crop.pt
    x, y = int(x), int(y)
    x1, y1 = max(0, x - crop_size // 2), max(0, y - crop_size // 2)
    x2, y2 = min(image.shape[1], x + crop_size // 2), min(image.shape[0], y + crop_size // 2)

    # Recortar el área alrededor del keypoint
    cropped_image = image[y1:y2, x1:x2]
    im = cv2.bitwise_not(cropped_image)

    # Guardar la imagen recortada
    cv2.imwrite(f'keypoint_{label_to_crop}_cropped.jpg', im)

if __name__ == '__main__':
    main()