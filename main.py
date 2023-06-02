from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import cv2
import numpy as np

#Agregar en la terminal
# pip install python-multipart
# pip install uvicorn

#------------------------- Con este corre el api--------------------------
#uvicorn main:app --reload

#-------------------- abrir postman---------------------
#generar un post
#http://127.0.0.1:8000/upload-image/
#ir a body -> form-data
# key: image, value(cambiar a file): direccion de la imagen
app = FastAPI()

def process_image(index_to_crop, file):
    # Cargar imagen en escala de grises
    #cv2.imwrite('real_image.jpg')
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    #
    cv2.imwrite('real_image.jpg', image)

    image = cv2.bitwise_not(image)
    # Crear detector SimpleBlobDetector con parámetros personalizados
    params = cv2.SimpleBlobDetector_Params()

    # Cambiar umbrales
    params.minThreshold = 1
    params.maxThreshold = 2000

    # Filtrar por Área.
    params.filterByArea = True
    params.minArea = 10
    params.maxArea = 100

    # Filtrar por Circularidad
    params.filterByCircularity = False
    params.minCircularity = 0.001
    params.maxCircularity = 100000

    # Filtrar por Convexidad
    params.filterByConvexity = True
    params.minConvexity = 0.9

    # Filtrar por Inercia
    params.filterByInertia = True
    params.minInertiaRatio = 0.6

    detector = cv2.SimpleBlobDetector_create(params)

    # Detectar keypoints en la imagen
    keypoints = detector.detect(image)

    # Dibujar keypoints y etiquetas en una copia de la imagen
    image_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0),
                                        flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

    for i, keypoint in enumerate(keypoints):
        x, y = keypoint.pt
        print(str(i)+" "+str(x))
        label = f'kp{i}'
        cv2.putText(image_keypoints, label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,
                    cv2.LINE_AA)

    # Guardar la imagen con keypoints y etiquetas en un archivo temporal
    temp_file = 'temp_image.jpg'
    cv2.imwrite(temp_file, image_keypoints)

    print(keypoint)
    # Definir el tamaño del área a recortar alrededor del keypoint
    crop_size = 30  # Ajusta este valor según el tamaño deseado

    keypoint_to_crop = keypoints[index_to_crop]

    # Calcular las coordenadas del área a recortar
    x, y = keypoint_to_crop.pt
    print("valor x:" + str(x))
    print("valor y:" + str(y))
    x, y = int(x), int(y)

    x1, y1 = max(0, x - crop_size // 2), max(0, y - crop_size // 2)
    x2, y2 = min(image.shape[1], x + crop_size // 2), min(image.shape[0], y + crop_size // 2)

    # Recortar el área alrededor del keypoint
    cropped_image = image[y1:y2, x1:x2]
    im = cv2.bitwise_not(cropped_image)

    result = 'result_image.jpg'
    cv2.imwrite(result, im)

    return temp_file

def procces_key_point():

    img1 = cv2.imread('result_image.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('real_image.jpg', cv2.IMREAD_GRAYSCALE)

    # Crear el detector y descriptor ORB
    sift = cv2.SIFT_create()  # Reemplazar por cv2.xfeatures2d.SURF_create() para SURF

    # Detectar y describir características en ambas imágenes
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # Crear y configurar el emparejador de características (matcher)
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Emparejar características
    matches = matcher.match(descriptors1, descriptors2)

    # Ordenar emparejamientos por distancia (cuanto menor es la distancia, más similares son las características)
    best_matches = sorted(matches, key=lambda x: x.distance)
    print(len(best_matches))
    # Dibujar los primeros 30 mejores emparejamientos en una imagen
    result = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:1], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Dibujar círculos rojos en la imagen img2 para los 30 mejores matches
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    for match in best_matches[:1]:
        # Obtener el keypoint correspondiente al match en la img2
        keypoint = keypoints2[match.trainIdx]

        # Dibujar un círculo en la posición del keypoint
        x, y = keypoint.pt
        print("valor x:" + str(x))
        print("valor y:" + str(y))
        cv2.circle(img2_color, (int(x), int(y)), 5, (0, 0, 255), 2)

    # Mostrar imagen con emparejamientos
    output2 = 'punto.jpg'
    cv2.imwrite(output2, img2_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return output2


@app.post('/upload-image/')
async def upload_image(keypoint: str, image: UploadFile = File(...)):
    index_to_crop = int(keypoint[2:])
    temp_file = process_image(index_to_crop, image.file)

    return StreamingResponse(open(temp_file, 'rb'), media_type='image/jpeg')
#
@app.post('/keypoint/')
async def key_point():
     #Procesar la imagen con keypoints seleccionados

     key = procces_key_point()

     return StreamingResponse(open(key, 'rb'), media_type='image/jpeg')