import cv2
import numpy as np
import os
from ModeloCNN import predict_image

Classes = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

path = "./imagenes"
imagenes = os.listdir(path)
for imagen in imagenes:
    img = cv2.imread(f'./imagenes/{imagen}')
    res = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    flipHorizontal = cv2.flip(gray, 1)
    #img = Image.open("./C.jpg").convert('L')
    #imagen = img.resize((28,28))
    numpydata = np.asarray(flipHorizontal, dtype="float32")
    #imagen = imagen.transpose(Image.FLIP_LEFT_RIGHT)
    predicciones = predict_image(numpydata)
    print(f'Prueba {imagen}: ',' Prediccion: ',Classes[predicciones[0].item()])