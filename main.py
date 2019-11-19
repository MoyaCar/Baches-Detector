from fastai.vision import *
import cv2 as cv
import time
import threading

#variables para modelo
root = '.'
path = Path(root)
classes = ['Negative data', 'Positive data']
data = ImageDataBunch.single_from_classes(path, classes, ds_tfms=get_transforms(do_flip=False),size=224).normalize(imagenet_stats)
learn = cnn_learner(data,models.resnet34)
#carga del modelo
learn.load('baches224_rn34')

#variables para prediccion
img = open_image(path/'frame.jpg')

#variables globales para opencv
cap = cv.VideoCapture(0)
fp = 0
terminado = False
prediccion = ''
font = cv.FONT_HERSHEY_SIMPLEX 
org = (50, 50) 
fontScale = 1
color = (255, 255, 255) 
thickness = 2

def reproducir_video():
    while(True):
        global terminado
        global fp
        global img

        fp = fp +1
        ret, frame = cap.read()

        frame = cv.putText(frame, prediccion, org, font,fontScale, color, thickness, cv.LINE_AA)
        cv.imshow('video', frame)

        if (fp % 90 == 0):
            cv.imwrite('frame.jpg',frame)
            print('captura guardada')
            img = open_image(path/'frame.jpg')
            print('imagen asignada')

        if cv.waitKey(1) & 0xFF == ord('q'):
            terminado = True
            break
    cap.release()
    cv.destroyAllWindows()

def analisis_segundoplano():
    time.sleep(10)
    while(terminado == False):

        if (fp % 90 == 0):
            predecir_img()

        if (terminado == True):
            break
    

def predecir_img():
    global prediccion
    start = time.time()

    pred_class,pred_idx,outputs = learn.predict(img)
    print(pred_class)
    print(outputs)

    if (outputs[pred_idx] > 0.8):
        print('Predicción segura')
        prediccion = str(pred_class)
    else:
        prediccion = 'Prediccion no segura'

    end = time.time()
    prediction_time = end - start
    print(f'Tiempo: {prediction_time} segundos para predecir')
    
    

t1 = threading.Thread(target=reproducir_video)
t2 = threading.Thread(target=analisis_segundoplano)

t1.start()
t2.start()

t1.join()
t2.join()

print(fp)