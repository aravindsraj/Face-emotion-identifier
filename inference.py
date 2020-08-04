import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input

from_path = ''                               # Input image path
to_path = ''                                 # Output image path
image_name = ''                              # Image name
predict_model = load_model('emotifier.h5')

# To draw the rectangle over the face and to mention the emotion it predicted
def face(image, filename, emotion):
    facedata = "./haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    img = cv2.imread(image)
    gray = cv2.cvtColor(img,  cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.3, 5)
    print(faces)

    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))
        cv2.putText(img, emotion, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        cv2.imwrite(to_path+str(filename)+".jpg", img)
    return img

img = image.load_img(from_path+image_name, target_size=(250, 250))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
pred = predict_model.predict(x)

print("### Pred ###", pred)

# Feel feel to increase/decrease the loop based on your classes
if pred[0][0] > 0.5:
  print("predicted array : ", pred[0][0])
  print("The image which you have uploaded is "+str(class1))
  op = face(from_path+image_name , image_name, str(class1))
elif pred[0][1] > 0.5:
  print("predicted array : ", pred[0][1])
  print("The image which you have uploaded is "+str(class2))
  op = face(from_path+image_name , image_name, str(class2))
elif pred[0][2] > 0.5:
  print("predicted array : ", pred[0][2])
  print("The image which you have uploaded is "+str(class3))
  op = face(from_path+image_name , image_name, str(class3))
elif pred[0][3] > 0.5:
  print("predicted array : ", pred[0][3])
  print("The image which you have uploaded is "+str(class4))
  op = face(from_path+image_name , image_name, str(class4))
else:
  pass
