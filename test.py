from keras.models import load_model
import numpy as np 
from keras.preprocessing.image import img_to_array
import cv2

model_path= 'models/simple_CNN.40-0.97.hdf5'
model = load_model(model_path,compile=False)

img_path = 'data/brokoli/brokoli.jpg'

image = cv2.imread(img_path)

image = cv2.resize(image,(150,150))

image = image.astype("float") /255.0
image = np.expand_dims(image,axis=0)
# image = img_to_array(image)
preds = model.predict(image)[0]
label =['Brokoli','Wortel']

label = label[preds.argmax()]

print(label)