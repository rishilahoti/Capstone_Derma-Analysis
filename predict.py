"""
    This file used to load our model and predict batch of images from a directory.
"""
import os
import numpy as np
from tqdm import tqdm
import tensorflow.keras as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array

model = K.models.load_model("model.h5")
types = ['acne', 'carcinoma', 'eczema', 'keratosis', 'millia', 'rosacea']

img_path = os.listdir('images_to_predict_dir..')
for i in tqdm(dict.keys()):
    fname = './dataset_dir'+'/'+i+'/'
    images = os.listdir(fname)
    for img_path in images:
        img = image.load_img(fname+img_path, target_size=(299, 299))
        x = img_to_array(img)
        x = K.applications.xception.preprocess_input(x)

    prediction = model.predict(np.array([x]))[0]
    test_pred = np.argmax(prediction)

    result = [(types[i], float(prediction[i]) * 100.0) for i in range(len(prediction))]
    result.sort(reverse=True, key=lambda x: x[1])

    print(f'Image name: {i}')
    for j in range(6):
        (class_name, prob) = result[j]
        print("Top %d ====================" % (j + 1))
        print(class_name + ": %.2f%%" % (prob))
    print("\n")
