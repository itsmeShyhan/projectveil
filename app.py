from flask import Flask, render_template, request

import cv2 as cv

# cv.read
# from keras.preprocessing import image_dataset

# load_image = image_dataset.load_image
# image_dataset.
# from keras.preprocessing.image import img_to_array
# from keras.applications.vgg16 import preprocess_input
# from keras.applications.vgg16 import decode_predictions
# from keras.applications.vgg16 import VGG16
# from keras.applications.resnet50 import ResNet50
import pickle
import tensorflow as tf
# import numpy as np
# from matplotlib import pyplot as plt
# import os

model = pickle.load(open('imageclassifierConv8e20b64.pkl', 'rb'))

app = Flask(__name__)
# model = ResNet50()

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    # try:
    #     imagefile= request.files['imagefile']
    #     image_path = "./images/" + imagefile.filename
    #     imagefile.save(image_path)

    #     img = cv.imread(image_path)
    #     resize = tf.image.resize(img, (256,256))
    #     # plt.imshow(resize.numpy().astype(int))
    #     # plt.show()

    #     yhat = model.predict(np.expand_dims(resize/255, 0))
    #     try:
    #         os.remove(image_path)
    #         print("Image successfully deletec")
    #     except:
    #         print("Error while removing image")
    #         return "Error has occurred while safely deleting your image"
        
    #     if yhat > 0.5:
    #         return render_template('index.html', prediction='Predicted class is Real')
    #     else:
    #         return render_template('index.html', prediction='Predicted class is Fake')
        
    # except:
    #     print("Error has occurred")
    #     return render_template('index.html', prediction='No image is selected')
        return "aew"
        


    # image = load_img(image_path, target_size=(224, 224))
    # image = img_to_array(image)
    # image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # image = preprocess_input(image)
    # yhat = model.predict(image)
    # label = decode_predictions(yhat)
    # label = label[0][0]

    # classification = '%s (%.2f%%)' % (label[1], label[2]*100)


    # return render_template('index.html', prediction=classification)
    # return 'hi there predict'


if __name__ == '__main__':
    app.run(port=3000)
