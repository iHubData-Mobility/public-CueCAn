import tensorflow as tf
import keras
import numpy as np
import os
import glob
import cv2
from PIL import Image
import argparse
import tensorflow.keras.applications as app
import tensorflow.keras.utils as tfutil
from numpy import savetxt
import pandas as pd
from ICAU import *
from ICAU_5_5_edge import *

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_dir", help="Path to the h5 model file")
parser.add_argument("-i", "--img_dir", help="path to image data")
args = parser.parse_args()

print("Loading the Model ", args.model_dir, " ...\n")
model = keras.models.load_model(args.model_dir, custom_objects={
                                "InpaintContextAttentionUnit": InpaintContextAttentionUnit, "InpaintContextAttentionUnit5edge": InpaintContextAttentionUnit5edge})

print("Loaded Model.")

#Pre-processing the input image
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg19.preprocess_input,
    rescale=1.0/255.0)

image_generator = datagen.flow_from_directory(
    directory=args.img_dir,
    classes=['no_context', 'context'],
    target_size=(252, 448),
    batch_size=32, shuffle=False,
    class_mode='binary')

res = (model.predict(image_generator))

ground_truth = (image_generator.labels)
file_names = []


for file in image_generator.filenames:
    file_names.append(os.path.basename(file))


if os.path.exists("out_classifier/"):
    for idx,file in enumerate(file_names):
        img = cv2.imread(file)[:,:, ::-1]
        img = cv2.putText(img, "Classifier Score: "+str(res[idx]), (10,10),cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0),1,cv2.LINE_AA)
        cv2.imwrite("out_classifier/"+file, img)
        
else:
    os.mkdir("out_classifier/")
    for idx,file in enumerate(file_names):
        img = cv2.imread(file)
        img = cv2.putText(img, "Classifier Score: "+str(res[idx]), (10,10),cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0),1,cv2.LINE_AA)
        cv2.imwrite("out_classifier/"+file, img)