import numpy as np
import tensorflow as tf
from tensorflow import keras
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import keras
import glob
from PIL import ImageDraw
import os
from ICAU import *
from ICAU_5_5_edge import *
import argparse
import cv2

def get_img_array(img_path, size):
    
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, text, cam_path="cam.jpg",alpha=0.4):
    # Load the original image
    img = tf.keras.utils.load_img(img_path)
    img = tf.keras.utils.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)
    #T1 = ImageDraw.Draw(superimposed_img)
    #T1.text((20,20),text,fill=(255,0,0))
    #w, h = superimposed_img.size
    #print(w,h)
    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_dir", help="Path to the h5 model file")
parser.add_argument("-i", "--img_dir", help="path to image data")
parser.add_argument("-last", "--last_conv", help="last conv layer to backpropagate till")
args = parser.parse_args()

last_conv_layer = args.last_conv
model_path = args.model_dir
model = keras.models.load_model(model_path, custom_objects = {"InpaintContextAttentionUnit":InpaintContextAttentionUnit,"InpaintContextAttentionUnit5edge":InpaintContextAttentionUnit5edge})

model.layers[-1].activation = None

files = sorted(glob.glob(args.img_dir))
if not os.path.exists():
    os.mkdir("gradOut_classifier/")
    
for file in files:
    img = cv2.imread(file)[:,:, ::-1]
    img = np.expand_dims(img,axis=0)
    img_array = tf.keras.applications.vgg19.preprocess_input(img)
    preds = model.predict(img_array)
    res = tf.keras.activations.sigmoid(preds).numpy()
    for item in res:
        for val in item:
            text = str(val)
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer)
    save_and_display_gradcam(file, heatmap,text=text,cam_path="gradOut_classifier/"+os.path.basename(file))