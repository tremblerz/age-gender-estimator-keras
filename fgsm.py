import tensorflow as tf
from keras.utils import np_utils
from keras.models import load_model
import keras.backend as K
from data_loader import DataManager
import random 
import numpy as np
import cv2
from models.mobile_net import MobileNetDeepEstimator
from config import IMG_SIZE
from preprocessor import _imread as imread
from preprocessor import _imresize as imresize
from matplotlib import pyplot as plt
import argparse

def load_all_images(input_path):
    dataset_name = 'imdb'
    data_loader = DataManager(dataset_name, input_path)
    data = data_loader.get_data()
    return data

def random_image():
    input_path = 'data/imdb_crop/'
    data = load_all_images('data/imdb_crop/imdb.mat')
    img_path = random.choice(list(data.keys()))
    
    gender, age = data[img_path] 
    gender_one_hot = np_utils.to_categorical(gender, 2)
    age_bins = np.linspace(0,100,21)
    age_step = np.digitize(age, age_bins)
    age_quantized = np_utils.to_categorical(age_step, 21)
    
    return input_path+img_path, gender_one_hot, age_quantized

def preprocess_image(image, input_shape):
    image = imresize(image, input_shape[:2])
    image = image.astype('float32')
    image = image/255.0
    image = image - 0.5
    image = image * 2.0
    image = np.expand_dims(image, axis=0)
    return image

def predict(model, image):
    predictions = model.predict(image)
    return predictions

def FGSM(x, gender_label,model,eps=0.3):
    sess = K.get_session()
    x_adv = x
    loss =  K.mean(K.binary_crossentropy(gender_label, model.output[0]))
    grads = K.gradients(loss, model.input)
    delta = K.sign(grads[0])
    x_adv = x_adv + eps * delta
    x_adv = K.clip(x_adv, 0.0 ,1.0)
    loss_np, gradients, x_adv_array = sess.run([loss, grads, x_adv], feed_dict={model.input:x})
    return x_adv_array

def plot_adversarial(orig_img, adv_img):
    plt.figure(figsize=(8,8))
    for n, img in enumerate([orig_img, adv_img]):
        plt.subplot(2,2,n+1)
        plt.imshow(img[0])
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Save model path')
    parser.add_argument('--epsilon', type=float, help='Epsilon for adversarial perturbation')

    args = parser.parse_args()
    
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    alpha =1
    n_age_bins =21
    model = MobileNetDeepEstimator(input_shape[0], alpha, n_age_bins, weights='imagenet')()
    model.load_weights(args.model)
    
    image_path, gender_one_hot, age_quantized = random_image()
    image = imread(image_path)
    image = preprocess_image(image, input_shape)

    predictions = predict(model, image)
    gender_orig_img, age_orig_img = predictions

    x_adv = FGSM(image, np.expand_dims(gender_one_hot, axis=0), model, eps=args.epsilon)
 
    predictions = predict(model, x_adv)
    gender_adv_img, age_adv_img = predictions

    age_result = 'Actual age bin: {}, Predicted age bin for original image: {}, Predicted age bin for adversarial image: {}'
    gender_result = 'Actual gender: {}, Predicted gender for original image: {}, Predicted gender for adversarial image: {}'

    print(age_result.format(np.argmax(age_quantized), np.argmax(age_orig_img), 
        np.argmax(age_adv_img)))
    print(gender_result.format(np.argmax(gender_one_hot), np.argmax(gender_orig_img), 
        np.argmax(gender_adv_img)))

    plot_adversarial(image, x_adv)

if __name__ == '__main__':
    main()

    
