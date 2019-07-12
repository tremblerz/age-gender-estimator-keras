import argparse
import logging
import os
import random
import string

import pandas as pd
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.optimizers import SGD
import keras.backend as K
from keras.models import load_model

from config import FINAL_WEIGHTS_PATH, IMG_SIZE
from data_generator import ImageGenerator
from data_loader import DataManager, split_imdb_data
from models.mobile_net import MobileNetDeepEstimator
from utils import mk_dir
from adversary import fgsm
from keras.metrics import categorical_accuracy
import tensorflow as tf

logging.basicConfig(level=logging.DEBUG)


def get_adversarial_acc_metric_gender(model, eps=0.3, clip_min=0.0, clip_max=1.0):
    def adv_acc(y, _):
        # Generate adversarial examples
        x_adv = fgsm(model, y, eps=eps, clip_min=clip_min, clip_max=clip_max)
        # Consider the attack to be constant
        x_adv = K.stop_gradient(x_adv)
        
        # Accuracy on the adversarial examples
        preds_gender, _ = model(x_adv)
        return categorical_accuracy(y, preds_gender)
    return adv_acc

def get_adversarial_acc_metric_age(model, eps=0.3, clip_min=0.0, clip_max=1.0):
    def adv_acc(y, _):
        # Generate adversarial examples
        y_gender = tf.get_default_graph().get_tensor_by_name("gender_target:0")
        x_adv = fgsm(model, y_gender, eps=eps, clip_min=clip_min, clip_max=clip_max)
        # Consider the attack to be constant
        x_adv = K.stop_gradient(x_adv)
        
        # Accuracy on the adversarial examples
        _, preds_age = model(x_adv)
        return categorical_accuracy(y, preds_age)
    return adv_acc

def get_args():
    parser = argparse.ArgumentParser(description="This script evaluates the CNN model for age and gender estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=str, required=True,
                        help="path to input database mat file")
    parser.add_argument("--model_weights", type=str, required=True,
                        help="path to model weights file")
    parser.add_argument("--eps", type=float, required=True,
                        help="epsilon value for adversarial perturbation")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size")
    parser.add_argument("--validation_split", type=float, default=0.2,
                        help="validation split ratio")

    args = parser.parse_args()
    return args

def main():

    args = get_args()
    input_path = args.input
    eps = args.eps
    batch_size = args.batch_size
    validation_split = args.validation_split
    model_weights = args.model_weights

    logging.debug("Loading data...")

    dataset_name = 'imdb'
    data_loader = DataManager(dataset_name, dataset_path=input_path)
    ground_truth_data = data_loader.get_data()
    train_keys, val_keys = split_imdb_data(ground_truth_data, validation_split)

    print("Samples: Training - {}, Validation - {}".format(len(train_keys), len(val_keys)))
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    images_path = 'data/imdb_crop/'

    n_age_bins = 21
    alpha=1
    model = MobileNetDeepEstimator(input_shape[0], alpha, n_age_bins, weights='imagenet')()

    image_generator = ImageGenerator(ground_truth_data, batch_size,
                                     input_shape[:2],
                                     train_keys, val_keys,
                                     path_prefix=images_path,
                                     vertical_flip_probability=0
                                     )


    opt = SGD(lr=0.001)
    adv_acc_metric_gender = get_adversarial_acc_metric_gender(model, eps=eps)
    adv_acc_metric_age = get_adversarial_acc_metric_age(model, eps=eps)

    model.compile(
        optimizer=opt,
        loss={'gender':'binary_crossentropy',
            'age':'categorical_crossentropy'},
        metrics={'gender':adv_acc_metric_gender,
            'age':adv_acc_metric_age}
    )
    model.load_weights(model_weights)

    eval_list = model.evaluate_generator(image_generator.flow(mode='val'),
            steps=int(len(val_keys) / batch_size))

    print(eval_list)

    del(model)
    K.clear_session()

    model = MobileNetDeepEstimator(input_shape[0], alpha, n_age_bins, weights='imagenet')()
    model.load_weights(model_weights)
    model.compile(
        optimizer=opt,
        loss={'gender':'binary_crossentropy',
            'age':'categorical_crossentropy'},
        metrics={'gender':'accuracy',
            'age':'accuracy'}
    )

    eval_list = model.evaluate_generator(image_generator.flow(mode='val'),
            steps=int(len(val_keys) / batch_size))

    print(eval_list)


if __name__ == '__main__':
    main()
