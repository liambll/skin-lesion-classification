# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:29:37 2019

@author: liam.bui

This file contains functions to train and evaluate convolutional neural networks
"""

import os
import sys
sys.path.insert(0, os.getcwd()) # add current working directory to pythonpath

import numpy as np
import pandas as pd
import warnings
import argparse
import gc
from sklearn.utils import class_weight
from utils.data import get_prediction_score
from utils import config

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"]="0" # Use only the 1st GPU
tf_config = tf.ConfigProto()
set_session(tf.Session(config=tf_config))
from keras import applications, callbacks, optimizers
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
K.tensorflow_backend.set_image_dim_ordering('tf')


def create_model(backbone='mobilenet'):
    """Build a CNN model with transfer learning from well-known architectures 
    :param backbone: str, backbone architecture to use, can be mobilenet, inceptionV3, or resnet50
    :return model: Keras model
    :return img_size: size of input image for the model
    :return preprocessing_function: function to preprocess image for the model
    """ 
    
    if backbone == 'mobilenet':
        img_size = 224
        base_model = applications.mobilenet.MobileNet(include_top=False, weights='imagenet',
                                          input_shape=(img_size, img_size, 3),
                                            pooling=None)
        preprocessing_function = applications.mobilenet.preprocess_input
    elif backbone == 'inceptionV3':
        img_size = 299
        base_model = applications.inception_v3.InceptionV3(include_top=False, weights='imagenet',
                                              input_shape=(img_size, img_size, 3),
                                                pooling=None)
        preprocessing_function = applications.inception_v3.preprocess_input
    elif backbone == 'resnet50':
        img_size = 224
        base_model = applications.resnet.ResNet50(include_top=False, weights='imagenet',
                                              input_shape=(img_size, img_size, 3),
                                                pooling=None)
        preprocessing_function = applications.resnet.preprocess_input
    else:
        raise ValueError('Backbone can only be mobilenet, inceptionV3, or resnet101.')
        
    for layer in base_model.layers:
        layer.trainable = False # disable training of backbone
    x = base_model.output
    
    # Add classification head
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(len(config.CLASSES), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    return model, img_size, preprocessing_function


def build_cnn_model(train_path, test_path,
                     backbone='mobilenet', batch_size=32, nb_epochs=100, lr=0.001,
                     save_path=None):
    """Train and evaluate CNN model
    :param train_path: path to train set,  which should have the below structure:
        train_path
            |---class_1
            |---class_2
            ...
            |---class_N
    :param test_path: path to test set,  which should have the below structure:
        test_path
            |---class_1
            |---class_2
            ...
            |---class_N
    :param backbone: str, contains backbone model name, which can be 'mobilenet', 'inceptionV3', 'resnet50'
    :param batch_size: int, batch size for model training
    :param nb_epochs: int, number of training epoches
    :param lr: float, learning rate
    :param save_path: path to save model
    :return model: fitted Keras model
    :return scores: dict, scores on test set for the fitted Keras model
    """
    
    # Create model
    model, img_size, preprocessing_function = create_model(backbone=backbone)
    
    # Prepare train and val data generator
    train_datagen = ImageDataGenerator(
            preprocessing_function=preprocessing_function,
            rotation_range=90,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True)
    train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            shuffle=True,
            class_mode='categorical')
    y_train = train_generator.classes
    
    # Compute class weights
    weight_list = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    weight_dict = {}
    for i in range(len(np.unique(y_train))):
        weight_dict[np.unique(y_train)[i]] = weight_list[i]
    
    val_datagen = ImageDataGenerator(
            preprocessing_function=preprocessing_function)
    val_generator = val_datagen.flow_from_directory(
            test_path,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            shuffle=False,
            class_mode='categorical')
    
    # Callback list
    callback_list = []
    # monitor val_loss and terminate training if no improvement
    early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, \
                patience=30, verbose=2, mode='auto', restore_best_weights=True)
    callback_list.append(early_stop)
    
    if save_path is not None:
        # save best model based on val_acc during training
        checkpoint = callbacks.ModelCheckpoint(os.path.join(save_path, backbone + '.h5'), monitor='val_acc', \
                    verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
        callback_list.append(checkpoint)
        
    # Train only classification head
    optimizer = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit_generator(train_generator, validation_data=val_generator, epochs=nb_epochs, \
                        class_weight=weight_dict, callbacks=callback_list, verbose=2)
    
    # Train all layers
    for layer in model.layers:
        layer.trainable = True
    optimizer = optimizers.SGD(lr=lr/10.0, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit_generator(train_generator, validation_data=val_generator, epochs=nb_epochs, \
                        class_weight=weight_dict, callbacks=callback_list, verbose=2)
    
    # Evaluate the model
    y_val = val_generator.classes
    y_val_predict_prob = model.predict_generator(val_generator)
    y_val_predict = np.argmax(y_val_predict_prob, axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # disable the warning on f1-score with not all labels
        scores = get_prediction_score(y_val, y_val_predict)
        
    return model, scores
    
    
if __name__ == '__main__':
    data_path = os.path.join(config.WORK_DIRECTORY, config.DATASET_FOLDER)
    cnn_model_names = ['mobilenet', 'inceptionV3', 'resnet50']
    batch_size = 16
    nb_epochs = 100
    lr = 0.001
    save_path = config.WORK_DIRECTORY

    # parse parameters
    parser = argparse.ArgumentParser(description='Build CNN models')
    parser.add_argument('--data_path', help='A path to folder containing train and test datasets')
    parser.add_argument('--batch_size', type=int, help='Batch size for model training')
    parser.add_argument('--nb_epochs', type=int, help='Number of training epoches')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--save_path', help='A path to save fitted models')
    
    args = parser.parse_args()
    if args.data_path:
        data_path = args.data_path
    if args.batch_size:
        batch_size = args.batch_size
    if args.nb_epochs:
        nb_epochs = args.nb_epochs
    if args.lr:
        lr = args.lr
    if args.save_path:
        save_path = args.save_path
        
    # Get path to train and test sets
    train_path = os.path.join(data_path, config.TRAIN_FOLDER)
    test_path = os.path.join(data_path, config.TEST_FOLDER)
    
    # Make save_path
    if save_path is not None:
        os.makedirs(os.path.join(save_path, 'cnn_models'), exist_ok=True)
    
    # Build CNN models
    cnn_model_scores = []
    for backbone in cnn_model_names:
        model, scores = build_cnn_model(train_path, test_path,
                     backbone=backbone, batch_size=batch_size, nb_epochs=nb_epochs, lr=lr,
                     save_path=os.path.join(save_path, 'cnn_models'))
        cnn_model_scores.append(scores)
        print(backbone, scores)
        
        # force release memory
        K.clear_session()
        del model
        gc.collect()
        
    # Summarize model performance
    model_df = pd.DataFrame({'model': cnn_model_names,
                             config.METRIC_ACCURACY: [score[config.METRIC_ACCURACY] for score in cnn_model_scores],
                            config.METRIC_F1_SCORE: [score[config.METRIC_F1_SCORE] for score in cnn_model_scores],
                            config.METRIC_COHEN_KAPPA: [score[config.METRIC_COHEN_KAPPA] for score in\
                                                        cnn_model_scores],
                            config.METRIC_CONFUSION_MATRIX: [score[config.METRIC_CONFUSION_MATRIX] for score in\
                                                             cnn_model_scores]                            
                             })
    model_df = model_df[['model', config.METRIC_ACCURACY, config.METRIC_F1_SCORE, config.METRIC_COHEN_KAPPA,
                         config.METRIC_CONFUSION_MATRIX]]
    model_df.to_csv(os.path.join(config.WORK_DIRECTORY, 'summary_cnn_model.csv'), index=False)
    model_df.sort_values(by=[config.METRIC_ACCURACY, config.METRIC_F1_SCORE, config.METRIC_COHEN_KAPPA],
                         ascending=False, inplace=True)
    print('Best model:\n' + str(model_df.iloc[0]))



    
    