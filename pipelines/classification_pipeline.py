# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 23:18:04 2019

@author: liam.bui

This file is the main pipeline to make prediction on new images
"""

import os
import sys
sys.path.insert(0, os.getcwd())  # add current working directory to pythonpath
import numpy as np
import pandas as pd

from keras.models import load_model
from keras import applications

from utils.data import get_image_generator
from utils import config
import argparse

if __name__ == '__main__':
    # parse parameters
    parser = argparse.ArgumentParser(description='Make prediction on new images')
    parser.add_argument('--data_path', help='A path to folder containing new images with extension .png, .jpg, .jpeg')
    parser.add_argument('--model_path', help='A path to h5 model file')
    parser.add_argument('--img_size', default=299, type=int, help='Image size for model input')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size for model inference')
    parser.add_argument('--save_path', help='A path to folder to save result.csv')

    args = parser.parse_args()
    if args.data_path:
        data_path = args.data_path
    else:
        data_path = input("Please enter a path to folder containing new images: ") 
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = input("Please enter a path to h5 model file: ")
    if args.batch_size:
        batch_size = args.batch_size
    if args.img_size:
        img_size = args.img_size
    if args.save_path:
        save_path = args.save_path
    else:
        save_path = data_path
        
    # Check input
    if not os.path.isdir(data_path):
        print('Data path does not exist.\n')
        sys.exit()
    if not os.path.exists(model_path):
        print('Model file does not exist.\n')
        sys.exit()
    if not os.path.isdir(data_path):
        print('Save path does not exist.\n')
        sys.exit()

    # Create data generator
    filenames = [filename for filename in os.listdir(data_path)
                 if os.path.splitext(filename)[-1].lower() in config.IMG_EXTENSION]
    image_generator = get_image_generator(data_path, batch_size=batch_size, img_size=(img_size, img_size),
                                          preprocess_input=applications.inception_v3.preprocess_input)

    if len(filenames) > 0:
        # Load pre-trained model
        print('Loading model file ...\n')
        model = load_model(model_path)

        # Make prediction
        print('Processing ' + str(len(filenames)) + ' images ...\n')
        prediction = []
        for i in range(int(np.ceil(len(filenames) / batch_size))):
            img_batch = image_generator.__next__()
            y_new_predict_prob = model.predict(img_batch)
            y_new_predict = np.argmax(y_new_predict_prob, axis=1)
            y_new_label = [config.CLASSES[i] for i in y_new_predict]
            prediction += y_new_label
        prediction = prediction[:len(filenames)]

        # Save prediction result
        prediction_df = pd.DataFrame({'file name': filenames,
                                      'detected country': prediction})
        prediction_df = prediction_df[['file name', 'detected country']]
        prediction_df.to_csv(os.path.join(save_path, 'result.csv'), index=False)
        print('Classification result is saved to ' + os.path.join(save_path, 'result.csv') + '\n')
