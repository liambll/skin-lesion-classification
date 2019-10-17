# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 14:58:05 2019

@author: liam.bui

The file contains functions to handle dataset and evaluation metrics on the dataset
"""

import os
import numpy as np
import shutil
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, confusion_matrix
from utils import config
import argparse


def split_train_test(input_path, output_path, test_ratio=0.0, seed=0):
    """Split original data into train data and test data.
    :param input_path: str, path to the original dataset folder, which should have the below structure:
        input_path
            |---class_1
            |---class_2
            ...
            |---class_N
    :param output_path: str, path to the folder to contain train and test dataset. It will have the below structure:
        output_path
            |---train
                 |---class_1
                 |---class_2
                 ...
            |---test
                 |---class_1
                 |---class_2
                 ...
    :param test_ratio: float, proportion of the original data for testset, must be from 0 to 1
    :param seed: int, randomization seed for reproducibility
    """
    
    # Set randomization seed
    np.random.seed(seed)
    
    # Get train and test path
    train_path = os.path.join(output_path, config.TRAIN_FOLDER)
    test_path = os.path.join(output_path, config.TEST_FOLDER)
    
    # Iterate through each class in the original dataset
    folder_list = os.listdir(input_path)
    for folder in folder_list:
        print('Processing: ' + folder)
        
        # Create folder for a class in train path and test path
        os.makedirs(os.path.join(train_path, folder), exist_ok=True)
        os.makedirs(os.path.join(test_path, folder), exist_ok=True)        
        
        folder_path = os.path.join(input_path, folder)
        file_list = os.listdir(folder_path)
        np.random.shuffle(file_list) # randomize files of a class
        
        # Iterate through each file of a class
        for i in range(len(file_list)):
            filename = file_list[i]
            if i < int(len(file_list)*test_ratio): # assign the file to test set
                shutil.copy(os.path.join(input_path, folder, filename), os.path.join(test_path, folder))
            else: # assign the file to train set
                shutil.copy(os.path.join(input_path, folder, filename), os.path.join(train_path, folder))
                
def get_prediction_score(y_label, y_predict):
    """Evaluate predictions using different evaluation metrics.
    :param y_label: list, contains true label
    :param y_predict: list, contains predicted label
    :return scores: dict, evaluation metrics on the prediction
    """
    scores = {}
    scores[config.METRIC_ACCURACY] = accuracy_score(y_label, y_predict)
    scores[config.METRIC_F1_SCORE] = f1_score(y_label, y_predict, labels=None, average='macro', sample_weight=None)
    scores[config.METRIC_COHEN_KAPPA] = cohen_kappa_score(y_label, y_predict)
    scores[config.METRIC_CONFUSION_MATRIX] = confusion_matrix(y_label, y_predict)
    
    return scores
 
if __name__ == '__main__':
    # default param
    input_path = os.path.join(config.WORK_DIRECTORY, config.ORIGINAL_DATA_FOLDER)
    output_path = os.path.join(config.WORK_DIRECTORY, config.DATASET_FOLDER)
    test_ratio = 0.2
    seed = 0

    # parse parameters
    parser = argparse.ArgumentParser(description='Split data into train and test set')
    parser.add_argument('--input_path', help='A path to folder containing original dataset')
    parser.add_argument('--output_path', help='A path to folder to store train and test set')
    parser.add_argument('--test_ratio', type=float, help='Proportion of dataset used for test set')
    parser.add_argument('--seed', type=int, help='Seed for randomization')
    
    args = parser.parse_args()
    if args.input_path:
        input_path = args.input_path
    if args.output_path:
        output_path = args.output_path
    if args.test_ratio:
        test_ratio = args.test_ratio
    if args.seed:
        seed = args.seed
    
    # Perform train/test split
    split_train_test(input_path, output_path, test_ratio, seed)