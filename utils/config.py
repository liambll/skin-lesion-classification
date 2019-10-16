# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 15:10:54 2019

@author: liam.bui

The file contains configuration and shared variables

"""

##########################
## FOLDER STURCTURE ######
##########################
WORK_DIRECTORY = 'C:/Users/liam.bui/Desktop/skin-legion/'
ORIGINAL_DATA_FOLDER = 'temples-train-hard'
DATASET_FOLDER = 'skin-cancer-malignant-vs-benign'
TRAIN_FOLDER = 'train'
TEST_FOLDER = 'test'

##########################
## EVALUATION METRICS ####
##########################
METRIC_ACCURACY = 'accuracy'
METRIC_F1_SCORE = 'f1-score'
METRIC_COHEN_KAPPA = 'Cohen kappa'
METRIC_CONFUSION_MATRIX = 'Confusion Matrix'

###############
## MODEL ######
###############
CLASSES = ['benign', 'malignant']