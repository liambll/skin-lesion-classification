# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 15:02:06 2019

@author: liam.bui

Unittest for utils/data.py

"""

import unittest
from utils.data import get_prediction_score
import numpy as np


class TestData(unittest.TestCase):
    def test_get_prediction_score(self):
        y_label = [1, 1, 1, 2, 2, 2]
        y_predict = [1, 1, 2, 1, 1, 2]
        scores = get_prediction_score(y_label, y_predict)
        expected = {'Cohen kappa': 0.0, 'Confusion Matrix': [[2, 1], [2, 1]], 'accuracy': 0.5, 'f1-score': 0.48}
        for key in scores:
            np.testing.assert_almost_equal(scores[key], expected[key], decimal=1)