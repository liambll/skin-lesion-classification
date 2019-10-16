# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 15:02:06 2019

@author: liam.bui

Unittest for models/feature_extraction.py

"""

import unittest
from models import feature_extraction
import numpy as np


class TestData(unittest.TestCase):
    def setUp(self):
        global img
        # create an image with horizontal red-bue-black stripe
        img = np.zeros((30, 30, 3), dtype='uint8') # a 6x6 black image
        img[:10, :] = [0, 0, 255]
        img[10:20, :] = [255, 0, 0]
        
    def test_extract_hu_moments(self):
        global img
        features = feature_extraction.extract_hu_moments(img)
        expected = [0.003, 0, 0, 0, 0, 0, 0]
        np.testing.assert_almost_equal(features, expected, decimal=3)
        
    def test_extract_zernike_moments(self):
        global img
        features = feature_extraction.extract_zernike_moments(img)
        expected = [0.318, 0, 0.508, 0.101, 0.0369, 0.00123, 0.15469, 0.237, 0.00569, 0.099, 0.0753, 0.0253,
                    0.0463, 0.1566, 0.0143, 0.01789, 0.11426, 0.14329, 0.0891, 0.01855, 0.03801, 0.04166, 0.06308,
                    0.04557, 0]
        np.testing.assert_almost_equal(features, expected, decimal=2)
        
    def test_extract_haralick(self):
        global img
        features = feature_extraction.extract_haralick(img)
        expected = [0.3009, 78.8793, 0.9591, 968.4431, 0.9483, 69.8448, 3794.8931, 1.8262, 1.8779, 0.01154, 0.3232,
                    -0.81488, 0.95939]
        np.testing.assert_almost_equal(features, expected, decimal=3)
        
    def test_extract_lbp(self):
        global img
        features = feature_extraction.extract_lbp(img, numPoints=8)
        expected = [0, 0.0355, 0.0533, 0.1333, 0.0711, 0.1822,
                    0, 0.04, 0.3644, 0.12]
        np.testing.assert_almost_equal(features, expected, decimal=3)
        
    def test_extract_color_histogram(self):
        global img
        features = feature_extraction.extract_color_histogram(img, n_bins=2)
        expected = [0.5773, 0, 0, 0.5773, 0, 0, 0, 0.5773]
        np.testing.assert_almost_equal(features, expected, decimal=3)