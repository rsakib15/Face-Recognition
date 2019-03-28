import dlib
from skimage import io, transform
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import openface
import pickle
import os
import sys
import argparse
import time

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

def training(people):
    pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, help='train or predict')
    parser.add_argument('--training_data',type=str,help="Path to training data folder.",default='./data/train')
    parser.add_argument('--testing_data',type=str,help="path to test data folder.",default='./data/test')

    args = parser.parse_args()
    people = os.listdir(args.training_data)
    print('{} people will be classified.'.format(len(people)))

    if args.mode == 'train':
        print("Train")
        training(people)
    elif args.mode == 'test':
        with open("./classifier.pkl", 'rb') as f:
            (le, clf) = pickle.load(f)
        for i, f in enumerate(glob.glob(args.testing_data)):
            img, _ = predict(f, le, clf)
            cv2.imwrite(args.testing_data + 'test_{}.jpg'.format(i), img)


