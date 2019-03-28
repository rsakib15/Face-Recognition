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

face_detector = dlib.get_frontal_face_detector()
face_encoder = dlib.face_recognition_model_v1('./data/model/dlib_face_recognition_resnet_model_v1.dat')
face_pose_predictor = dlib.shape_predictor('./data/model/shape_predictor_68_face_landmarks.dat')


def get_detected_faces(filename):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, face_detector(image, 1)


def get_face_encoding(image, detected_face):
    pose_landmarks = face_pose_predictor(image, detected_face)
    face_encoding = face_encoder.compute_face_descriptor(image, pose_landmarks, 1)
    return np.array(face_encoding)


def training(people):
    df = pd.DataFrame()
    for p in people:
        l = []
        for filename in glob.glob('./data/train/%s/*' % p):
            image, face_detect = get_detected_faces(filename)
            face_encoding = get_face_encoding(image, face_detect[0])
            l.append(np.append(face_encoding, [p]))
        temp = pd.DataFrame(np.array(l))
        df = pd.concat([df, temp])
    df.reset_index(drop=True, inplace=True)

    le = LabelEncoder()
    y = le.fit_transform(df[128])
    print("Training for {} classes.".format(len(le.classes_)))
    X = df.drop(128, axis=1)
    print("Training with {} pictures.".format(len(X)))

    clf = SVC(C=1, kernel='linear', probability=True)
    clf.fit(X, y)

    fName = "./classifier.pkl"
    print("Saving classifier to '{}'".format(fName))
    with open(fName, 'wb') as f:
        pickle.dump((le, clf), f)


def predict(filename, le=None, clf=None, verbose=False):
    print(filename)
    if not le and not clf:
        with open("./classifier.pkl", 'rb') as f:
            (le, clf) = pickle.load(f)
    image, detected_faces = get_detected_faces(filename)
    prediction = []
    if verbose:
        print('{} faces detected.'.format(len(detected_faces)))
    img = np.copy(image)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for face_detect in detected_faces:
        # draw bounding boxes
        cv2.rectangle(img, (face_detect.left(), face_detect.top()),
                      (face_detect.right(), face_detect.bottom()), (255, 0, 0), 2)
        start_time = time.time()
        # predict each face
        p = clf.predict_proba(get_face_encoding(image, face_detect).reshape(1, 128))
        # throwing away prediction with low confidence
        a = np.sort(p[0])[::-1]
        if a[0] - a[1] > 0.5:
            y_pred = le.inverse_transform(np.argmax(p))
            prediction.append([y_pred, (face_detect.left(), face_detect.top()),
                               (face_detect.right(), face_detect.bottom())])
        else:
            y_pred = 'unknown'
        # Verbose for debugging
        if verbose:
            print('\n'.join(['%s : %.3f' % (k[0], k[1]) for k in list(zip(map(le.inverse_transform,
                                                                              np.argsort(p[0])),
                                                                          np.sort(p[0])))[::-1]]))
            print('Prediction took {:.2f}s'.format(time.time() - start_time))

        cv2.putText(img, y_pred, (face_detect.left(), face_detect.top() - 5), font, np.max(img.shape[:2]) / 1800,
                    (255, 0, 0))
    return img, prediction


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, help='train or predict')
    parser.add_argument('--training_data',type=str,help="Path to training data folder.",default='./data/train/')
    parser.add_argument('--testing_data',type=str,help="Path to test data folder.",default='./data/test/test.jpg')

    args = parser.parse_args()
    people = os.listdir(args.training_data)
    print('{} people will be classified.'.format(len(people)))
    if args.mode == 'train':
        training(people)
    elif args.mode == 'test':
        with open("./classifier.pkl", 'rb') as f:
            (le, clf) = pickle.load(f)
        for i, f in enumerate(glob.glob(args.testing_data)):
            img, _ = predict(f, le, clf)
            cv2.imwrite(args.testing_data + 'test_{}.jpg'.format(i), img)
