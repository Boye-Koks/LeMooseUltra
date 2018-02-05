#! /usr/bin/python3

import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense

def create_model(nodelist):
    model = Sequential()
    model.add(Dense(784, input_shape=(784,)))
    for nb_nodes in nodelist:
        model.add(Dense(nb_nodes))
    model.add(Dense(10))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    print(model.summary())
    return model

trn_data = pd.read_csv("Model_Boye/train.csv")
tst_data = pd.read_csv("Model_Boye/test.csv")
trn_labels = trn_data['label'].values
trn_img = trn_data.drop(['label'], axis=1).values

create_model([128,128])
