#! /usr/bin/python3

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers import Dense

def create_model(nodelist):
    model = Sequential()
    model.add(Dense(784, input_shape=(784,)))
    for nb_nodes in nodelist:
        model.add(Dense(nb_nodes))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

trn_data = pd.read_csv("Model_Boye/train.csv")
tst_data = pd.read_csv("Model_Boye/test.csv")
trn_labels = trn_data['label'].values
trn_img = 1/255 * trn_data.drop(['label'], axis=1).values

# TODO convert trn_labels to one hot encoding

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(trn_labels)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

trn_enc_labels = onehot_encoded

new_model = create_model([10, 10])

new_model.fit(x=trn_img, y=trn_enc_labels, epochs=10, batch_size=128, verbose=2)
