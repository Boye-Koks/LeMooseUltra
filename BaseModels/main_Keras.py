#! /usr/bin/python3
import pandas as pd
import numpy as np
import utils as utils
import lib.Keras_lib as model_lib

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold

# Get the data
data = pd.read_csv('BaseModels/data/train.csv')
_tst = pd.read_csv('BaseModels/data/test.csv')

# Extract the labels
_lbl = data[['label']]
_img = data.drop(['label'], axis=1)

# Convert to numpy arrays
_lbl = _lbl.values
_img = np.array(_img, dtype=np.float32)
_tst = np.array(_tst, dtype=np.float32)
del data

# We 'normalize the images, such that features have comparable input strength', since the test,
# train and validation sets will be quite similar we do it 'quick and dirty'. If the data is not
# as nice we should do it with the folds!
scaler = MinMaxScaler()
scaler.fit(_img)

_img = scaler.transform(_img)
_tst = scaler.transform(_tst)


# transform the label to categorical
class_lbl = _lbl
_lbl = utils.to_categorical(_lbl)

# Make it an image
_img = np.reshape(_img, (42000, 28, 28, 1))

# Setup the cross validation
skf = StratifiedKFold(n_splits=5)
for fold_nb, (trn_idx, val_idx) in enumerate(skf.split(_img, class_lbl)):
    # Create training and validation split
    trn_img = _img[trn_idx]
    trn_lbl = _lbl[trn_idx]
    val_img = _img[val_idx]
    val_lbl = _lbl[val_idx]



    # Create our model
    MLP = model_lib.crt_CNN((28, 28, 1), layer_list=[512, 512], act_func='relu', use_bias=False)
    MLP.fit(trn_img,
            trn_lbl,
            validation_data=[val_img, val_lbl],
            epochs=2,
            batch_size=512,
            verbose=2
           )
