#! /usr/bin/python3
import pandas as pd
import numpy as np
import utils as utils
import lib.TF_lib as model_lib

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

# Setup the cross validation
skf = StratifiedKFold(n_splits=5)
for fold_nb, (trn_idx, val_idx) in enumerate(skf.split(_img, class_lbl)):
    # Create training and validation split
    trn_img = _img[trn_idx]
    trn_lbl = _lbl[trn_idx]
    val_img = _img[val_idx]
    val_lbl = _lbl[val_idx]

    # Create our model
    MLP = model_lib.MLP(input_size=[trn_img.shape[1]], nb_class=_lbl.shape[1])
    trn_loss, val_loss, trn_acc, val_acc = MLP.fit(
        trn_data=[trn_img, trn_lbl],
        val_data=[val_img, val_lbl],
        nb_epoch=5,
        batch_sz=512,
        verbose=1
    )
