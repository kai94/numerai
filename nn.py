import numpy as np
import pandas as pd
from keras.models import Model,Sequential
from keras.layers import Input, merge, Convolution2D, MaxPooling2D,AveragePooling2D
from keras.utils import np_utils
from keras.layers.core import Dense,Flatten,Activation,Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.optimizers import Adam,SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping,Callback
from keras import backend as K
from sklearn.cross_validation import KFold,train_test_split
import os
def nn():
    input = Input(shape=(21,))
    out = Dense(200)(input)
    out = BatchNormalization()(out)
    out = ELU()(out)
    out = Dropout(0.5)(out)
    out = Dense(20)(out)
    out = BatchNormalization()(out)
    out = ELU()(out)
    out = Dropout(0.2)(out)
    out = Dense(2,activation='softmax')(out)
    model = Model(input=input,output=out)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-4),
                  metrics=['accuracy'])
    return model


dat = pd.read_csv('numerai_training_data.csv')

target = dat['target']
dat.drop(['target'],axis=1, inplace = True)
print dat.columns
dat = dat.as_matrix()
mean = np.mean(dat,axis=0)
std = np.std(dat,axis=0)
dat = (dat - mean)/std
X_pred = pd.read_csv('numerai_tournament_data.csv')
tid = X_pred['t_id']
X_pred.drop(['t_id'],axis=1, inplace = True)
X_pred = X_pred.as_matrix()
X_pred = (X_pred - mean)/std
np.save('tid',np.array(tid))
if __name__ == '__main__':
    np.random.seed(2016)
    random_state = 2016
    kf = KFold(len(dat),n_folds=5,shuffle=True,random_state = random_state)

    predictions = []
    val_loss = []
    num_fold = 0
    for train_idx,test_idx in kf:    
        num_fold+=1
        X_train,y_train = dat[train_idx],target.loc[train_idx].values
        X_test, y_test = dat[test_idx],target.loc[test_idx].values
        print X_train.shape,X_test.shape
        
        y_train = np_utils.to_categorical(y_train,2)
        y_test = np_utils.to_categorical(y_test,2)
        model = nn()
        weight_path = 'weights.h5'
        callbacks = [
            ModelCheckpoint(weight_path, monitor='val_loss', save_best_only=True, verbose=1),
            EarlyStopping(monitor='val_loss', patience=10, verbose=0),
        ]
        if num_fold==1:
            print model.summary()
        model.fit(X_train, y_train, batch_size = 32,nb_epoch=50,
                      validation_data = (X_test, y_test),callbacks=callbacks)
        if os.path.isfile(weight_path):
            model.load_weights(weight_path)

        pred = model.predict(X_pred)
        predictions.append(pred)
    predictions = np.array(predictions)
    print val_loss
    np.save('X_pred_6.npy',predictions)
    val_loss = np.array(val_loss)
    #print np.mean(val_loss)


