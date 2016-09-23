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
    input = Input(shape=(38,))
    out = Dense(600)(input)
    out = BatchNormalization()(out)
    out = ELU()(out)
    out = Dropout(0.5)(out)
    out = Dense(60)(out)
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
#dat = (dat - mean)/std
X_pred = pd.read_csv('numerai_tournament_data.csv')
tid = X_pred['t_id']
X_pred.drop(['t_id'],axis=1, inplace = True)
X_pred = X_pred.as_matrix()
#X_pred = (X_pred - mean)/std
n = len(dat)
print n
X = np.load('tsne_data_3d_perp5.npy').astype('float32')
tsnedat1 = X[:n,:]
tsne_X_pred1 = X[n:,:]
X = np.load('tsne_data_3d_perp10.npy').astype('float32')
tsnedat2 = X[:n,:]
tsne_X_pred2 = X[n:,:]
X = np.load('tsne_data_3d_perp15.npy').astype('float32')
tsnedat3 = X[:n,:]
tsne_X_pred3 = X[n:,:]
X = np.load('tsne_data_3d_perp20.npy').astype('float32')
tsnedat4 = X[:n,:]
tsne_X_pred4 = X[n:,:]
X = np.load('tsne_data_3d_perp30.npy').astype('float32')
tsnedat5 = X[:n,:]
tsne_X_pred5 = X[n:,:]
X = np.load('tsne_data_3d_perp40.npy').astype('float32')
tsnedat6 = X[:n,:]
tsne_X_pred6 = X[n:,:]
X = np.load('tsne_data_3d_perp50.npy').astype('float32')
tsnedat7 = X[:n,:]
tsne_X_pred7 = X[n:,:]
X = np.load('tsne_data_2d_perp30.npy').astype('float32')
tsnedat8 = X[:n,:]
tsne_X_pred8 = X[n:,:]
"""
doesn't improve
pca = np.load('pca10.npy')
datpca = pca[:n,:]
X_pred_pca = pca[n:,:]
"""
#tsnedat = np.hstack((tsnedat1,tsnedat2,tsnedat3,tsnedat4,tsnedat5,tsnedat6,tsnedat7))
#tsne_X_pred = np.hstack((tsne_X_pred1,tsne_X_pred2,tsne_X_pred3,tsne_X_pred4,tsne_X_pred5,tsne_X_pred6,tsne_X_pred7))

tsnedat = np.hstack((tsnedat2,tsnedat5,tsnedat7,tsnedat3,tsnedat4,tsnedat8))
tsne_X_pred = np.hstack((tsne_X_pred2,tsne_X_pred5,tsne_X_pred7,tsne_X_pred3,tsne_X_pred4,tsne_X_pred8))

"""
poly = PolynomialFeatures(2,interaction_only=True)
tsnedat = poly.fit_transform(tsnedat)
tsnedat = tsnedat[:,1:]
tsne_X_pred = poly.fit_transform(tsne_X_pred)
tsne_X_pred = tsne_X_pred[:,1:]
"""
dat = np.hstack((dat,tsnedat))
X_pred = np.hstack((X_pred,tsne_X_pred))

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
        model.fit(X_train, y_train, batch_size = 32,nb_epoch=80,
                      validation_data = (X_test, y_test),callbacks=callbacks)
        if os.path.isfile(weight_path):
            model.load_weights(weight_path)

        pred = model.predict(X_pred)
        np.save('pred%dfold'%num_fold,pred)
        predictions.append(pred)
    predictions = np.array(predictions)
    print val_loss
    np.save('X_pred_6.npy',predictions)
    val_loss = np.array(val_loss)
    #print np.mean(val_loss)


