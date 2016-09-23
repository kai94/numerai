import numpy as np
import pandas as pd
import glob,os
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier,BaggingClassifier,VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, mean_squared_error,log_loss
from sklearn.cross_validation import train_test_split, KFold,cross_val_score
from sklearn.linear_model import LogisticRegression,Perceptron,SGDClassifier
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.svm import SVC,LinearSVC,libsvm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.mixture import GMM
import xgboost as xgb
from sklearn import cross_validation
from sklearn.feature_extraction import DictVectorizer
import scipy.stats as stats
import gc
import matplotlib.pylab as plt
from sklearn.grid_search import GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures,MinMaxScaler
from fastFM import als
from pyfm import pylibfm
from scipy.sparse import csc_matrix
from sklearn.pipeline import Pipeline
dtypes = {u'feature1':np.float32, u'feature2':np.float32, u'feature3':np.float32, u'feature4':np.float32, u'feature5':np.float32,
       u'feature6':np.float32, u'feature7':np.float32, u'feature8':np.float32, u'feature9':np.float32, u'feature10':np.float32,
       u'feature11':np.float32, u'feature12':np.float32, u'feature13':np.float32, u'feature14':np.float32, u'feature15':np.float32,
       u'feature16':np.float32, u'feature17':np.float32, u'feature18':np.float32, u'feature19':np.float32, u'feature20':np.float32,
          u'feature21':np.float32}

dat = pd.read_csv('numerai_training_data.csv',dtype=dtypes)

target = dat['target'].values
n = len(target)
for i in range(n):
    if target[i]==0:
        target[i]=-1
dat.drop(['target'],axis=1, inplace = True)
print dat.columns
X_pred = pd.read_csv('numerai_tournament_data.csv',dtype=dtypes)
tid = X_pred['t_id']
X_pred.drop(['t_id'],axis=1, inplace = True)
dat = dat.as_matrix()
X_pred = X_pred.as_matrix()
mean = np.mean(dat,axis=0)
std = np.std(dat,axis=0)
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
    num_fold=0
    predictions = []
    val_loss = []
    pipeline = Pipeline(steps=[
        ('poly', PolynomialFeatures(degree=2,interaction_only=True)),
        #('scaler', MinMaxScaler()),
    ])
    #dat = pipeline.fit(dat,target).transform(dat)
    #X_pred = pipeline.transform(X_pred)
    
    for train_idx,test_idx in kf:
        #fm
        """
        fmdata = [ {v: k for k, v in dict(zip(i, range(len(i)))).items()}  for i in dat]
        fmdata = np.array(fmdata)
        X_train,y_train = fmdata[train_idx],target[train_idx]
        X_test, y_test = fmdata[test_idx],target[test_idx]
        v = DictVectorizer()
        X_train = v.fit_transform(X_train)
        X_test = v.transform(X_test)
        """
        X_train,y_train = dat[train_idx],target[train_idx]
        X_test, y_test = dat[test_idx],target[test_idx]
        num_fold+=1
        params={'max_depth': [6],
            'subsample': [0.5,0.85,1],
            'colsample_bytree': [0.85,1.0],
            'learning_rate':[0.01,0.005,0.002],
            'objective':['binary:logistic'],
            'seed':[1440],
            'min_child_weight':[0.15,0.30,0.5,1],
            'n_estimators':[1000]
        }
        
        clf = xgb.XGBClassifier(max_depth=6,
                                   learning_rate=0.005, 
                                   n_estimators=1000, 
                                   silent=True, 
                                   objective='binary:logistic', 
                                   nthread=-1, 
                                   gamma=0,
                                   min_child_weight=0.15,
                                   max_delta_step=0, 
                                   subsample=0.5, 
                                   colsample_bytree=0.85,
                                   colsample_bylevel=1, 
                                   reg_alpha=0, 
                                   reg_lambda=1, 
                                   scale_pos_weight=1, 
                                   seed=1440, 
                                   missing=None)


        

        """
    clf = xgb.XGBClassifier()        
    gs = RandomizedSearchCV(clf,params,cv=5,scoring='log_loss',verbose=2,n_iter=10)
    gs.fit(dat,target)
    print gs.best_score_
    print gs.best_estimator_
    pred = gs.best_estimator_.predict_proba(X_pred)
    np.save('X_pred.npy',pred)
        """
        """
        res = clf.fit(X_train, y_train, eval_metric='logloss', verbose = True, 
                      eval_set = [(X_test, y_test)],early_stopping_rounds=200)
        xgb.plot_importance(clf)
        plt.savefig('feature_importance_xgb.png')
        val_loss.append(res.best_score)
        pred = clf.predict_proba(X_pred)
        predictions.append(pred)
        """
        #clf = GradientBoostingClassifier(learning_rate= 0.02,max_depth=6,subsample=0.8,random_state=random_state,max_features=0.8)
        #clf = BaggingClassifier(KNeighborsClassifier(n_neighbors=100),n_jobs=-1)
        #clf = ExtraTreesClassifier(n_estimators=100,max_depth=6)
        #clf = DecisionTreeClassifier(max_depth=6)
        #clf = RandomForestClassifier(n_estimators=100,max_depth=3)
        clf = KNeighborsClassifier(n_neighbors=100)
        #clf = BaggingClassifier(LogisticRegression(),n_estimators=100,max_features=0.8,max_samples=0.8,n_jobs=-1)
        #clf = BaggingClassifier(LinearDiscriminantAnalysis(),n_estimators=100,max_features=0.8,max_samples=0.8,n_jobs=-1)
        #clf = AdaBoostClassifier(learning_rate=0.05)
        #clf = LinearDiscriminantAnalysis()
        #clf  = LogisticRegression(penalty='l2',C=1.)
        #clf = als.FMClassification(n_iter=300, rank=8, l2_reg_w=1e-2, l2_reg_V=1e-2)
        #clf = SVC()
        
        if num_fold==1:
            print X_train.shape,X_test.shape
        #fm
        #csc_matrix
        clf.fit(X_train,y_train)
        loss = log_loss(y_test,clf.predict_proba(X_test))
        print loss
        val_loss.append(loss)
        pred = clf.predict_proba(X_pred)
        predictions.append(pred)
        
    predictions = np.array(predictions)
    print val_loss
    np.save('X_pred.npy',predictions)
    val_loss = np.array(val_loss)
    print np.mean(val_loss)


