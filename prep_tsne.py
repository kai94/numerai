#
# preprocess data to feed it to bhtsne
#

import pandas as pd
import numpy as np

if __name__ == '__main__':
    train = pd.read_csv('data/week40/train_data.csv')
    test = pd.read_csv('data/week40/test_data.csv')
    valid = pd.read_csv('data/week40/valid_data.csv')
    
    print train.columns
    print test.columns
    train.drop(['target'],axis=1,inplace=True)
    valid.drop(['target'],axis=1,inplace=True)
    test.drop(['t_id','is_test'],axis=1,inplace=True)
    train = train.as_matrix()
    test = test.as_matrix()
    valid = valid.as_matrix()
    
    print 'Train data shape %s'%str(train.shape)
    print 'Valid data shape %s'%str(valid.shape)
    print 'Test data shape %s'%str(test.shape)
    
    data = np.vstack((train,valid,test))

    np.savetxt('data/week/dat.txt',data,fmt='%.10f',delimiter='\t')
