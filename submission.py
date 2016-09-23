import pandas as pd
import numpy as np

idx = np.load('tid.npy')
pred = np.load('X_pred_6.npy')
print pred.shape
pred = np.mean(pred,axis=0)
pred = pred[:,1]

d = {'t_id':pd.Series(idx),
     'probability':pd.Series(pred)
     }

submit = pd.DataFrame(data=d,columns=['t_id','probability'])
submit.to_csv('submission52_valloss_nn.csv',index=False)
