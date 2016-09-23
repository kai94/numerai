import pandas as pd
import numpy as np

idx = np.load('tid.npy')
pred1 = pd.read_csv('submission52_valloss_nn.csv')
pred3 = pd.read_csv('submission51_068654_valloss_nn.csv')
pred5 = pd.read_csv('submission47_valloss_lda_bag.csv')
pred6 = pd.read_csv('submission45_valloss_lr_bag.csv')
pred7 = np.load('pred1fold.npy')[:,1]

gm = (pred1['probability']*pred3['probability']*pred7*pred5['probability'])**(1/4.)
am = (pred1['probability']*0.45+pred3['probability']*0.45+pred7*0.1)
d = {'t_id':pd.Series(idx),
     'probability':gm
     }

submit = pd.DataFrame(data=d,columns=['t_id','probability'])
submit.to_csv('submission62_average_gm.csv',index=False)
