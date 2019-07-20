

import pandas as pd
import numpy as np
import sklearn.metrics as sm
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

#Import data
df = pd.read_csv('mckinsey.csv')
df = pd.read_excel('dataset.xls', header = 1)
df = df.rename(columns = {'default payment next month': 'Default'})

df_y = df.Default
df_x = df.drop(['ID','Default'], axis = 1)

model = KMeans(n_clusters = 3)

model.fit(df_x)

pred_y = model.labels_

#Change ones to zeros and vice-versa
pred_y[pred_y == 1] = 2
pred_y[pred_y == 0] = 1
pred_y[pred_y == 2] = 0


#Print Confusion matrix
conf_mat_lda = confusion_matrix(df_y,pred_y)
print('Confusion Matrix of Linear Discriminant Analysis:\n', conf_mat_lda)
