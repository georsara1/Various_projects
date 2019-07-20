
#Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras import optimizers, regularizers
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc
from sklearn import preprocessing
sns.set_style("whitegrid")
np.random.seed(697)

#Import data
df = pd.read_excel('dataset.xls', header = 1)
df = df.rename(columns = {'default payment next month': 'Default'})

#---------------------------Vizualizations-----------------------------
#Distribution of variable 'Limit-Bal' versus Default
plt.figure()
fig1 = sns.violinplot(y = df.LIMIT_BAL, x = df.Default)
plt.show(fig1)

#Distribution of variable 'Age' versus Default
plt.figure()
fig2 = sns.violinplot(y = df.AGE, x = df.Default)
plt.show(fig2)

#Default depending on Education category and Sex
plt.figure()
fig3 = sns.barplot(x = df.EDUCATION, y = df.Default, hue = df.SEX)
plt.show(fig3)

#Default depending on Marriage category
plt.figure()
fig4 = sns.barplot(x = df.MARRIAGE, y = df.Default)
plt.show(fig4)

#---------------------------Pre-processing-------------------------
#Check for missing values
df.isnull().sum() #No missing values thus no imputations needed

#Drop unneeded variables
df = df.drop(['ID'], axis = 1)

#Encode categorical variables to ONE-HOT
print('Converting categorical variables to numeric...')

categorical_columns = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

df = pd.get_dummies(df, columns = categorical_columns,
                    #drop_first = True #Does not affect the algorithm's performance
                    )

#Scale variables to [0,1] range
columns_to_scale = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5'
    , 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

df[columns_to_scale]=df[columns_to_scale].apply(lambda x: (x-x.min())/(x.max()-x.min()))

#Split in 75% train and 25% test set
train_df, test_df = train_test_split(df, test_size = 0.25, random_state= 1984)

#Make sure labels are equally distributed in train and test set
train_df.Default.sum()/train_df.shape[0] #0.2233
test_df.Default.sum()/test_df.shape[0] #0.2148

train_all_ones = train_df[train_df.Default==1]
train_all_zeros = train_df[train_df.Default==0]
train_all_zeros2 = train_all_zeros.iloc[0:train_all_ones.shape[0]+1500,:]

train_final = pd.concat([train_all_ones,train_all_zeros2], axis = 0)
train_final = train_final.sample(frac=1)
train_final = train_final.reset_index(drop = True)

train_y = train_final.Default
test_y = test_df.Default

train_x = train_final.drop(['Default'], axis = 1)
test_x = test_df.drop(['Default'], axis = 1)

train_x =np.array(train_x)
test_x = np.array(test_x)

train_y = np.array(train_y)
test_y = np.array(test_y)

#-------------------Build the Neural Network model-------------------
print('Building Neural Network model...')
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = optimizers.adam(lr = 0.005, decay = 0.0000001)

model = Sequential()
model.add(Dense(32, input_dim=train_x.shape[1],
                kernel_initializer='normal',
                #kernel_regularizer=regularizers.l2(0.02),
                activation="relu"))
# model.add(Dropout(0.2))
# model.add(Dense(24,
#                 #kernel_regularizer=regularizers.l2(0.02),
#                 activation="tanh"))
# model.add(Dropout(0.3))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer='adam')

history = model.fit(train_x, train_y, validation_split=0.2, epochs=15, batch_size=64)

# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

#Predict on test set
predictions_NN_prob = model.predict(test_x)
predictions_NN_prob = predictions_NN_prob[:,0]

predictions_NN_01 = np.where(predictions_NN_prob > 0.5, 1, 0) #Turn probability to 0-1 binary output

#Print accuracy
acc_NN = accuracy_score(test_y, predictions_NN_01)
print('Overall accuracy of Neural Network model:', acc_NN)

#Print Area Under Curve
false_positive_rate, recall, thresholds = roc_curve(test_y, predictions_NN_prob)
roc_auc = auc(false_positive_rate, recall)
plt.figure()
plt.title('Receiver Operating Characteristic (ROC)')
plt.plot(false_positive_rate, recall, 'b', label = 'AUC = %0.3f' %roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out (1-Specificity)')
plt.show()

#Print Confusion Matrix
cm = confusion_matrix(test_y, predictions_NN_01)
labels = ['No Default', 'Default']
plt.figure(figsize=(8,6))
sns.heatmap(cm,xticklabels=labels, yticklabels=labels, annot=True, fmt='d', cmap="Blues", vmin = 0.2);
plt.title('Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()






