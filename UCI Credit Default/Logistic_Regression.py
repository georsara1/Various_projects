
#Import modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc
import statsmodels.api as sm
sns.set_style("whitegrid")

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
df.isnull().sum() #No missing values

df = df.drop(['ID'], axis = 1)

#Encode categorical variables to ONE-HOT
print('Converting categorical variables to numeric...')

# One Hot encoding in Logistic Regression outputs 'Singular Matrix' error, so it is not used
# categorical_columns = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
#
# df = pd.get_dummies(df, columns = categorical_columns,
#                     drop_first = True #Better performance with n columns in One-Hot encoding
#                     )

#Scaling worsens the results. I left the code below in comment for reference purposes

# #Scale variables to [0,1] range
# columns_to_scale = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5'
#     , 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
#
# df[columns_to_scale]=df[columns_to_scale].apply(lambda x: (x-x.min())/(x.max()-x.min()))

#Split in 75% train and 25% test set
train_df, test_df = train_test_split(df, test_size = 0.25, random_state= 1984)

#Make sure labels are equally distributed in train and test set
train_df.Default.sum()/train_df.shape[0] #0.2233
test_df.Default.sum()/test_df.shape[0] #0.2148

train_y = train_df.Default
test_y = test_df.Default

train_x = train_df.drop(['Default'], axis = 1)
test_x = test_df.drop(['Default'], axis = 1)

#------------------------Build Logistic Regression Model-----------------------

logreg = LogisticRegression(class_weight= 'balanced')

rfe = RFE(logreg,18)
rfe = rfe.fit(train_x, train_y)

print(rfe.support_) #Maybe delete BILL_AMT3 to 6
print(rfe.ranking_)

train_x = train_x.drop(['BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6'], axis = 1)
test_x = test_x.drop(['BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6'], axis = 1)


logit_model = sm.Logit(train_y, train_x)
result = logit_model.fit()
print(result.summary())
logreg.fit(train_x, train_y)

#Predict on test set
predictions_LogReg_prob = logreg.predict_proba(test_x)
predictions_LogReg_prob = predictions_LogReg_prob[:,1]
predictions_LogReg_01 = logreg.predict(test_x)

#Print accuracy
acc_logit = accuracy_score(test_y, predictions_LogReg_01)
print('Overall accuracy of Logistic Regression model:', acc_logit)

#Print Area Under Curve
false_positive_rate, recall, thresholds = roc_curve(test_y, predictions_LogReg_prob)
roc_auc = auc(false_positive_rate, recall)
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
cm = confusion_matrix(test_y, predictions_LogReg_01)
labels = ['No Default', 'Default']
plt.figure(figsize=(8,6))
sns.heatmap(cm,xticklabels=labels, yticklabels=labels, annot=True, fmt='d', cmap="Blues", vmin = 0.2);
plt.title('Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()