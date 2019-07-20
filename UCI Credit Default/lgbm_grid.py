
#Import modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc
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

df = df.drop(['ID','BILL_AMT4', 'BILL_AMT6'], axis = 1)

#Encode categorical variables to ONE-HOT
print('Converting categorical variables to numeric...')

categorical_columns = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

df = pd.get_dummies(df, columns = categorical_columns,
                    #drop_first = True #Slightly better performance with n columns in One-Hot encoding
                    )

#Scaling slightly worsened the results in Gradient Boosting (kept in comments below for reference purposes)
#
# #Scale variables to [0,1] range
# columns_to_scale = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT5'
#     , 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
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

#------------------------Build LightGBM Model-----------------------
train_data=lgb.Dataset(train_x,label=train_y)

#Select Hyper-Parameters
params = {'boosting_type': 'gbdt',
          'max_depth' : -1,
          'objective': 'binary',
          'nthread': 5, # Updated from nthread
          'num_leaves': 64,
          'learning_rate': 0.05,
          'max_bin': 512,
          'subsample_for_bin': 200,
          'subsample': 1,
          'subsample_freq': 1,
          'colsample_bytree': 0.8,
          'reg_alpha': 5,
          'reg_lambda': 10,
          'min_split_gain': 0.5,
          'min_child_weight': 1,
          'min_child_samples': 5,
          'scale_pos_weight': 1,
          'num_class' : 1,
          'metric' : 'binary_error'
          }

# Create parameters to search
gridParams = {
    'learning_rate': [0.005],
    'n_estimators': [8,16],
    'num_leaves': [16, 20, 24],
    'boosting_type' : ['gbdt'],
    'objective' : ['binary'],
    'random_state' : [501], # Updated from 'seed'
    'colsample_bytree' : [0.64, 0.65],
    'subsample' : [0.7,0.75],
    'reg_alpha' : [1, 1.2],
    'reg_lambda' : [ 1.2, 1.4],
    }

# Create classifier to use. Note that parameters have to be input manually, not as a dict!
mdl = lgb.LGBMClassifier(boosting_type= 'gbdt',
          objective = 'binary',
          n_jobs = 5, # Updated from 'nthread'
          silent = True,
          max_depth = params['max_depth'],
          max_bin = params['max_bin'],
          subsample_for_bin = params['subsample_for_bin'],
          subsample = params['subsample'],
          subsample_freq = params['subsample_freq'],
          min_split_gain = params['min_split_gain'],
          min_child_weight = params['min_child_weight'],
          min_child_samples = params['min_child_samples'],
          scale_pos_weight = params['scale_pos_weight'])

# To view the default model params:
mdl.get_params().keys()

# Create the grid
grid = GridSearchCV(mdl, gridParams, verbose=1, cv=4, n_jobs=-1)

# Run the grid
grid.fit(train_x, train_y)

# Print the best parameters found
print(grid.best_params_)
print(grid.best_score_)

# Using parameters already set above, replace in the best from the grid search
params['colsample_bytree'] = grid.best_params_['colsample_bytree']
params['learning_rate'] = grid.best_params_['learning_rate']
# params['max_bin'] = grid.best_params_['max_bin']
params['num_leaves'] = grid.best_params_['num_leaves']
params['reg_alpha'] = grid.best_params_['reg_alpha']
params['reg_lambda'] = grid.best_params_['reg_lambda']
params['subsample'] = grid.best_params_['subsample']
# params['subsample_for_bin'] = grid.best_params_['subsample_for_bin']

print('Fitting with params: ')
print(params)

#Train model on selected parameters and number of iterations
lgbm = lgb.train(params,
                 train_data,
                 2500,
                 #early_stopping_rounds= 40,
                 verbose_eval= 4
                 )

#Predict on test set
predictions_lgbm_prob = lgbm.predict(test_x)
predictions_lgbm_01 = np.where(predictions_lgbm_prob > 0.5, 1, 0) #Turn probability to 0-1 binary output

#--------------------------Print accuracy measures and variable importances----------------------
#Plot Variable Importances
lgb.plot_importance(lgbm, max_num_features=21, importance_type='split')

#Print accuracy
acc_lgbm = accuracy_score(test_y,predictions_lgbm_01)
print('Overall accuracy of Light GBM model:', acc_lgbm)

#Print Area Under Curve
plt.figure()
false_positive_rate, recall, thresholds = roc_curve(test_y, predictions_lgbm_prob)
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
plt.figure()
cm = confusion_matrix(test_y, predictions_lgbm_01)
labels = ['No Default', 'Default']
plt.figure(figsize=(8,6))
sns.heatmap(cm, xticklabels = labels, yticklabels = labels, annot = True, fmt='d', cmap="Blues", vmin = 0.2);
plt.title('Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()

