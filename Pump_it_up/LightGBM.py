
#---------------------------------------Import libraries-------------------------------
print('Importing needed libraries...')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import lightgbm as lgb
from sklearn.cluster import KMeans


#--------------------------------------Import data------------------------------------
print('Importing data...')
train_set = pd.read_csv('train_features.csv', sep = ',')
train_set_labels = pd.read_csv('train_labels.csv', sep = ',')
test_set = pd.read_csv('test_set.csv', sep = ',')
submission_df = pd.read_csv('SubmissionFormat.csv', sep = ',')

#-----------------------------Exploratory Data Analysis----------------------------
#Merge into one data set for data wrangling
train_set['is_train'] = 1
test_set['is_train'] = 0

train_test_set = pd.concat([train_set, test_set], axis=0, ignore_index=True)

train_test_set = train_test_set.drop(["id","recorded_by","num_private","lga", "ward", "scheme_name", "subvillage",
                          "wpt_name", "funder", "installer", "amount_tsh", "quantity_group", "quality_group",
                          "waterpoint_type_group", "region", "extraction_type_class", "extraction_type_group",
                          "payment_type", "source_type", "management_group", "scheme_management"], axis=1)

#Create Month variable
train_test_set['month_recorded'] = train_test_set['date_recorded'].str[5:7]
train_test_set['month_recorded'] = train_test_set['month_recorded'].astype(int)

#Create variable that counts years in operation for each pump
train_test_set['year_recorded'] = train_test_set['date_recorded'].str[:4]
train_test_set['year_recorded'] = train_test_set['year_recorded'].astype(int)
train_test_set['years_in_operation'] = 2014 - train_test_set['year_recorded']

#Delete date recorded and year recorded variables
train_test_set = train_test_set.drop(['year_recorded','date_recorded'], axis = 1)

#Replace zero values from 'longitude' with median
train_test_set['longitude'] = train_test_set['longitude'].replace(0, train_test_set['longitude'].mean(skipna=True))

'''
#Create geographical clusters from longitude and latitude
X = train_test_set.as_matrix(columns=['latitude', 'longitude'])
kmeans = KMeans(n_clusters=12, random_state=0).fit(X)
train_test_set['geo_cluster'] = kmeans.labels_
'''

#Make some vizualizations
train_set_w_labels= pd.concat([train_set,train_set_labels], axis = 1)
train_set_w_labels = train_set_w_labels.drop(["id"], axis=1)
'''
sns.violinplot("status_group", "years_in_operation", data=train_set_w_labels,
               palette=["lightblue", "lightpink", "orange"])

sns.barplot("permit", "years_in_operation", data=train_set_w_labels, hue = "status_group")
'''

train_set_w_labels['longitude'] = train_set_w_labels['longitude'].replace(0, train_set_w_labels['longitude'].mean(skipna=True))
sns.lmplot('longitude', # Horizontal axis
           'latitude', # Vertical axis
           data=train_set_w_labels, # Data source
           fit_reg=False, # Don't fix a regression line
           hue="status_group", # Set color
           scatter_kws={"marker": "D", # Set marker style
                        "s": 15}) # S marker size
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

#--------------------------Preprocessing and Creating a LightGBM model-------------------
#Convert categorical variables to numeric
print('Converting categorical variables to numeric...')
var_numeric = train_test_set.select_dtypes(include=['number']).copy()
var_non_numeric = train_test_set.select_dtypes(exclude=['number']).copy()

col_names = list(var_non_numeric)

for col in col_names:
    var_non_numeric[col] = var_non_numeric[col].astype('category', errors='ignore')

for col in col_names:
    var_non_numeric[col] = var_non_numeric[col].cat.codes

train_test_set= pd.concat([var_numeric,var_non_numeric], axis = 1)

#Split again in train and test sets
train_set = train_test_set.loc[train_test_set['is_train'] == 1]
test_set = train_test_set.loc[train_test_set['is_train'] == 0]

train_set = train_set.drop(['is_train'], axis = 1)
test_set = test_set.drop(['is_train'], axis = 1)

train_set_labels['status_group'] = train_set_labels['status_group'].astype('category')
labels = train_set_labels['status_group'].cat.codes

#Build LightGBM Model
train_data=lgb.Dataset(train_set,label=labels)

param = {'num_leaves': 24, 'objective':'multiclass', 'max_depth':10,
         'learning_rate':.08, 'num_class': 3, 'metric': 'multi_error',
         'feature_fraction': 0.8, 'zero_as_missing': True}

cv_mod = lgb.cv(param,
                train_data,
                num_boost_round = 1000,
                #min_data = 1,
                nfold = 5,
                early_stopping_rounds = 20,
                verbose_eval=100,
                stratified = True,
                show_stdv=True,
                )


num_boost_rounds_lgb = len(cv_mod['multi_error-mean'])

lgbm = lgb.train(param, train_data, num_boost_rounds_lgb)

ax = lgb.plot_importance(lgbm, max_num_features=21)
plt.show()

predictions = np.argmax(lgbm.predict(test_set), axis=1)

submission_df['status_group'][predictions==0] = 'functional'
submission_df['status_group'][predictions==2] = 'non functional'
submission_df['status_group'][predictions==1] = 'functional needs repair'

submission_df.to_csv('my_submission.csv', sep = ',',index = False)

