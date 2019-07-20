
#Import libraries
print('Importing needed libraries...')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.layers import Flatten, Dropout, Convolution1D
from keras.layers.embeddings import Embedding
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder

from keras.utils import np_utils

#Import data
print('Importing data...')
train_set = pd.read_csv('train_features.csv', sep = ',')
train_set_labels = pd.read_csv('train_labels.csv', sep = ',')
test_set = pd.read_csv('test_set.csv', sep = ',')
submission_df = pd.read_csv('SubmissionFormat.csv', sep = ',')

#Define train and Test sets
train_x = train_set.drop(["id","recorded_by","num_private","lga", "ward", "scheme_name", "subvillage",
                          "wpt_name", "funder", "installer", "amount_tsh", "quantity_group", "quality_group",
                          "waterpoint_type_group", "region", "extraction_type_class", "extraction_type_group",
                          "payment_type", "source_type", "management_group", "scheme_management"],axis=1)
train_y = train_set_labels.drop(["id"], axis=1)

test_x = test_set.drop(["id","recorded_by","num_private","lga", "ward", "scheme_name", "subvillage",
                          "wpt_name", "funder", "installer", "amount_tsh", "quantity_group", "quality_group",
                          "waterpoint_type_group", "region", "extraction_type_class", "extraction_type_group",
                          "payment_type", "source_type", "management_group", "scheme_management"], axis=1)

#Make two new variables: month and years_in_operation
#1. For train set
train_x['year_recorded'] = train_x['date_recorded'].str[:4].astype(int)
train_x['years_in_operation'] = train_x['year_recorded'] - train_x['construction_year']
med = train_x['years_in_operation'][train_x['years_in_operation']<2000].median()
row_index = train_x['years_in_operation'] > 2000
train_x.loc[row_index, 'years_in_operation'] = med

train_x['month_recorded'] = train_x['date_recorded'].str[5:7]

train_x = train_x.drop(['date_recorded', 'construction_year', 'year_recorded'], axis = 1)

#2. For test set
test_x['year_recorded'] = test_x['date_recorded'].str[:4].astype(int)
test_x['years_in_operation'] = test_x['year_recorded'] - test_x['construction_year']
med = test_x['years_in_operation'][test_x['years_in_operation']<2000].median()
row_index = test_x['years_in_operation'] > 2000
test_x.loc[row_index, 'years_in_operation'] = med

test_x['month_recorded'] = test_x['date_recorded'].str[5:7]

test_x = test_x.drop(['date_recorded', 'construction_year', 'year_recorded'], axis = 1)

#Feature Engineering
#1. Train set
row_index = train_x['quantity'] == 'unknown'
train_x.loc[row_index, 'quantity'] = 'dry'

med = train_x['gps_height'][train_x['gps_height'] != 0].median()
row_index = train_x['gps_height'] == 0
train_x.loc[row_index, 'gps_height'] = med

#2. Test set
row_index = test_x['quantity'] == 'unknown'
test_x.loc[row_index, 'quantity'] = 'dry'

med = test_x['gps_height'][test_x['gps_height'] != 0].median()
row_index = test_x['gps_height'] == 0
test_x.loc[row_index, 'gps_height'] = med


#------------------------Check for missing values-----------------------
train_x.isnull().sum()
test_x.isnull().sum()

train_x['permit'].value_counts()
train_x['public_meeting'].value_counts()


row_index = train_x['permit'].isnull()
train_x.loc[row_index, 'permit'] = 'Other'
row_index = train_x['public_meeting'].isnull()
train_x.loc[row_index, 'public_meeting'] = True

row_index = test_x['permit'].isnull()
test_x.loc[row_index, 'permit'] = 'Other'
row_index = test_x['public_meeting'].isnull()
test_x.loc[row_index, 'public_meeting'] = True


#---------------------------Convert to numeric------------------------------
#1. Train set
for col in train_x.columns:
    if train_x[col].dtype == object:
        train_x[col] = pd.Categorical(train_x[col], categories=train_x[col].unique()).codes

#2. Test set
for col in test_x.columns:
    if test_x[col].dtype == object:
        test_x[col] = pd.Categorical(test_x[col], categories=test_x[col].unique()).codes

#3. Train labels
train_y['status_group'] = train_y['status_group'].replace('functional', 1)
train_y['status_group'] = train_y['status_group'].replace('non functional', 2)
train_y['status_group'] = train_y['status_group'].replace('functional needs repair', 3)

#-----------------------------Convert to array------------------------------
train_x = np.array(train_x)
train_y = np.array(train_y)

encoder1 = LabelEncoder()
encoder1.fit(train_y)
encoded_train_Y = encoder1.transform(train_y)
dummy_train_y = np_utils.to_categorical(encoded_train_Y)
dummy_train_y.astype(int)

test_x = np.array(test_x)

#--------------------Build a Convolutional Neural Network model------------------------
from keras.layers.normalization import BatchNormalization
print('Building the best model in the world...')
model = Sequential()
model.add(Embedding(256, 16, input_length=19))
model.add(Convolution1D(16, 3, padding='same'))
model.add(Flatten())
model.add(BatchNormalization())
#model.add(Dense(8, activation="sigmoid"))
#model.add(BatchNormalization())
model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

#------------------------------Fit the model-----------------------------
print('Fitting the best model in the world...')
history = model.fit(train_x, dummy_train_y, validation_split=0.2, epochs=8, batch_size=16, verbose=2)

# Final evaluation of the model
test_pred = model.predict(test_x)
test_pred = np.argmax(test_pred, axis=1)
test_pred = test_pred.astype(str)
test_pred[test_pred=='0'] = 'functional'
test_pred[test_pred=='1'] = 'non functional'
test_pred[test_pred=='2'] = 'functional needs repair'

submission_df['status_group'] = test_pred

submission_df.to_csv('my_python_submission.csv',index = False)