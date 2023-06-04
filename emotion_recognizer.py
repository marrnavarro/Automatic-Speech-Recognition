# IMPORT NECESSARY LIBRARIES
import resampy
import librosa
# %matplotlib inline
import matplotlib.pyplot as plt
import librosa.display
from IPython.display import Audio
import numpy as np
import tensorflow as tf
from matplotlib.pyplot import specgram
import pandas as pd
from sklearn.metrics import confusion_matrix
import IPython.display as ipd  # To play sound in the notebook
import os # interface with underlying OS that python is running on
import sys
import warnings
# ignore warnings 
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization, Dense
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

"""
Data Preprocessing
"""

!git lfs clone 'https://github.com/CheyneyComputerScience/CREMA-D.git'

# CREATE DIRECTORY OF AUDIO FILES 
audio = "CREMA-D/AudioWAV"
actor_folders = os.listdir(audio) #list files in audio directory
actor_folders.sort()

# CREATE FUNCTION TO EXTRACT EMOTION NUMBER, ACTOR AND GENDER LABEL
df = pd.DataFrame(columns=('emotion','actor','path'))
races = pd.DataFrame(columns=('actor','race'))
for i in actor_folders:
  actor=i[0:4]
  file_name='CREMA-D/AudioWAV/'+i
  emotion=i[9:12]
  df = df.append({'emotion':emotion, 'path':file_name, 'actor':str(actor)}, ignore_index=1)
r = pd.read_csv('CREMA-D/VideoDemographics.csv')
for a in range(len(r)):
  actor=r['ActorID'][a]
  race=r['Race'][a]
  races = races.append({'actor':str(actor),'race':race}, ignore_index=1)
audio_df = pd.merge(df,races)

audio_df_asia = audio_df[audio_df['race']=='Asia']
#to do the experiments for the other races, change 'Asia' for 'Caucasian' or 'African American'

audio_df = audio_df[audio_df['race']!='Unknown']

audio_df_asia = audio_df_asia.reset_index(drop=True)  #adding the mel spectogram values later won't work if we don't do this

audio_df = audio_df.reset_index(drop=True)  #adding the mel spectogram values later won't work if we don't do this

"""
Feature Extraction
"""

# ITERATE OVER ALL AUDIO FILES AND EXTRACT LOG MEL SPECTROGRAM MEAN VALUES INTO DF FOR MODELING 
def mel_spectrogram(dataframe):
  df = pd.DataFrame(columns=['mel_spectrogram'])
  counter = 0
  for index,path in enumerate(dataframe.path):
      X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=3,sr=44100,offset=0.5)
      
      #get the mel-scaled spectrogram (ransform both the y-axis (frequency) to log scale, and the “color” axis (amplitude) to Decibels, which is kinda the log scale of amplitudes.)
      spectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128,fmax=8000) 
      db_spec = librosa.power_to_db(spectrogram)
      #temporally average spectrogram
      log_spectrogram = np.mean(db_spec, axis = 0)
      
      df.loc[counter] = [log_spectrogram]
      counter=counter+1
  return df   

df = mel_spectrogram(audio_df)
df_asia = mel_spectrogram(audio_df_asia)

# TURN ARRAY INTO LIST AND JOIN WITH AUDIO_DF TO GET CORRESPONDING EMOTION LABELS
df_combined = pd.concat([audio_df,pd.DataFrame(df['mel_spectrogram'].values.tolist())],axis=1)
df_combined = df_combined.fillna(0)

df_combined_asia = pd.concat([audio_df_asia,pd.DataFrame(df_asia['mel_spectrogram'].values.tolist())],axis=1)
df_combined_asia = df_combined_asia.fillna(0)

# DROP PATH COLUMN FOR MODELING
df_combined.drop(columns='path',inplace=True)

df_combined_asia.drop(columns='path',inplace=True)

"""
Prepping Data for Modeling
"""

# TRAIN TEST SPLIT DATA
train_asia,test_asia = train_test_split(df_combined_asia, test_size=0.2, random_state=None,
                               stratify=None)
if first == True:
    train = pd.concat([df_combined,test_asia]).drop_duplicates(keep=False)
    
else:
    test_asia.index.values
test_asia = df_combined_asia.iloc[[ 940,  292,  563,  673,  540, 1386, 1250,  865, 1582, 1801,  232,
        141, 1093,  958, 1546, 1231, 1126,  229, 1608, 1351,  198,  758,
       1052,  773, 1583,    5, 1008,  488, 1157, 1536,  326, 1368,   18,
       1179,   69, 1167, 1115,   62, 1791,  263, 1161,  703, 1737,  791,
        414,  101, 1751, 1109, 1459,  982,   99, 1200,   16, 1399, 1148,
        360,  983,  626, 1592, 1705, 1521,  201,  790,  718,  545,  904,
        411,  435, 1275, 1482, 1545, 1268, 1571, 1191, 1601,  425,  600,
       1534, 1362,   64,  364, 1288,  374, 1045,  389,  212,   68,  636,
        412,  870, 1281,  202, 1038,  662,  907, 1795,  952, 1361,   77,
       1136, 1453,  371, 1303,  166, 1094, 1335, 1059,  149, 1622, 1042,
         24,  353,  302,  959, 1258,  113,  310,   44, 1717,  651,  564,
       1620,  689,  186,  255, 1746,  275,  993, 1208, 1234, 1416,   48,
        754,  355, 1332, 1600,  239, 1057,   93, 1340,  415,  591, 1498,
        121, 1137,   49, 1529, 1672,  887, 1759, 1603,  765,  195,  884,
       1341,  395,  335,  437,  246, 1402, 1031, 1420,  808,  519, 1302,
       1584, 1616,  847,   63, 1078, 1027,   40,  208,  237, 1437, 1377,
       1401, 1001,  487,  833,  769,  334,  296, 1541,  712, 1727,   73,
        681, 1193, 1016, 1646,  473, 1144, 1564,  590, 1796, 1479,   87,
       1711,  668, 1064,  829, 1319,  849, 1472, 1069,  510,  863, 1218,
        466,  686, 1563,   17, 1527, 1070,  165,  153, 1138,  630,  133,
        436, 1354, 1729, 1356,  937, 1630,  527, 1724,   72,  784,  362,
       1458,  380, 1708,  697,  517, 1162,  484,  112,  269,  361, 1253,
        279,  692,  511, 1049, 1654,  746,  708,  933,  161,  936, 1370,
        160, 1371, 1567,  934,   94,  979, 1422,  851, 1295, 1518,  612,
        792,   55, 1802, 1387, 1569,  611,  528, 1448,  683, 1428, 1750,
        472, 1009,  430, 1673,   45,  278,  420, 1095,  842, 1269,   36,
        756,  338,  613, 1079,  925, 1292, 1471,  890, 1722,  762, 1228,
       1506, 1279,  561,  408, 1554,  643,  939, 1744,   82,  655, 1390,
       1585,  778, 1773, 1252,  156,  710, 1696, 1146, 1124,  516, 1631,
        874, 1182, 1385, 1219, 1058,  800,  603,  178, 1315, 1020, 1142,
        546,  743, 1360,  702, 1046,  444, 1055, 1643, 1658,  881,  593,
       1666, 1522,  671, 1799, 1220,  844,   79,  898, 1374,  177, 1762,
       1512,   67,  854,  501,  582,   51, 1777,  974, 1462]
]
train_asia = pd.concat([df_combined_asia,test_asia]).drop_duplicates(keep=False)
train = pd.concat([df_combined,test_asia]).drop_duplicates(keep=False)

train_asia.emotion.value_counts().plot(kind='bar')

#test_asia.to_csv('sample_data/test_asia.csv')

#test_asia = pd.read_csv('sample_data/test_asia.csv',index_col=['Unnamed: 0'])

X_train = train.iloc[:, 3:]
y_train = train.iloc[:,:1] #y_train = train.iloc[:,:2].drop(columns=['gender']) así es como es en el código original
print(X_train.shape)

#MAR
X_train_asia = train_asia.iloc[:, 3:]
y_train_asia = train_asia.iloc[:,:1] #y_train = train.iloc[:,:2].drop(columns=['gender']) así es como es en el código original
print(X_train_asia.shape)

X_test = test.iloc[:,3:]
y_test = test.iloc[:,:1] #y_test = test.iloc[:,:2].drop(columns=['gender'])
print(X_test.shape)

#MAR
X_test_asia = test_asia.iloc[:,3:]
y_test_asia = test_asia.iloc[:,:1] #y_test = test.iloc[:,:2].drop(columns=['gender'])
print(X_test_asia.shape)

"""# Data Preprocessing"""

# NORMALIZE DATA
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean)/std
X_test = (X_test - mean)/std

# TURN DATA INTO ARRAYS FOR KERAS
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# ONE HOT ENCODE THE TARGET
# CNN REQUIRES INPUT AND OUTPUT ARE NUMBERS
lb = LabelEncoder()
y_train = to_categorical(lb.fit_transform(y_train))
y_test = to_categorical(lb.fit_transform(y_test))

print(y_test[0:10])

print(lb.classes_)

'''
# RESHAPE DATA TO INCLUDE 3D TENSOR 
X_train = X_train[:,:,np.newaxis]
X_test = X_test[:,:,np.newaxis]

X_train.shape
'''

#MAR
def data_preprocessing(lb,X_train,X_test,y_train,y_test):
  mean = np.mean(X_train, axis=0)
  std = np.std(X_train, axis=0)
  X_train = (X_train - mean)/std
  X_test = (X_test - mean)/std
  X_train = np.array(X_train)
  y_train = np.array(y_train)
  X_test = np.array(X_test)
  y_test = np.array(y_test)
  #lb = LabelEncoder()
  y_train = to_categorical(lb.fit_transform(y_train))
  y_test = to_categorical(lb.fit_transform(y_test))
  return X_train,X_test,y_train,y_test

lb = LabelEncoder()
X_train,X_test,y_train,y_test = data_preprocessing(lb,X_train,X_test_asia,y_train,y_test_asia) #change the name of X_test_asia and y_test_asia bc otherwise the next function won't work as we are passing X_test_asia and y_test_asia
lb_asia = LabelEncoder()
X_train_asia,X_test_asia,y_train_asia,y_test_asia = data_preprocessing(lb_asia,X_train_asia,X_test_asia,y_train_asia,y_test_asia)

X_test_asia.shape

"""## Base Model"""

X_train.shape

X_test.shape

import numpy as np
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier(strategy="stratified")
dummy_clf.fit(X_train, y_train)
DummyClassifier(strategy='stratified')
dummy_clf.predict(X_test)
dummy_clf.score(X_test, y_test)

from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
clf.predict(X_test)
clf.score(X_test, y_test)

"""## Initial Model"""

# RESHAPE DATA TO INCLUDE 3D TENSOR 
X_train = X_train[:,:,np.newaxis]
X_test = X_test[:,:,np.newaxis]

X_train.shape

X_train_asia.shape

#MAR
#X_train = X_train[:,:,np.newaxis]
X_test_asia = X_test_asia[:,:,np.newaxis]
X_train_asia = X_train_asia[:,:,np.newaxis]

#print(X_train.shape)
X_train_asia.shape
#if they are not the same, change the following cell code

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model

#BUILD 1D CNN LAYERS
def cnn_model(X_train,optimizer):
  model = tf.keras.Sequential()
  model.add(layers.Conv1D(64, kernel_size=(10), activation='relu', input_shape=(X_train.shape[1],1)))
  model.add(layers.Conv1D(128, kernel_size=(10),activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
  model.add(layers.MaxPooling1D(pool_size=(8)))
  model.add(layers.Dropout(0.4))
  model.add(layers.Conv1D(128, kernel_size=(10),activation='relu'))
  model.add(layers.MaxPooling1D(pool_size=(8)))
  model.add(layers.Dropout(0.4))
  model.add(layers.Flatten())
  model.add(layers.Dense(256, activation='relu'))
  model.add(layers.Dropout(0.4))
  model.add(layers.Dense(6, activation='sigmoid'))  #I changed '8' by '6' bc in our project we have 6 classes not 8
  if optimizer=='Adam': opt = keras.optimizers.Adam(learning_rate=0.001)
  elif optimizer=='SGD': opt = keras.optimizers.SGD(learning_rate=0.001)
  model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
  model.summary()
  return model

model = cnn_model(X_train,'Adam')
model_asia = cnn_model(X_train_asia,'SGD')

import tensorflow.keras as keras

# FIT MODEL AND USE CHECKPOINT TO SAVE BEST MODEL
checkpoint = ModelCheckpoint("best_initial_model.hdf5", monitor='val_accuracy', verbose=1,
    save_best_only=True, mode='max', period=1, save_weights_only=True)

#model_history=model.fit(X_train, y_train,batch_size=32, epochs=40, validation_data=(X_test, y_test),callbacks=[checkpoint])

model_history=model.fit(X_train, y_train,batch_size=30, epochs=75, validation_data=(X_test_asia, y_test_asia),callbacks=[checkpoint])

#MAR
model_history_asia=model_asia.fit(X_train_asia, y_train_asia,batch_size=30, epochs=50, validation_data=(X_test_asia, y_test_asia),callbacks=[checkpoint])

# PLOT MODEL HISTORY OF ACCURACY AND LOSS OVER EPOCHS
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('Initial_Model_Accuracy.png')
plt.show()
# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Initial_Model_loss.png')
plt.show()

#MAR
plt.plot(model_history_asia.history['accuracy'])
plt.plot(model_history_asia.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('Initial_Model_Accuracy_asia.png')
plt.show()
# summarize history for loss
plt.plot(model_history_asia.history['loss'])
plt.plot(model_history_asia.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Initial_Model_loss_asia.png')
plt.show()

"""## Post-Model Analysis"""

# PRINT LOSS AND ACCURACY PERCENTAGE ON TEST SET
print("Loss of the model is - " , model.evaluate(X_test,y_test)[0])
print("Accuracy of the model is - " , model.evaluate(X_test,y_test)[1]*100 , "%")

#MAR
print("Loss of the model is - " , model_asia.evaluate(X_test_asia,y_test_asia)[0])
print("Accuracy of the model is - " , model_asia.evaluate(X_test_asia,y_test_asia)[1]*100 , "%")

# PREDICTIONS
predictions = model.predict(X_test)
predictions=predictions.argmax(axis=1)
predictions = predictions.astype(int).flatten()
#lb = LabelEncoder()
predictions = (lb.inverse_transform((predictions)))
predictions = pd.DataFrame({'Predicted Values': predictions})

# ACTUAL LABELS
actual=y_test.argmax(axis=1)
actual = actual.astype(int).flatten()
actual = (lb.inverse_transform((actual)))
actual = pd.DataFrame({'Actual Values': actual})

# COMBINE BOTH 
finaldf = actual.join(predictions)
finaldf[140:150]

#MAR
predictions = model_asia.predict(X_test_asia)
predictions=predictions.argmax(axis=1)
predictions = predictions.astype(int).flatten()
print(predictions)
#lb = LabelEncoder()  #added by me, bc it appears before but I placed in inside a function
#y_train_asia = to_categorical(lb.fit_transform(y_train_asia))  #added by me
#y_test_asia = to_categorical(lb.fit_transform(y_test_asia))  #added by me, crec q no funciona pq a la funció y_test_asia canvia més abaix
#crec q si poso lb aquí, es crea un nou encoder i haig de tornar a fer y_train_asia, solució és no fer aqui lb treurel fora de la funció
predictions = (lb_asia.inverse_transform((predictions)))
predictions = pd.DataFrame({'Predicted Values': predictions})

# ACTUAL LABELS
actual=y_test_asia.argmax(axis=1)
actual = actual.astype(int).flatten()
actual = (lb.inverse_transform((actual)))
actual = pd.DataFrame({'Actual Values': actual})

# COMBINE BOTH 
finaldf = actual.join(predictions)
finaldf[140:150]

# CREATE CONFUSION MATRIX OF ACTUAL VS. PREDICTION 
cm = confusion_matrix(actual, predictions)
plt.figure(figsize = (12, 10))
cm = pd.DataFrame(cm , index = [i for i in lb.classes_] , columns = [i for i in lb.classes_])
ax = sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.title('Confusion Matrix', size=20)
plt.xlabel('Predicted Labels', size=14)
plt.ylabel('Actual Labels', size=14)
plt.savefig('Initial_Model_Confusion_Matrix.png')
plt.show()

predictions

print(classification_report(actual, predictions, target_names = ['ANG','DIS','FEA','HAP','NEU','SAD']))

"""## Hyperparameter Tuning"""

# TRAIN TEST SPLIT DATA
train,test = train_test_split(df_combined, test_size=0.2, random_state=0,
                               stratify=df_combined[['race','actor']])

X_train = train.iloc[:, 3:]
y_train = train.iloc[:,:1]#.drop(columns=['gender'])
print(X_train.shape)

X_test = test.iloc[:,3:]
y_test = test.iloc[:,:1]#.drop(columns=['gender'])
print(X_test.shape)

# NORMALIZE DATA
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean)/std
X_test = (X_test - mean)/std

# TURN DATA INTO ARRAYS FOR KERAS
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# RESHAPE TO INCLUDE 3D TENSOR
X_train = X_train[:,:,np.newaxis]
X_test = X_test[:,:,np.newaxis]

from keras.utils import np_utils, to_categorical

lb = LabelEncoder()

y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))

# CREATE FUNCTION FOR KERAS CLASSIFIER
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model

opt = keras.optimizers.Adam(learning_rate=0.0001)
def make_classifier(optimizer=opt):
    #BUILD CNN MODEL
    model = Sequential()
    model.add(layers.Conv1D(64, kernel_size=(10), activation='relu', input_shape=(X_train.shape[1],1)))
    model.add(layers.Conv1D(128, kernel_size=(10),activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(layers.MaxPooling1D(pool_size=(8)))
    model.add(layers.Dropout(0.4))
    model.add(layers.Conv1D(128, kernel_size=(10),activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=(8)))
    model.add(layers.Dropout(0.4))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(6, activation='sigmoid'))
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
    return model

y_trainHot=np.argmax(y_train, axis=1)
y_trainHot_asia=np.argmax(y_train_asia, axis=1)

# GRID SEARCH PARAMETERS TO FIND BEST VALUES
classifier = KerasClassifier(build_fn = make_classifier)
params = {
    'batch_size': [30, 36, 42],  #I changed [30, 32, 34] by [30, 36, 42] so that they are multiple of 6 (num_classes)
    'nb_epoch': [25, 50, 75, 100],  #I added 100
    'optimizer':['adam','SGD']}

grid_search = GridSearchCV(estimator=classifier,
                           param_grid=params,
                           scoring='accuracy',
                           cv=5)

grid_search = grid_search.fit(X_train,y_trainHot)

opt = keras.optimizers.Adam(learning_rate=0.0001)
def make_classifier_asia(optimizer=opt):
    #BUILD CNN MODEL
    model = Sequential()
    model.add(layers.Conv1D(64, kernel_size=(10), activation='relu', input_shape=(X_train_asia.shape[1],1)))
    model.add(layers.Conv1D(128, kernel_size=(10),activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(layers.MaxPooling1D(pool_size=(8)))
    model.add(layers.Dropout(0.4))
    model.add(layers.Conv1D(128, kernel_size=(10),activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=(8)))
    model.add(layers.Dropout(0.4))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(6, activation='sigmoid'))
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
    return model

classifier = KerasClassifier(build_fn = make_classifier_asia)
params = {
    'batch_size': [30, 36, 42],  #I changed [30, 32, 34] by [30, 36, 42] so that they are multiple of 6 (num_classes)
    'nb_epoch': [25, 50, 75, 100],  #I added 100
    'optimizer':['adam','SGD']}

grid_search = GridSearchCV(estimator=classifier,
                           param_grid=params,
                           scoring='accuracy',
                           cv=5)

grid_search = grid_search.fit(X_train_asia,y_trainHot_asia)

print(grid_search.best_params_)
grid_search.best_score_
