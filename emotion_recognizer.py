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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from keras.utils import np_utils, to_categorical
import tensorflow.keras as keras

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
train = pd.concat([df_combined,test_asia]).drop_duplicates(keep=False)

train_asia.emotion.value_counts().plot(kind='bar')

X_train = train.iloc[:, 3:]
y_train = train.iloc[:,:1]
print(X_train.shape)

X_train_asia = train_asia.iloc[:, 3:]
y_train_asia = train_asia.iloc[:,:1]
print(X_train_asia.shape)

X_test_asia = test_asia.iloc[:,3:]
y_test_asia = test_asia.iloc[:,:1] #y_test = test.iloc[:,:2].drop(columns=['gender'])
print(X_test_asia.shape)

"""
Data Preprocessing
"""

def data_preprocessing(lb,X_train,X_test,y_train,y_test):
  mean = np.mean(X_train, axis=0)
  std = np.std(X_train, axis=0)
  X_train = (X_train - mean)/std
  X_test = (X_test - mean)/std
  X_train = np.array(X_train)
  y_train = np.array(y_train)
  X_test = np.array(X_test)
  y_test = np.array(y_test)
  y_train = to_categorical(lb.fit_transform(y_train))
  y_test = to_categorical(lb.fit_transform(y_test))
  return X_train,X_test,y_train,y_test

lb = LabelEncoder()
X_train,X_test,y_train,y_test = data_preprocessing(lb,X_train,X_test_asia,y_train,y_test_asia)
lb_asia = LabelEncoder()
X_train_asia,X_test_asia,y_train_asia,y_test_asia = data_preprocessing(lb_asia,X_train_asia,X_test_asia,y_train_asia,y_test_asia)

"""
Hyperparameter Tuning
"""

# RESHAPE DATA TO INCLUDE 3D TENSOR 
X_train = X_train[:,:,np.newaxis]
X_test = X_test[:,:,np.newaxis]

X_test_asia = X_test_asia[:,:,np.newaxis]
X_train_asia = X_train_asia[:,:,np.newaxis]

# CREATE FUNCTION FOR KERAS CLASSIFIER
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
    'batch_size': [30, 36, 42],  #multiple of 6 (num_classes)
    'nb_epoch': [25, 50, 75, 100],
    'optimizer':['adam','SGD']}

grid_search = GridSearchCV(estimator=classifier,
                           param_grid=params,
                           scoring='accuracy',
                           cv=5)

grid_search = grid_search.fit(X_train,y_trainHot)

dictionary = grid_search.best_params_
batch_size, epochs, optimizer = dictionary['batch_size'],dictionary['nb_epoch'],dictionary['optimizer']

# DO THE SAME FOR THE TRAINING SET WITH ONLY DATA FROM ONE RACE
opt = keras.optimizers.Adam(learning_rate=0.0001)
def make_classifier_asia(optimizer=opt):
    #BUILD CNN MODEL
    model = Sequential()
    model.add(layers.Conv1D(64, kernel_size=(10), activation='relu', input_shape=(X_train_asia.shape[1],1)))  #the shape is different than in the previous model
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

dictionary = grid_search.best_params_
batch_size_asia, epochs_asia, optimizer_asia = dictionary['batch_size'],dictionary['nb_epoch'],dictionary['optimizer']

"""
Model
"""

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

model = cnn_model(X_train,optimizer)
model_asia = cnn_model(X_train_asia,optimizer_asia)

# FIT MODEL AND USE CHECKPOINT TO SAVE BEST MODEL
checkpoint = ModelCheckpoint("best_initial_model.hdf5", monitor='val_accuracy', verbose=1,
    save_best_only=True, mode='max', period=1, save_weights_only=True)

model_history=model.fit(X_train, y_train,batch_size=batch_size, epochs=epochs, validation_data=(X_test_asia, y_test_asia),callbacks=[checkpoint])
model_history_asia=model_asia.fit(X_train_asia, y_train_asia,batch_size=batch_size_asia, epochs=epochs_asia, validation_data=(X_test_asia, y_test_asia),callbacks=[checkpoint])

"""
Post-Model Analysis
"""

# PRINT LOSS AND ACCURACY PERCENTAGE ON TEST SET
print("Loss of the model is - " , model.evaluate(X_test,y_test)[0])
print("Accuracy of the model is - " , model.evaluate(X_test,y_test)[1]*100 , "%")

print("Loss of the model is - " , model_asia.evaluate(X_test_asia,y_test_asia)[0])
print("Accuracy of the model is - " , model_asia.evaluate(X_test_asia,y_test_asia)[1]*100 , "%")

# CREATE CONFUSION MATRIX OF ACTUAL VS. PREDICTION
#change 'model_asia' by 'model' and 'X_test_asia' by 'X_test' to create the confusion matrix of the model trained with data from the three races
predictions = model_asia.predict(X_test_asia)
predictions=predictions.argmax(axis=1)
predictions = predictions.astype(int).flatten()
predictions = (lb_asia.inverse_transform((predictions)))
predictions = pd.DataFrame({'Predicted Values': predictions})

actual=y_test_asia.argmax(axis=1)
actual = actual.astype(int).flatten()
actual = (lb.inverse_transform((actual)))
actual = pd.DataFrame({'Actual Values': actual})
 
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
