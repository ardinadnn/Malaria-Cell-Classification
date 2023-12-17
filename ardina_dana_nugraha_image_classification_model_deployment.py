# -*- coding: utf-8 -*-
"""Ardina Dana Nugraha_Image Classification Model Deployment.ipynb

## **Belajar Pengembangan Machine Learning**
---
**Proyek Akhir - Image Classification Model Deployment**

**Nama**: Ardina Dana Nugraha

**Dataset yang digunakan**: Malaria Dataset, diperoleh dari https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

## **Import Modul**
"""

import os
from PIL import Image
import matplotlib.pyplot as plt
import splitfolders
import pandas as pd
import seaborn as sns
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras import models
import pathlib
import cv2
import os

"""## **Membaca Dataset**

Sea Animals Dataset dibaca terlebih dahulu.
"""

dataset = r"cell_images"

kelas = [class_name for class_name in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, class_name))]

banyak_citra=0
for class_name in kelas:
    class_path = os.path.join(dataset, class_name)
    image_files = [file for file in os.listdir(class_path) if file.lower().endswith(('.jpg','.png'))]
    number_of_images = len(image_files)
    banyak_citra=banyak_citra+number_of_images
print("Banyak citra =",banyak_citra)
print("\nDaftar kelas =",kelas,"\n\nBanyak kelas =", len(kelas))

def show_image_shapes(dataset_path):
    listUkuran = set()
    image_files = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]

    for image_file in image_files:
        image_path = os.path.join(dataset_path, image_file)
        image = cv2.imread(image_path)

        if image is not None:
            image_shape = image.shape
            listUkuran.add(image_shape)
    print(listUkuran,"\n")

    print("Kesimpulan".center(100,"─"))
    if (len(listUkuran)>1):
        print("Dataset memiliki ukuran resolusi yang berbeda-beda.")
    elif len(listUkuran==1):
        print("Dataset memiliki ukuran resolusi yang sama.")

dataset_path = dataset+"/Parasitized"
show_image_shapes(dataset_path)

"""Kode di atas menunjukkan bahwa dataset memiliki ukuran resolusi yang berbeda-beda.

## **Split Data Training dan Testing**
Dataset dibagi menjadi rasio training:testing 80:20.
"""

splitfolders.ratio(dataset, output="datasetSplit", seed=1307, ratio=(0.8, 0.0, 0.2))

IMAGE_SIZE = (100, 100, 3)
dataset="datasetSplit"
train_path = dataset+'/train'
test_path = dataset+'/test'

print(len(train_path))

quantity_train = {}
quantity_test = {}
for folder in os.listdir(train_path):
    quantity_train[folder] = len(os.listdir(train_path+'/'+folder))

for folder in os.listdir(test_path):
    quantity_test[folder] = len(os.listdir(test_path+'/'+folder))

quantity_train = pd.DataFrame(list(quantity_train.items()), index=range(0,len(quantity_train)), columns=['class','count'])
quantity_test = pd.DataFrame(list(quantity_test.items()), index=range(0,len(quantity_test)), columns=['class','count'])

figure, ax = plt.subplots(1,2,figsize=(20,5))
sns.barplot(x='class',y='count',data=quantity_train,ax=ax[0])
sns.barplot(x='class',y='count',data=quantity_test,ax=ax[1])

print("Banyak citra di training set:", sum(quantity_train['count'].values))
print("Banyak citra di testing set:",sum(quantity_test['count'].values))
print("Total citra:",sum(quantity_train['count'].values)+sum(quantity_test['count'].values))

plt.show()

def save_history(history, model_name):
    #convert the history.history dict to a pandas DataFrame:
    hist_df = pd.DataFrame(history.history)

    # save ke json
    hist_json_file = 'model_name'+'_history.json'
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)

    # or save ke csv
    hist_csv_file = 'model_name'+'_history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

def plot_accuracy_from_history(history, isinception=False):
  try:
    color = sns.color_palette()
    if(isinception == False):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
    else:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']


    epochs = range(len(acc))

    sns.lineplot(epochs, acc, label='Training Accuracy')
    sns.lineplot(epochs, val_acc,label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.figure()
    plt.show()
  except TypeError:
    pass

def plot_loss_from_history(history):
  try:
    color = sns.color_palette()
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    sns.lineplot(epochs, loss,label='Training Loss')
    sns.lineplot(epochs, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.figure()
    plt.show()
  except TypeError:
    pass

def do_history_stuff(i, history, history_file_name, isinception=False):
    save_history(i, history, history_file_name)
    plot_accuracy_from_history(history, isinception)
    plot_loss_from_history(history)

tf.keras.backend.clear_session()

"""## **Image Data Generator**"""

#normalisasi
train_datagen = ImageDataGenerator(rescale = 1.0/255.,shear_range=0.2,zoom_range=0.2)
train_generator = train_datagen.flow_from_directory(train_path,
                                                    batch_size=32,
                                                    shuffle=True,
                                                    class_mode='categorical',
                                                    target_size=(100, 100))

test_datagen = ImageDataGenerator(rescale = 1.0/255.,validation_split = 0.4)
test_generator = test_datagen.flow_from_directory(test_path, target_size=(100, 100),
    batch_size=32,
    shuffle=True,
    class_mode='categorical')

"""## **Model Training**"""

model = models.Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(2, activation='softmax'))

model.summary()

acc_thresh = 0.95

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if (logs.get('acc')>acc_thresh and logs.get('val_acc')>acc_thresh):
            print("Akurasi mencapai 95% yaitu",logs.get('acc'),"dan akurasi validasi sebesar",logs.get('val_acc'))
            self.model.stop_training = True

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
epoch = 100
model.compile(loss = 'categorical_crossentropy', optimizer= optimizer, metrics=['acc'])

filepath = 'saved-model-{epoch:02d}-acc-{val_acc:.2f}.hdf5'
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(train_generator, validation_data = test_generator, epochs=epoch, batch_size = 32, callbacks=[checkpoint,myCallback()], verbose=2)

model.save('model.h5')

"""## **Plot Akurasi dan Loss**"""

plt.style.use("ggplot")
plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="training")
plt.plot(history.history["val_loss"], label="validation")
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.figure(figsize=(10, 5))
plt.plot(history.history["acc"], label="training")
plt.plot(history.history["val_acc"], label="validation")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

"""## **Konversi ke TFLite**"""

export_dir = 'saved_model/'
tf.saved_model.save(model, export_dir)

# Convert SavedModel menjadi vegs.tflite
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
tflite_model = converter.convert()

tflite_model_file = pathlib.Path('vegs.tflite')
tflite_model_file.write_bytes(tflite_model)