
---
title: "AI BIZ image classification "
date: 2017-10-20 08:26:28 -0400
categories: jekyll update
---

캐글 커널을 활용하여 이미지 분류
https://www.kaggle.com/puneet6060/intel-image-classification

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.


FAST_RUN = False
IMAGE_WIDTH=150
IMAGE_HEIGHT=150
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3

foldernames=os.listdir("../input/intel-image-classification/seg_train/seg_train/")
categories=[]
imagefilename=[]


for foldername in foldernames:
    category = foldername.split('.')[0]
    if category == 'buildings':
        for imagefilenames in os.listdir("../input/intel-image-classification/seg_train/seg_train/"+category+r'/'):
            imagefilename.append(category+'/'+imagefilenames)
            categories.append(0)
    elif category == 'forest':
        for imagefilenames in os.listdir("../input/intel-image-classification/seg_train/seg_train/"+category+r'/'):
            imagefilename.append(category+'/'+imagefilenames)
            categories.append(1)
    elif category == 'glacier':
        for imagefilenames in os.listdir("../input/intel-image-classification/seg_train/seg_train/"+category+r'/'):
            imagefilename.append(category+'/'+imagefilenames)
            categories.append(2)
    elif category == 'mountain':
        for imagefilenames in os.listdir("../input/intel-image-classification/seg_train/seg_train/"+category+r'/'):
            imagefilename.append(category+'/'+imagefilenames)
            categories.append(3)
    elif category == 'sea':
        for imagefilenames in os.listdir("../input/intel-image-classification/seg_train/seg_train/"+category+r'/'):
            imagefilename.append(category+'/'+imagefilenames)
            categories.append(4)       
    else :
        for imagefilenames in os.listdir("../input/intel-image-classification/seg_train/seg_train/"+category+r'/'):
            imagefilename.append(category+'/'+imagefilenames)
            categories.append(5)

df = pd.DataFrame({
    'imagefilename': imagefilename,
    'category': categories
})

df['category'] = df['category'].astype('str')

sample = random.choice(imagefilename)
image = load_img("../input/intel-image-classification/seg_train/seg_train/"+sample)
plt.imshow(image)
print(sample)

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

earlystop = EarlyStopping(patience=10)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction]

train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=15

train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df,"../input/intel-image-classification/seg_train/seg_train/", 
    x_col='imagefilename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
   "../input/intel-image-classification/seg_train/seg_train/", 
    x_col='imagefilename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    "../input/intel-image-classification/seg_train/seg_train/", 
    x_col='imagefilename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical'
)

plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()

epochs=3 if FAST_RUN else 35
history = model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size)

model.save_weights("model.h5")
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(12,12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="Validation loss")
ax1.set_xticks(np.arange(1, epochs,1))
ax1.set_yticks(np.arange(0,1,0.1))

ax2.plot(history.history['acc'], color='b', label="Training accuracy")
ax2.plot(history.history['val_acc'], color='r', label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs,1))

legend = plt.legend(loc="best", shadow =True)
plt.tight_layout()
plt.show()

test_filenames = os.listdir("../input/intel-image-classification/seg_pred/seg_pred/")
test_df=pd.DataFrame({
    'filename' : test_filenames
})
nb_samples = test_df.shape[0]
test_gen = ImageDataGenerator(rescale = 1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df,
    "../input/intel-image-classification/seg_pred/seg_pred",
    x_col = 'filename',
    y_col = None, 
    class_mode= None,
    target_size = IMAGE_SIZE,
    batch_size = batch_size,
    shuffle = False
)

predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))

test_classes=[]
test_category=[]

for i in range(7301):
    test_classes.append(predict[i].tolist().index(max(predict[i].tolist())))
    if predict[i].tolist().index(max(predict[i].tolist()))==0:
        test_category.append('buildings')
    elif predict[i].tolist().index(max(predict[i].tolist()))==1:
        test_category.append('forest')
    elif predict[i].tolist().index(max(predict[i].tolist()))==2:
        test_category.append('glacier')
    elif predict[i].tolist().index(max(predict[i].tolist()))==3:
        test_category.append('mountain')
    elif predict[i].tolist().index(max(predict[i].tolist()))==4:
        test_category.append('sea')
    else :
        test_category.append('street')

final_df =pd.DataFrame({
'filename' : test_filenames,
'class' : test_classes,
'category' : test_category
})

final_df['category'].value_counts().plot.bar()


sample_test = final_df.head(18)
sample_test.head()
plt.figure(figsize=(12,24))
for index , row in sample_test.iterrows():
    filename=row['filename']
    category=row['category']
    img = load_img ( "../input/intel-image-classification/seg_pred/seg_pred/" +filename,
                target_size = IMAGE_SIZE)
    plt.subplot(6,3,index+1)
    plt.imshow(img)
    plt.xlabel(filename+'('+"{}".format(category)+')')
plt.tight_layout()
plt.show()
