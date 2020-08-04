from glob import glob
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_images = ''                           # Mention the train data folder
valid_images = ''                           # Mention the train data folder
folders = glob('')                          # Mention the train data folder
print("##### folders #####", len(folders))

input_shape=(250,250,3)
model = Sequential()
model.add(Conv2D(16, 3, activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(3))
model.add(Conv2D(32, 3, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, 3, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(train_images, target_size=(250,250), batch_size=32, class_mode='categorical')
valid_set = valid_datagen.flow_from_directory(valid_images, target_size=(250,250), batch_size=32, class_mode='categorical')


run_train = model.fit_generator(train_set,
                                validation_data=valid_set,
                                epochs=30,
                                steps_per_epoch=15,
                                validation_steps=15
                                )
model.save('emotifier.h5')
