# 本脚本实现了 5.3.1 节的 *不使用数据增强的快速特征提取* 算法

from keras.applications import VGG16
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from keras import models
from keras import layers
from keras import optimizers

import matplotlib.pyplot as plt

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150,150,3))

conv_base.summary()

train_dir = 'train'
validation_dir = 'validation'
test_dir = 'test'

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
            directory,
            target_size=(150,150),
            batch_size=batch_size,
            class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

# 得到卷积基模型，保存在 numpy.ndarray 中
train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

type(train_features)  # numpy.ndarray
train_features.shape  # (2000, 4, 4, 512)

train_features_res = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features_res = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features_res = np.reshape(test_features, (1000, 4 * 4 * 512))

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim= 4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(train_features_res, train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features_res, validation_labels))

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
# 这里 `val` 前缀表示 validation，所以 val_loss 表示 loss on validation set

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
