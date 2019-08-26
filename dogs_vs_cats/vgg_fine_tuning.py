# 本脚本实现了 5.3.2 节微调模型算法

from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers

from keras.applications import VGG16
import matplotlib.pyplot as plt

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150,150,3))
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# --- 向 vgg_extended.py 中增加新代码开始

conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

# --- 增加代码结束 ---

train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_dir = 'train'
validation_dir = 'validation'
test_dir = 'test'

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])

history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,    # 一个 epoch 包含100个 steps
        epochs=100,
        validation_data=validation_generator,
        validation_steps=50)

# step 和 epoch 的区别和联系：
# 一个 step 在一个 batch （包含 batch_size 个 sample，这里是 20） 上进行一次梯度下降计算；
# 100 个 steps 处理 100 个 batch，共 2000 个 sample，完成一个 epoch（见书 p42 对 epoch 的定义），
# 所以 总 sample 数量 = batch_size * steps_per_epoch，即 2000 = 20 * 100，
# 这里总 sample 数量就是 train 文件夹下总图片文件数

# 问题：如果 batch_size * steps_per_epoch 超过了总 sample 数，会出现什么情况？

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

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
