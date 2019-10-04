# for section 6.3.1 ~ 6.3.7

import numpy as np
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from jena_load_data import float_data, data_gen

lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = data_gen(float_data, lookback=lookback, delay=delay,
        min_index=0, max_index=200000, shuffle=True,
        step=step, batch_size=batch_size)
val_gen = data_gen(float_data, lookback=lookback, delay=delay,
        min_index=200001, max_index=300000,
        step=step, batch_size=batch_size)
test_gen = data_gen(float_data, lookback=lookback, delay=delay,
        min_index=300001, max_index=None,
        step=step, batch_size=batch_size)

val_steps = (300000 - 200001 - lookback) // batch_size

test_steps = (len(float_data) - 300001 - lookback) // batch_size

# A common sense, non-machine learning baseline

def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        # 用取所有样本的最后一个时间点上的第2列（列名：t (degC)）数据作为预测值：
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    return np.mean(batch_maes)

print(evaluate_naive_method())

# A basic machine learning approach

model = Sequential()
model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=20,
        validation_data=val_gen, validation_steps=val_steps)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'ro', label='Double Dense MAE Training loss')
plt.plot(epochs, val_loss, 'b', label='Double Dense MAE Validation loss')
plt.title('Double Dense MAE Training and validation loss')
plt.legend()
plt.show()

# 实际运行结果，在 epoch 8 处，loss 和 validation loss 都出现了尖峰，
# 与书上曲线趋势不一致。

# A first recurrent baseline

model_gru = Sequential()
model_gru.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))
model_gru.add(layers.Dense(1))

model_gru.compile(optimizer=RMSprop(), loss='mae')
history = model_gru.fit_generator(train_gen, steps_per_epoch=500,
        epochs=8, validation_data=val_gen, validation_steps=val_steps)
# Dell 笔记本在 epoch 9 处卡死，这里将 epoch 数改成了8，
# 趋势已足够清楚

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'ro', label='GRU Training loss')
plt.plot(epochs, val_loss, 'b', label='GRU Validation loss')
plt.title('GRU Training and validation loss')
plt.legend()
plt.show()

# Using recurrent dropout to fight overfitting

model_dropout = Sequential()
model_dropout.add(layers.GRU(32, dropout=0.2, recurrent_dropout=0.2,
    input_shape=(None, float_data.shape[-1])))
model_dropout.add(layers.Dense(1))

model_dropout.compile(optimizer=RMSprop(), loss='mae')
history = model_dropout.fit_generator(train_gen, steps_per_epoch=500,
        epochs=8, validation_data=val_gen, validation_steps=val_steps)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'ro', label='Dropout Training loss')
plt.plot(epochs, val_loss, 'b', label='Dropout Validation loss')
plt.title('Dropout Training and validation loss')
plt.legend()
plt.show()

# Stacking recurrent layers

model_stack = Sequential()
model_stack.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5,
    return_sequences=True, input_shape=(None, float_data.shape[-1])))
model_stack.add(layers.GRU(64, activation='relu', dropout=0.1,
    recurrent_dropout=0.5))
model_stack.add(layers.Dense(1))
model_stack.compile(optimizer=RMSprop(), loss='mae')
history = model_stack.fit_generator(train_gen, steps_per_epoch=500,
        epochs=20, validation_data=val_gen, validation_steps=val_steps)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'ro', label='Stack Training loss')
plt.plot(epochs, val_loss, 'b', label='Stack Validation loss')
plt.title('Stack Training and validation loss')
plt.legend()
plt.show()

# code listing 6-44 训练并评估一个双向 GRU

model_bigru = Sequential()
model_bigru.add(layers.Bidirectional(layers.GRU(32),
    input_shape=(None, float_data.shape[-1])))
model_bigru.add(layers.Dense(1))

model_bigru.compile(optimizer=RMSprop(), loss='mae')
history = model_bigru.fit_generator(train_gen, steps_per_epoch=500, epochs=3,
        validation_data=val_gen, validation_steps=val_steps)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'ro', label='Bidirectional GRU MAE Training loss')
plt.plot(epochs, val_loss, 'b', label='Bidirectional GRU MAE Validation loss')
plt.title('Bidirectional GRU MAE Training and validation loss')
plt.legend()
plt.show()
