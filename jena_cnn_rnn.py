# for section 6.4.4

import numpy as np
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from jena_load_data import data_gen, float_data

lookback = 720
step = 3
delay = 144
batch_size = 128

train_gen = data_gen(float_data, lookback=lookback, delay=delay,
        min_index=0, max_index=200000, shuffle=True, step=step)
val_gen = data_gen(float_data, lookback=lookback, delay=delay,
        min_index=200001, max_index=300000, step=step)
test_gen = data_gen(float_data, lookback=lookback, delay=delay,
        min_index=300001, max_index=None, step=step)

val_steps = (300000 - 200001 - lookback) // batch_size
test_steps = (len(float_data) - 300001 - lookback) // batch_size

model = Sequential()
model.add(layers.Conv1D(32, 5, activation='relu',
    input_shape=(None, float_data.shape[-1])))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 5, activation='relu'))
model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5))
model.add(layers.Dense(1))
model.summary()

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
        steps_per_epoch=500,
        epochs=3,
        validation_data=val_gen,
        validation_steps=val_steps)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'ro', label='CNN-RNN MAE Training loss')
plt.plot(epochs, val_loss, 'b', label='CNN-RNN MAE Validation loss')
plt.title('CNN-RNN MAE Training and validation loss')
plt.legend()
plt.show()
