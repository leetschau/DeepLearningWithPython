# 使用 `ipython --matplotlib=qt` 启动 IPython console，避免输入 `plt.show()`
# 为了每次只显示一张图，在下面标记出来的3行处加断点，例如：
# >>> run -d -b23 visual_cnn.py
# ipdb> b 32
# ipdb> b 44

from keras.models import load_model
from keras.preprocessing import image
from keras import models
import numpy as np
import matplotlib.pyplot as plt

model = load_model('dogs_vs_cats/cats_and_dogs_small_2.h5')
model.summary()

img_path = 'dogs_vs_cats/test/cats/cat.11350.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
print(img_tensor.shape)

plt.imshow(img_tensor[0])  # add breakpoint add this line，原始图片

layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]
print(first_layer_activation.shape)

plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')  # add breakpoint add this line
plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')  # 第一层的两个特征，与原始图片相似度较高

layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

# 每行16张图，包含16个featrue
images_per_row = 16

# 每个循环绘制一个layer中所有特征的激活图，共绘制8张图（8个layer）
for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]  # add breakpoint add this line

    size = layer_activation.shape[1]

    n_cols = n_features // images_per_row
    # 生成整个空白画布（二维数组）
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # 填充一列特征图
    for col in range(n_cols):
        # 填充一行特征图
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            # 填充一个特征图
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image

    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

# 第1轮输出图中包含2行16列共32张图，与第0层 conv2d_5 的输出形状
# (None, 148, 148, 32) 中的32个 feature 吻合
# 后续循环中，每张图中包含的子图越来越多（feature 越来越多），
# 每个子图越来越小，例如 conv2d_6（第3层）每个子图 72 * 72，共 64 张图，
# conv2d_8（第7层）每个子图 15 * 15，共 128 张图
