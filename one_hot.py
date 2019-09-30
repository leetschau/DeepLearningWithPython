import numpy as np
import string

# code listing 6.1

samples = ['The cat sat on the mat.', 'The dog ate my homework.', 'Some other new sentence.',
           'A very long and complex sentence whose length exceeds the max_length.']

token_index = {}

for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index) + 1

print('token_index:', token_index)
max_length = 10  # 每个 sample 中单词的最大长度，超出的10个单词的句子只截取前10个单词

results = np.zeros(shape=(len(samples),
                          max_length,
                          max(token_index.values()) + 1))

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:  # 这里通过 :max_length 实现了截取操作
        index = token_index.get(word)
        results[i, j, index] = 1.

print('shape of results:', results.shape)

# code listing 6.2

characters = string.printable
token_index = dict(zip(characters, range(1, len(characters) + 1)))
print('token_index:', token_index)

max_length = 50
results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
for i, sample in enumerate(samples):
    # print('sample:', sample)
    for j, character in enumerate(sample[:max_length]):
        index = token_index.get(character)
        # print('char and index: %s, %s' % (character, index))
        results[i, j, index] = 1.

print('shape of results:', results.shape)

# code listing 6.3

from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
one_hot_results.shape
# (4, 1000), 其中 4 表示 samples 包含4个元素，1000表示每个元素（句子）被编码为长度为1000的向量
# 编码只能反映一个句子包含哪些单词，丢掉了单词前后顺序信息

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# code listing 6.4

dimensionality = 1000  # 整个 samples 中单词的容量
max_length = 12        # 一个 sample 最多包含的单词数
results = np.zeros((len(samples), max_length, dimensionality))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = abs(hash(word)) % dimensionality
        results[i, j, index] = 1.
print(results.shape)
# (4, 12, 1000)
# 与 6.3 的实现方法相比，增加了一个维度，体现出了单词的顺序
